#include "deblur.hpp"
#include "hls_fence.h"
#include <iostream>

// -----------------------------------------------------------------------------
// Helper macro for clean buffer selection (compile-time resolvable)
// -----------------------------------------------------------------------------
#define SELECT_Z(buf_i, idx, role) \
    ((role) == 0 ? ((buf_i) == 0 ? Z1[idx] : ((buf_i) == 1 ? Z2[idx] : Z0[idx])) : \
     (role) == 1 ? ((buf_i) == 0 ? Z2[idx] : ((buf_i) == 1 ? Z0[idx] : Z1[idx])) : \
                   ((buf_i) == 0 ? Z0[idx] : ((buf_i) == 1 ? Z1[idx] : Z2[idx])))

// role = 0 → v_prev
// role = 1 → v_cur
// role = 2 → v_next

// -----------------------------------------------------------------------------c:\Users\Bram\Documents\Git\DeepWaveAccel\Hardware\vitis\DeepWaveAccel\theta_data.hpp
// Bit-pack helpers: convert to unified 32-bit AXIS payload
// -----------------------------------------------------------------------------
template <typename T>
static inline word_t pack_word(T x) {
#pragma HLS INLINE
    word_t out;
    out.range() = x.range();
    return out;
}

// -----------------------------------------------------------------------------
// Sequential sparse Laplacian multiply (center + ND off-diagonals)
// -----------------------------------------------------------------------------
static inline acc_t lap_pixel_seq(
    const img_t  Z0[IMG_LEN],
    const img_t  Z1[IMG_LEN],
    const img_t  Z2[IMG_LEN],
    int           buf_i,    // k % 3
    idx_t         i,
    const idx_t   offs[ND],
    const lap_t   lap_rest[ND][IMG_LEN],
    acc_t         acc_in,
    int           d)
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=offs complete

    acc_t acc = acc_in;

    int o  = (int)offs[d];
    int il = (int)i - o;
    int iu = (int)i + o;

    if (il >= 0)
        acc -= (acc_t)lap_rest[d][i] * (acc_t)SELECT_Z(buf_i, il, 1);
    if (iu < IMG_LEN)
        acc -= (acc_t)lap_rest[d][iu] * (acc_t)SELECT_Z(buf_i, iu, 1);

    return acc;
}

// -----------------------------------------------------------------------------
// Deblur kernel (with global preloaded Laplacian + per-frame norm prepend)
// Output order per frame:   [ norm ] , then IMG_LEN pixelsc:\Users\Bram\Documents\Git\DeepWaveAccel\Hardware\vitis\DeepWaveAccel\laplacian_data.hpp
// -----------------------------------------------------------------------------
void deblur(
    hls::stream<img_t>      &bp_stream,
    hls::stream<word_t> &param_in,
    hls::stream<out_axis_t> &out_stream,
    hls::stream<norm_sum_t> &norm_stream,
    const deblur_config           &cfg,
    status_db_t             &status)
{
    AP_CTRL_NONE;
    AXIS_IN_OUT(bp_stream);
    AXIS_IN_OUT(param_in);
    AXIS_IN_OUT(out_stream);
    AXIL_CFG(cfg);
    AXIL_CFG(status);
    // No loop-level PIPELINE on whole function; intra-stage pipelining is used.

    // ---------------- Persistent memories ----------------
    static lap_t lap_main;
    static lap_t lap_rest[ND][IMG_LEN];
    static theta_t theta[MAX_ORDER];
    static idx_t lap_offsets[ND];
    static ap_uint<8> K;

#pragma HLS BIND_STORAGE variable=theta type=ram_1p impl=auto
#pragma HLS ARRAY_PARTITION variable=lap_offsets complete
#pragma HLS BIND_STORAGE variable=lap_rest type=ram_2p impl=bram
// #pragma HLS ARRAY_PARTITION variable=lap_rest dim=1 complete

    static img_t bp_buf[IMG_LEN];
    static acc_t y_acc[IMG_LEN];
#pragma HLS BIND_STORAGE variable=bp_buf type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=y_acc  type=ram_2p impl=bram
#pragma HLS DEPENDENCE variable=y_acc inter false
#pragma HLS DEPENDENCE variable=bp_buf inter false

    // ---------------- Triple buffers ----------------
    static img_t Z0[IMG_LEN];
    static img_t Z1[IMG_LEN];
    static img_t Z2[IMG_LEN];
#pragma HLS BIND_STORAGE variable=Z0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=Z1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=Z2 type=ram_2p impl=bram

    // ---------------- FSM ----------------
    enum St {
        LOAD_PARAMS,
        LOAD_BP,
        CHEB_Y0,
        CHEB_COMPUTE,
        LAYER_ADD_BP,
        OUTPUT
    };
    enum StL { // Loading states   
        READ_K,
        READ_THETA,
        READ_OFFSETS,
        READ_MAIN,
        READ_REST
    };
    static St st = LOAD_PARAMS;
    static StL stl = READ_K;

    // static bool frame_sent = false;

    static int   k = 1;
    static acc_t acc_tmp = 0;
    static acc_t y_tmp   = 0;
    static bool  config_loaded = false;

    static word_t pixels_in = 0;
    static word_t pixels_out = 0;

    switch (st)
    {
    case LOAD_PARAMS: {
        if (!param_in.empty()){
            ap_uint<32> w = param_in.read();

            // We can only receive in the following order: K(1), theta(K+1), lap_offsets(ND), lap_main(1), lap_rest(ND)(2234)
            switch (stl) {
                case READ_K: {
                    K = w;
                    config_loaded = false; // Reset, we are reloading everything
                    status.config_loaded = false;
                    stl = READ_THETA;
                    break;
                }
                case READ_THETA: {
                    static int i = 0;
                    theta[i].range() = w.range();
                    i++;
                    if (i > K) {
                        i = 0;
                        stl = READ_OFFSETS;
                    }
                    break;
                }
                case READ_OFFSETS: {
                    static int i = 0;
                    lap_offsets[i] = w;
                    i++;
                    if (i == ND) {
                        i = 0;
                        stl = READ_MAIN;
                    }
                    break;
                }
                case READ_MAIN: {
                    static int i = 0;
                    lap_main.range() = w.range();
                    stl = READ_REST;
                    break;
                }
                case READ_REST: {
                    static int i = 0;
                    static int d = 0;
                    lap_rest[d][i].range() = w.range();
                    i++;
                    if (i == IMG_LEN) {
                        i = 0;
                        d++;
                        if (d == ND) {
                            d = 0;
                            config_loaded = true;
                            status.config_loaded = true;
                            stl = READ_K;
                            // Do not necessarily go straight to LOAD_BP, maybe we'd like to reload params. Just stay in LOAD_PARAMS and go to LOAD_BP when bp_stream is no longer empty
                        }
                        break;
                    }
                }

            }
            status.param_state = stl;
        } else if (config_loaded && !bp_stream.empty()) { // We can go straight to LOAD_BP if we have already loaded all params and there is data available
            st = LOAD_BP;
        }
        break;
    }

    case LOAD_BP: {
        static int i = 0;
        if (!bp_stream.empty()) {
            bp_buf[i] = bp_stream.read();
            status.idx = i;
            status.pixels_in = ++pixels_in;
            
            Z0[i] = (img_t)0;
            y_acc[i] = (acc_t)0;
            ++i;
            if (i == IMG_LEN) {
                i = 0;
                st = CHEB_COMPUTE; // skip CHEB_Y0 (θ0 applied in loop)
            }
        }
        break;
    }

    case CHEB_Y0: {
        static int i = 0;
        // Optional: initial y = θ0 * Z0 (if you prefer explicit first-step)
        y_acc[i] = (acc_t)theta[0] * (acc_t)Z0[i];
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            st = CHEB_COMPUTE;
        }
        break;
    }

    // ---------------- Chebyshev compute ----------------
    case CHEB_COMPUTE: {
        static int i = 0;
        static bool centre = true;
        static int buf_i = 1;

        if (centre) {
            img_t center_val = SELECT_Z(buf_i, i, 1);
            acc_tmp = (acc_t)lap_main * (acc_t)center_val;
            y_tmp   = y_acc[i];
            centre = false;
        }
        else {
            static int d = 0;
            acc_tmp = lap_pixel_seq(Z0, Z1, Z2, buf_i, i, lap_offsets, lap_rest, acc_tmp, d);

            if (d == ND-1) {
                d = 0;
                centre = true;
                img_t t = (img_t)acc_tmp;

                img_t z_prev_val = SELECT_Z(buf_i, i, 0);
                img_t z_next_val = (k == 1) ? t
                                            : (img_t)((t<<1) - z_prev_val);

                // Rotate triple buffer
                switch (buf_i) {
                case 0: Z0[i] = z_next_val; break;
                case 1: Z1[i] = z_next_val; break;
                default:Z2[i] = z_next_val; break;
                }

                // Accumulate Chebyshev weighted sum
                y_acc[i] = y_tmp + (acc_t)theta[k] * (acc_t)z_next_val;
                ++i;
                if (i == IMG_LEN) {
                    i = 0; 
                    ++k;
                    ++buf_i;
                    if (buf_i == 3) buf_i = 0;  // wrap every 3 iterations

                    if (k <= K) st = CHEB_COMPUTE;
                    else {           
                        st = LAYER_ADD_BP;
                        k = 1;
                        buf_i = 1;  // reset for next frame/layer (optional)
                    }
                }
            } else {
                ++d;
            }
        }
        break;
    }

    case LAYER_ADD_BP: {
        static int i = 0;
        static int layer = 0;
        img_t t = (img_t)(y_acc[i] + (acc_t)bp_buf[i]);
        if (t < 0) t = 0;
        Z0[i] = t;
        ++i;
        if (i == IMG_LEN) {
            i = 0; ++layer;
            if (layer < (int)cfg.n_layers) {
                st = CHEB_Y0;
            } else {
                layer = 0;
                st = OUTPUT;
            }
        }
        break;
    }

    case OUTPUT: {
#ifdef __SYNTHESIS__           // We output one frame twice, only in the actual kernel. Stupid AXIS meuk
#define N_REPEAT 2
#else
#define N_REPEAT 1
#endif
        static int i_out = 0;
        static int frames_out = 0;
        static int repeat = 0;
        static word_t norm;
        static bool norm_valid = false;
        out_axis_t out;

        // First read norm, once a frame
        if (!norm_valid){
            norm = pack_word((norm_out_t)norm_stream.read()); // Pack norm in norm_out_t format
            norm_valid = true;
        }

        hls::fence({norm},{out});
        if (i_out==0) {
            // First write norm
            out.data = ((repeat & 0b1) <<18) | norm;
        } else {
            // Otherwise write pixels
            out.data = (i_out<<19) | ((repeat & 0b1) <<18) | pack_word(Z0[i_out-1]);
            status.pixels_out = ++pixels_out;
        }
        out.last = (i_out==IMG_LEN) && (repeat == N_REPEAT-1);
        hls::fence({out},{out_stream});
        out_stream.write(out);
        hls::fence({out_stream},{i_out});
        status.idx = i_out;

        ++i_out;
        if (i_out == IMG_LEN + 1) {
            i_out = 0;
            repeat++;
            if (repeat == N_REPEAT) { // 2 passes (issues with skipping first ~147 values couldn't be solved...)
                repeat = 0;
                norm_valid = false;
                frames_out++;
                st = LOAD_PARAMS;   // next frame
            }
        }
        break;
    }
    } // switch
    status.fsm_state = st;
}

#undef SELECT_Z
