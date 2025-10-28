#include "deblur.hpp"
#include <iostream>

// -----------------------------------------------------------------------------
// Helper macro for clean buffer selection (compile-time resolvable)
// -----------------------------------------------------------------------------
#define SELECT_Z(k_mod, idx, role) \
    ((role) == 0 ? ((k_mod) == 0 ? Z1[idx] : ((k_mod) == 1 ? Z2[idx] : Z0[idx])) : \
     (role) == 1 ? ((k_mod) == 0 ? Z2[idx] : ((k_mod) == 1 ? Z0[idx] : Z1[idx])) : \
                   ((k_mod) == 0 ? Z0[idx] : ((k_mod) == 1 ? Z1[idx] : Z2[idx])))

// role = 0 → v_prev
// role = 1 → v_cur
// role = 2 → v_next

// -----------------------------------------------------------------------------
// Bit-pack helpers: convert to unified 32-bit AXIS payload
// -----------------------------------------------------------------------------
template <typename T>
static inline out_axis_t pack_word(T x) {
#pragma HLS INLINE
    return (out_axis_t)x.range();
}

// -----------------------------------------------------------------------------
// Sequential sparse Laplacian multiply (center + ND off-diagonals)
// -----------------------------------------------------------------------------
static inline acc_t lap_pixel_seq(
    const img_t  Z0[IMG_LEN],
    const img_t  Z1[IMG_LEN],
    const img_t  Z2[IMG_LEN],
    int           k_mod,    // k % 3
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
        acc -= (acc_t)lap_rest[d][i] * (acc_t)SELECT_Z(k_mod, il, 1);
    if (iu < IMG_LEN)
        acc -= (acc_t)lap_rest[d][iu] * (acc_t)SELECT_Z(k_mod, iu, 1);

    return acc;
}

// -----------------------------------------------------------------------------
// Deblur kernel (with global preloaded Laplacian + per-frame norm prepend)
// Output order per frame:   [ norm ] , then IMG_LEN pixels
// -----------------------------------------------------------------------------
void deblur(
    hls::stream<img_t>      &bp_stream,
    lap_t                    lap_main,
    const lap_t              lap_rest[ND][IMG_LEN],
    hls::stream<out_axis_t> &out_stream,
    hls::stream<norm_sum_t> &norm_stream,
    deblur_config           &cfg)
{
    AXIS_IN_OUT(bp_stream);
    AXIS_IN_OUT(out_stream);
    AXIL_CFG(cfg);
    AP_CTRL_NONE;
    // No loop-level PIPELINE on whole function; intra-stage pipelining is used.

    // ---------------- Persistent memories ----------------
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

#pragma HLS ARRAY_PARTITION variable=cfg.lap_off complete
#pragma HLS ARRAY_PARTITION variable=cfg.theta   complete

    // ---------------- FSM ----------------
    enum St {
        LOAD_BP,
        CHEB_Y0,
        CHEB_COMPUTE,
        LAYER_ADD_BP,
        OUTPUT
    };
    static St st = LOAD_BP;

    static idx_t i = 0;
    static int   d = 0;
    static int   k = 0;
    static int   layer = 0;
    static acc_t acc_tmp = 0;
    static bool  centre  = true;
    static acc_t y_tmp   = 0;

    switch (st)
    {
    case LOAD_BP: {
        if (!bp_stream.empty()) {
            bp_buf[i] = bp_stream.read();
            Z0[i] = (img_t)0;
            y_acc[i] = (acc_t)0;
            ++i;
            if (i == IMG_LEN) {
                i = 0; layer = 0; k = 1;
                centre = true; d = 0;
                st = CHEB_COMPUTE; // skip CHEB_Y0 (θ0 applied in loop)
            }
        }
        break;
    }

    case CHEB_Y0: {
        // Optional: initial y = θ0 * Z0 (if you prefer explicit first-step)
        y_acc[i] = (acc_t)cfg.theta[0] * (acc_t)Z0[i];
        ++i;
        if (i == IMG_LEN) {
            i = 0; k = 1; d = 0; centre = true;
            st = CHEB_COMPUTE;
        }
        break;
    }

    // ---------------- Chebyshev compute ----------------
    case CHEB_COMPUTE: {
        int k_mod = k % 3;

        if (centre) {
            img_t center_val = SELECT_Z(k_mod, i, 1);
            acc_tmp = (acc_t)lap_main * (acc_t)center_val;
            y_tmp   = y_acc[i];
            d = 0;
            centre = false;
        }
        else {
            acc_tmp = lap_pixel_seq(Z0, Z1, Z2, k_mod, i, cfg.lap_off, lap_rest, acc_tmp, d);

            if (d == ND-1) {
                centre = true;
                img_t t = (img_t)acc_tmp;

                img_t z_prev_val = SELECT_Z(k_mod, i, 0);
                img_t z_next_val = (k == 1) ? t
                                            : (img_t)((t<<1) - z_prev_val);

                // Rotate triple buffer
                switch (k_mod) {
                case 0: Z0[i] = z_next_val; break;
                case 1: Z1[i] = z_next_val; break;
                default:Z2[i] = z_next_val; break;
                }

                // Accumulate Chebyshev weighted sum
                y_acc[i] = y_tmp + (acc_t)cfg.theta[k] * (acc_t)z_next_val;
                ++i;
                if (i == IMG_LEN) {
                    i = 0; ++k;
                    if (k <= cfg.K) st = CHEB_COMPUTE;
                    else            st = LAYER_ADD_BP;
                }
            } else {
                ++d;
            }
        }
        break;
    }

    case LAYER_ADD_BP: {
        img_t t = (img_t)(y_acc[i] + (acc_t)bp_buf[i]);
        if (t < 0) t = 0;
        Z0[i] = t;
        ++i;
        if (i == IMG_LEN) {
            i = 0; ++layer;
            if (layer < (int)cfg.n_layers) {
                st = CHEB_Y0;
                k  = 1; d = 0; centre = true;
            } else {
                // before OUTPUT, fetch normalization and send it
                norm_sum_t nv = norm_stream.read();
                out_stream.write(pack_word(nv));
                st = OUTPUT;
            }
        }
        break;
    }

    case OUTPUT: {
        out_stream.write(pack_word(Z0[i]));
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            st = LOAD_BP;   // next frame
        }
        break;
    }
    } // switch
}

#undef SELECT_Z
