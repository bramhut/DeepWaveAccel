#include "deblur.hpp"

// -------------------------------------------------------------
// Single-pixel sparse Laplacian multiply: w[i] = (L * v)[i]
// Uses: main scalar + ND off-diagonals (symmetric).
// Called once per cycle from the FSM.
// -------------------------------------------------------------
static inline img_t lap_pixel(
    const img_t                 v[IMG_LEN],
    idx_t                       i,
    coef_t                      lap_main,
    const off_t                 off[ND],
    const coef_t                lap_rest[ND][IMG_LEN])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=off complete
#pragma HLS ARRAY_PARTITION variable=lap_rest dim=1 complete

    acc_t acc = (acc_t)lap_main * (acc_t)v[i];

    // ND symmetric off-diagonals
    for (int d = 0; d < ND; ++d) {
#pragma HLS UNROLL
        int o  = (int)off[d];
        int il = (int)i - o;
        int iu = (int)i + o;
        coef_t a = lap_rest[d][(int)i];

        if (il >= 0)         acc -= (acc_t)a * (acc_t)v[il];
        if (iu < IMG_LEN)    acc -= (acc_t)a * (acc_t)v[iu];
    }
    return (img_t)acc;
}

// -------------------------------------------------------------
// Top-level: cycle-accurate FSM (one operation per call)
//   Process: load Lap → load BP → for each layer run Chebyshev
//            recurrence with sparse L, then add BP, stream out.
// -------------------------------------------------------------
void deblur(
    hls::stream<AxisWordImg> &bp_stream,
    hls::stream<AxisWordLap> &lap_stream,
    hls::stream<AxisWordImg> &img_stream,
    deblur_config            &cfg)
{
    AXIS_IN_OUT(bp_stream);
    AXIS_IN_OUT(lap_stream);
    AXIS_IN_OUT(img_stream);
    AXIL_CFG(cfg);
    AP_CTRL_NONE;
#pragma HLS PIPELINE II=1

    // ------------------- Persistent memories -------------------
    static coef_t lap_main = 0;
    static coef_t lap_rest[ND][IMG_LEN];
#pragma HLS BIND_STORAGE variable=lap_rest type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=lap_rest dim=1 complete

    // working images
    static img_t bp_buf[IMG_LEN];   // backprojection per frame
    static img_t y_acc[IMG_LEN];    // accumulator for Chebyshev sum
    static img_t z0[IMG_LEN], z1[IMG_LEN], z2[IMG_LEN];
#pragma HLS BIND_STORAGE variable=bp_buf type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=y_acc  type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=z0    type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=z1    type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=z2    type=ram_1p impl=bram

    // ------------------- FSM state -------------------
    enum St {
        LOAD_MAIN, LOAD_OFF_LINES,
        LOAD_BP,
        LAYER_CLEAR_Z0,       // seed zeros for first layer
        CHEB_Y0,              // y = θ0*z0
        CHEB_Z1,              // z1 = L*z0
        CHEB_ADD_T1,          // y += θ1*z1
        CHEB_LOOP_LZ1,        // t = L*z1  (reuse z2 as t)
        CHEB_FORM_ADD,        // z2 = 2*t - z0, y += θk*z2
        CHEB_SHIFT,           // z0=z1; z1=z2
        LAYER_ADD_BP,         // z0 = y + bp (next layer input) OR final
        OUTPUT                // stream final image and go back to LOAD_BP
    };
    static St st = LOAD_MAIN;

    static idx_t i = 0;                  // pixel index
    static int   d = 0;                  // diag index
    static int   k = 0;                  // Chebyshev k
    static int   layer = 0;              // layer index

    // AXI-lite arrays – partition for parallel access
#pragma HLS ARRAY_PARTITION variable=cfg.lap_off complete
#pragma HLS ARRAY_PARTITION variable=cfg.theta   complete

    switch (st)
    {
    // ---------------- Load Laplacian once ----------------
    case LOAD_MAIN:
        if (!lap_stream.empty()) {
            lap_main = lap_stream.read().data;
            d = 0; i = 0;
            st = LOAD_OFF_LINES; // offsets come from AXI-Lite; just advance
        }
        break;

    case LOAD_OFF_LINES:
        if (!lap_stream.empty()) {
            lap_rest[d][i] = lap_stream.read().data;
            ++i;
            if (i == IMG_LEN) { i = 0; ++d; }
            if (d == ND) {
                d = 0; i = 0;
                st = LOAD_BP;
            }
        }
        break;

    // ---------------- Load backprojection frame ----------------
    case LOAD_BP:
        if (!bp_stream.empty()) {
            AxisWordImg w = bp_stream.read();
            bp_buf[i] = w.data;
            ++i;
            if (i == IMG_LEN) {
                i = 0; layer = 0;
                st = LAYER_CLEAR_Z0;
            }
        }
        break;

    // ---------------- Start layer: seed z0 = 0 ----------------
    case LAYER_CLEAR_Z0:
        z0[i] = (img_t)0;    // first layer uses zero seed - this can be changed if required
        y_acc[i] = (img_t)0; // ensure clean accumulator
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            st = CHEB_Y0;
        }
        break;

    // y = θ0 * z0
    case CHEB_Y0:
        y_acc[i] = (img_t)((acc_t)cfg.theta[0] * (acc_t)z0[i]);
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            st = CHEB_Z1;
        }
        break;

    // z1 = L * z0
    case CHEB_Z1:
        z1[i] = lap_pixel(z0, i, lap_main, cfg.lap_off, lap_rest);
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            st = CHEB_ADD_T1;
        }
        break;

    // y += θ1 * z1
    case CHEB_ADD_T1:
        y_acc[i] = (img_t)((acc_t)y_acc[i] + (acc_t)cfg.theta[1] * (acc_t)z1[i]);
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            k = 2;
            st = (cfg.K >= 2) ? CHEB_LOOP_LZ1 : LAYER_ADD_BP;
        }
        break;

    // t = L * z1  (reuse z2 as temporary t)
    case CHEB_LOOP_LZ1:
        z2[i] = lap_pixel(z1, i, lap_main, cfg.lap_off, lap_rest);
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            st = CHEB_FORM_ADD;
        }
        break;

    // z2 = 2*t - z0 ; y += θk * z2
    case CHEB_FORM_ADD: {
        acc_t two_t = ((acc_t)z2[i])<<1;
        z2[i] = (img_t)(two_t - (acc_t)z0[i]);
        y_acc[i] = (img_t)((acc_t)y_acc[i] + (acc_t)cfg.theta[k] * (acc_t)z2[i]);
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            st = CHEB_SHIFT;
        }
        break;
    }

    // z0 <- z1 ; z1 <- z2 ; advance k
    case CHEB_SHIFT:
        z0[i] = z1[i];
        z1[i] = z2[i];
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            ++k;
            if (k <= (int)cfg.K) {
                st = CHEB_LOOP_LZ1;   // next Chebyshev term
            } else {
                st = LAYER_ADD_BP;    // layer done
            }
        }
        break;

    // Finish layer: z0 = y + bp  (next layer input), or go to OUTPUT
    case LAYER_ADD_BP:
        z0[i] = (img_t)((acc_t)y_acc[i] + (acc_t)bp_buf[i]);
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            ++layer;
            if (layer < (int)cfg.n_layers) {
                // Next layer: restart sequence with current z0 as seed
                st = CHEB_Y0;
            } else {
                // All layers done → stream final result (z0)
                st = OUTPUT;
            }
        }
        break;

    // Stream out final result and wait for next BP frame
    case OUTPUT: {
        AxisWordImg ow;
        ow.data = z0[i];
        ow.last = (i == IMG_LEN-1);
        ow.user = (i == 0);
        img_stream.write(ow);
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            st = LOAD_BP;
        }
        break;
    }
    }
}
