#include "backproj.hpp"
#include "pair_rom_data.hpp"

// Compute y = 2 * sum_{j<k} Re{ conj(b_j) * Sigma_{jk} * b_k }  -  tau[pix]
// - tau is pre-biased with the diagonal term per your new flow
void backprojection(hls::stream<AxisWordDFTc> &corr_stream,
                    hls::stream<b_t>          &b_stream,
                    hls::stream<tau_t>        &tau_stream,
                    hls::stream<AxisWordImg>  &img_stream)
{
    AXIS_IN_OUT(corr_stream);
    AXIS_IN_OUT(b_stream);
    AXIS_IN_OUT(tau_stream);
    AXIS_IN_OUT(img_stream);
    AP_CTRL_NONE;
#pragma HLS PIPELINE II=1

    // ---------------- Persistent storage ----------------

    // Σ upper triangle (in pair ROM order) — diagonals are no longer streamed
    static DFTc_t Sigma_up[NPAIR];
#pragma HLS BIND_STORAGE variable=Sigma_up type=ram_1p impl=bram

    // Steering vectors b[elem][pixel], large, stored in URAM
    static b_t bvec[N_ELEM][IMG_LEN];
#pragma HLS BIND_STORAGE variable=bvec type=ram_2p impl=uram
#pragma HLS ARRAY_RESHAPE variable=bvec complete dim=1



    // Tau vector (per pixel) in BRAM (loaded once); already contains y_diag compensation
    static tau_t tau_mem[IMG_LEN];
#pragma HLS BIND_STORAGE variable=tau_mem type=ram_1p impl=bram

    // Per-pixel cache of steering vector line: b_line[j] = bvec[j][pix]
    static b_t b_line[N_ELEM];
#pragma HLS ARRAY_PARTITION variable=b_line complete dim=1


    // Predefined ROM lookup tables (compile-time)
#pragma HLS BIND_STORAGE variable=j_rom type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=k_rom type=rom_1p impl=bram

#pragma HLS DEPENDENCE variable=bvec inter false
#pragma HLS DEPENDENCE variable=b_line inter false
#pragma HLS DEPENDENCE variable=Sigma_up inter false
#pragma HLS DEPENDENCE variable=tau_mem inter false


    // ---------------- FSM ----------------
    enum St { LOAD_TAU, LOAD_B, LOAD_UP, LOAD_BLINE, COMPUTE_UP, OUTPUT };
    static St  st   = LOAD_TAU;

    static int idx  = 0;          // generic index for current state
    static int pix  = 0;          // pixel index [0..IMG_LEN)
    static int elem = 0;          // element index for b loading
    static int pdx  = 0;          // pair loop index
    static acc_fix_t y_acc = 0;   // scalar accumulator for current pixel
    static int p = 0;             // pixel index while loading b
    static int e = 0;             // element index while loading b

    switch (st)
    {
    // Load tau once into BRAM (already pre-biased with y_diag)
    case LOAD_TAU:
        if (!tau_stream.empty()) {
            tau_mem[idx] = tau_stream.read();
            ++idx;
            if (idx == IMG_LEN) {
                idx = 0;
                st  = LOAD_B;
            }
        }
        break;

    // Load b once into BRAM (banked by element)

    case LOAD_B:
        if (!b_stream.empty()) {
            bvec[e][p] = b_stream.read();

            // advance indices without / or %
            ++e;
            if (e == N_ELEM) { e = 0; ++p; }
            if (p == IMG_LEN) {
                e = 0; p = 0;
                st = LOAD_UP;
            }
        }
        break;


    // Load only the upper-triangular Σ pairs; diagonals are not streamed
    case LOAD_UP:
        if (!corr_stream.empty()) {
            AxisWordDFTc w = corr_stream.read();
            Sigma_up[idx] = DFTc_t(w.re, w.im);
            ++idx;
            if (idx == NPAIR) {
                idx = 0;
                pix = 0;
                st  = LOAD_BLINE;   // preload b_line for first pixel
            }
        }
        break;

    // Preload one pixel’s steering vector line into local registers
    case LOAD_BLINE:
        // Read one element per cycle: b_line[idx] = bvec[idx][pix]
        b_line[idx] = bvec[idx][pix];
        ++idx;
        if (idx == N_ELEM) {
            idx  = 0;
            pdx  = 0;
            y_acc = 0;
            st   = COMPUTE_UP;
        }
        break;

    case COMPUTE_UP:
    {
        // Off-diagonal contribution for current (j,k) pair:
        // 2 * Re{ conj(b_j) * Σ_jk * b_k }
        const int j = j_rom[pdx];
        const int k = k_rom[pdx];

        // Use cached b_line instead of rereading BRAM
        b_t bj_b = b_line[j];
        b_t bk_b = b_line[k];

        // Sigma_up is stored as complex (upper triangle)
        acc_fix_t s_re = (acc_fix_t)Sigma_up[pdx].real();
        acc_fix_t s_im = (acc_fix_t)Sigma_up[pdx].imag();

        // Registers for b_j / b_k components
        acc_fix_t bj_re = (acc_fix_t)bj_b.real();
        acc_fix_t bj_im = (acc_fix_t)bj_b.imag();
        acc_fix_t bk_re = (acc_fix_t)bk_b.real();
        acc_fix_t bk_im = (acc_fix_t)bk_b.imag();

        // One complex multiply: u = s * b_k
        acc_fix_t u_re = s_re * bk_re - s_im * bk_im;
        acc_fix_t u_im = s_re * bk_im + s_im * bk_re;

        // Two real MACs: Re{ conj(b_j) * u } = bj_re*u_re + bj_im*u_im
        acc_fix_t contrib = bj_re * u_re + bj_im * u_im;

        // 2× for symmetric pair (j!=k)
        y_acc += (contrib << 1);

        // advance pair index
        ++pdx;
        if (pdx == NPAIR) {
            pdx = 0;
            st  = OUTPUT;
        }
        break;
    }

    case OUTPUT:
    {
        // Subtract (pre-biased) tau for this pixel and output result
        tau_t tau_val = tau_mem[pix];
        acc_fix_t y_sub = y_acc - (acc_fix_t)tau_val;

        bool first = (pix == 0);
        bool last  = (pix == IMG_LEN - 1);
        img_stream.write(AxisWordImg((img_t)y_sub, last, first));

        // next pixel
        ++pix;
        y_acc = 0;

        if (pix == IMG_LEN) {
            pix = 0;
            st  = LOAD_UP;     // Next Σ frame
        } else {
            st  = LOAD_BLINE;  // Preload the next pixel's b_line
        }
        break;
    }
    } // switch
}
