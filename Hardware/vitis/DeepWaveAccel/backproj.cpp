#include "backproj.hpp"
#include "pair_rom_data.hpp"

// (magsq no longer needed)

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

    // -------- Persistent storage --------
    // Σ upper triangle (in pair ROM order) — diagonals are no longer streamed
    static DFTc_t Sigma_up[NPAIR];
#pragma HLS BIND_STORAGE variable=Sigma_up type=ram_1p impl=bram

    // Steering vectors b[elem][pixel], stored in URAM (loaded once)
    static b_t bvec[N_ELEM][IMG_LEN];
#pragma HLS BIND_STORAGE variable=bvec type=ram_2p impl=uram
#pragma HLS ARRAY_RESHAPE variable=bvec complete dim=1

    // Tau vector (per pixel) in BRAM (loaded once); already contains y_diag compensation
    static tau_t tau_mem[IMG_LEN];
#pragma HLS BIND_STORAGE variable=tau_mem type=ram_1p impl=bram

    // ---- Predefined ROM lookup tables (compile-time) ----
#pragma HLS BIND_STORAGE variable=j_rom type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=k_rom type=rom_1p impl=bram

    // -------- FSM --------
    enum St { LOAD_TAU, LOAD_B, LOAD_UP, COMPUTE_UP, OUTPUT };
    static St  st = LOAD_TAU;

    static int idx = 0;           // generic index for current state
    static int pix = 0;           // pixel index [0..IMG_LEN)
    static int elem = 0;          // element index for b loading
    static int pdx = 0;           // pair loop index
    static acc_fix_t y_acc = 0;   // scalar accumulator for current pixel

    switch (st)
    {
    case LOAD_TAU:
        // Load tau once into BRAM (already pre-biased with y_diag)
        if (!tau_stream.empty()) {
            tau_mem[idx] = tau_stream.read();
            ++idx;
            if (idx == IMG_LEN) {
                idx = 0;
                st = LOAD_B;
            }
        }
        break;

    case LOAD_B:
        // Load b once into URAM
        if (!b_stream.empty()) {
            // Flattened load: element varies fastest
            int e = elem % N_ELEM;
            int p = elem / N_ELEM;
            bvec[e][p] = b_stream.read();
            ++elem;
            if (elem == N_ELEM * IMG_LEN) {
                elem = 0;
                st = LOAD_UP;
            }
        }
        break;

    case LOAD_UP:
        // Load only the upper-triangular Σ pairs; diagonals are not streamed
        if (!corr_stream.empty()) {
            AxisWordDFTc w = corr_stream.read();
            Sigma_up[idx] = DFTc_t(w.re, w.im);
            ++idx;
            if (idx == NPAIR) {
                idx = 0;
                pix = 0;
                pdx = 0;
                y_acc = 0;
                st = COMPUTE_UP;
            }
        }
        break;

    case COMPUTE_UP:
        {
            // Off-diagonal contribution for current (j,k) pair:
            // 2 * Re{ conj(b_j) * Σ_jk * b_k }
            const int j = j_rom[pdx];
            const int k = k_rom[pdx];

            // Load b_j and b_k once (URAM → regs)
            b_t bj_b = bvec[j][pix];
            b_t bk_b = bvec[k][pix];

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
            y_acc += contrib<<1;

            // advance pair index
            ++pdx;
            if (pdx == NPAIR) {
                pdx = 0;
                st = OUTPUT;
            }
        }
        break;

    case OUTPUT:
        {
            // Subtract (pre-biased) tau for this pixel and output result
            tau_t tau_val = tau_mem[pix];
            acc_fix_t y_sub = y_acc - (acc_fix_t)tau_val;

            bool first = (pix == 0);
            bool last  = (pix == IMG_LEN - 1);
            img_stream.write(AxisWordImg((bp_out_t)y_sub, last, first));

            ++pix;
            y_acc = 0;

            if (pix == IMG_LEN) {
                pix = 0;
                st = LOAD_UP; // Next Σ frame
            } else {
                st = COMPUTE_UP; // Next pixel, same Σ
            }
        }
        break;
    }
}
