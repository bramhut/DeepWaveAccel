#include "backproj.hpp"
#include "pair_rom_data.hpp"
#include "b_vectors_data.hpp"
#include "tau_data.hpp"

// -----------------------------------------------------------------------------
// Compute y = 2 * Σ_{j<k} Re{ conj(b_j) * Σ_jk * b_k } - τ[pix]
// τ is pre-biased with the diagonal term (done in preprocessing).
// -----------------------------------------------------------------------------
void backprojection(hls::stream<AxisWordDFTc> &corr_stream,
                    hls::stream<img_t>        &img_stream)
{
    AXIS_IN_OUT(corr_stream);
    AXIS_IN_OUT(img_stream);

    // ---------------- Persistent storage ----------------
    // Σ upper triangle (pair ROM order)
    static DFTc_t Sigma_up[NPAIR];
#pragma HLS BIND_STORAGE variable=Sigma_up type=ram_1p impl=bram

    // Temporary per-pixel cache of steering line
    static b_t b_line[N_ELEM];
#pragma HLS ARRAY_PARTITION variable=b_line complete dim=1

    // ROM lookup tables (compile-time constants)
#pragma HLS BIND_STORAGE variable=j_rom type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=k_rom type=rom_1p impl=bram

    // Dependency relaxation
#pragma HLS DEPENDENCE variable=Sigma_up inter false
#pragma HLS DEPENDENCE variable=b_line   inter false

    // ---------------- FSM ----------------
    enum St { LOAD_UP, LOAD_BLINE, COMPUTE_UP, OUTPUT };
    static St st = LOAD_UP;

    static int pdx = 0;      // Σ pair index
    static int pix = 0;      // pixel index [0..IMG_LEN)
    static int idx = 0;      // generic index
    static acc_fix_t y_acc = 0;

    switch (st)
    {
    // Load upper-triangular correlation matrix from stream
    case LOAD_UP:
        if (!corr_stream.empty()) {
            AxisWordDFTc w = corr_stream.read();
            Sigma_up[idx] = DFTc_t(w.re, w.im);
            ++idx;
            if (idx == NPAIR) {
                idx = 0;
                pix = 0;
                st  = LOAD_BLINE;  // start processing first pixel
            }
        }
        break;

    // Load current pixel's steering vector from b_mem into local registers
    case LOAD_BLINE:
        b_line[idx] = b_vectors_rom[pix][idx];
        ++idx;
        if (idx == N_ELEM) {
            idx = 0;
            pdx = 0;
            y_acc = 0;
            st = COMPUTE_UP;
        }
        break;

    // Compute off-diagonal contributions for current pixel
    case COMPUTE_UP:
    {
        const int j = j_rom[pdx];
        const int k = k_rom[pdx];

        b_t bj_b = b_line[j];
        b_t bk_b = b_line[k];

        acc_fix_t s_re = (acc_fix_t)Sigma_up[pdx].real();
        acc_fix_t s_im = (acc_fix_t)Sigma_up[pdx].imag();

        acc_fix_t bj_re = (acc_fix_t)bj_b.real();
        acc_fix_t bj_im = (acc_fix_t)bj_b.imag();
        acc_fix_t bk_re = (acc_fix_t)bk_b.real();
        acc_fix_t bk_im = (acc_fix_t)bk_b.imag();

        // Complex multiply u = s * b_k
        acc_fix_t u_re = s_re * bk_re - s_im * bk_im;
        acc_fix_t u_im = s_re * bk_im + s_im * bk_re;

        // Re{ conj(b_j) * u } = bj_re*u_re + bj_im*u_im
        acc_fix_t contrib = bj_re * u_re + bj_im * u_im;
   
        y_acc += (contrib << 1);  // symmetric pair factor

        ++pdx;
        if (pdx == NPAIR) {
            pdx = 0;
            st  = OUTPUT;
        }
        break;
    }

    // Output final pixel result
    case OUTPUT:
    {
        acc_fix_t y_sub = y_acc - (acc_fix_t)tau_rom[pix];
        img_stream.write((img_t)y_sub);

        ++pix;
        y_acc = 0;

        if (pix == IMG_LEN) {
            pix = 0;
            st  = LOAD_UP;   // next correlation frame
        } else {
            st  = LOAD_BLINE; // next pixel
        }
        break;
    }
    } // switch
}
