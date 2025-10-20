#include "backproj.hpp"

// Build ROM with all (j,k) pairs for j<k
static void build_pair_rom(int j_rom[NPAIR], int k_rom[NPAIR]) {
#pragma HLS INLINE
    int p = 0;
    for (int j = 0; j < N_ELEM; ++j) {
        for (int k = j + 1; k < N_ELEM; ++k) {
#pragma HLS PIPELINE II=1
            j_rom[p] = j;
            k_rom[p] = k;
            ++p;
        }
    }
}

void backprojection(hls::stream<AxisWordDFTc> &corr_stream,
                    hls::stream<AxisWordDFTc> &b_stream,
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
    // Σ diagonal and Σ upper triangle (in pair ROM order)
    static DFTc_t Sigma_diag[N_ELEM];
    static DFTc_t Sigma_up[NPAIR];
#pragma HLS RESOURCE variable=Sigma_diag core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=Sigma_up   core=RAM_1P_BRAM

    // Steering vector buffer for current pixel
    static DFTc_t bvec[N_ELEM];
#pragma HLS RESOURCE variable=bvec core=RAM_1P_BRAM

    // Tau vector (per pixel) in BRAM
    static tau_t tau_mem[IMG_LEN];
#pragma HLS RESOURCE variable=tau_mem core=RAM_1P_BRAM

    // ROM for (j,k) index pairs
    static int j_rom[NPAIR], k_rom[NPAIR];
#pragma HLS BIND_STORAGE variable=j_rom type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=k_rom type=rom_1p impl=bram
    static bool rom_inited = false;
    if (!rom_inited) { build_pair_rom(j_rom, k_rom); rom_inited = true; }

    // -------- FSM --------
    enum St { LOAD_TAU, LOAD_DIAG, LOAD_UP, LOAD_B, COMPUTE_DIAG, COMPUTE_UP, OUTPUT };
    static St  st = LOAD_TAU;

    static int idx = 0;           // generic index for current state
    static int pix = 0;           // pixel index [0..IMG_LEN)
    static int jdx = 0;           // diagonal loop index
    static int pdx = 0;           // pair loop index
    static acc_fix_t y_acc = 0;   // scalar accumulator for current pixel

    switch (st)
    {
    case LOAD_TAU:
        // Preload exactly IMG_LEN tau values once
        if (!tau_stream.empty()) {
            tau_mem[idx] = tau_stream.read();
            ++idx;
            if (idx == IMG_LEN) {
                idx = 0;
                st  = LOAD_DIAG;
            }
        }
        break;

    case LOAD_DIAG:
        if (!corr_stream.empty()) {
            AxisWordDFTc w = corr_stream.read();
            Sigma_diag[idx] = DFTc_t(w.re, w.im); // imag ignored
            ++idx;
            if (idx == N_ELEM) { idx = 0; st = LOAD_UP; }
        }
        break;

    case LOAD_UP:
        if (!corr_stream.empty()) {
            AxisWordDFTc w = corr_stream.read();
            Sigma_up[idx] = DFTc_t(w.re, w.im);
            ++idx;
            if (idx == NPAIR) { idx = 0; st = LOAD_B; }
        }
        break;

    case LOAD_B:
        if (!b_stream.empty()) {
            AxisWordDFTc w = b_stream.read();
            bvec[idx] = DFTc_t(w.re, w.im);
            ++idx;
            if (idx == N_ELEM) {
                idx   = 0;
                y_acc = (acc_fix_t)0;
                jdx   = 0;
                st    = COMPUTE_DIAG;
            }
        }
        break;

    case COMPUTE_DIAG:
        {
            // y += Σ_jj * |b_j|^2
            // use complex to compute conj(bj)*bj, then take real part.
            acc_cplx_t bj( (acc_fix_t)bvec[jdx].real(), (acc_fix_t)bvec[jdx].imag() );
            acc_cplx_t bj_abs = std::conj(bj) * bj;      // |b_j|^2 (real)
            acc_fix_t  abs2   = bj_abs.real();

            acc_fix_t sjj = (acc_fix_t)Sigma_diag[jdx].real();
            y_acc += sjj * abs2;

            ++jdx;
            if (jdx == N_ELEM) {
                pdx = 0;
                st  = COMPUTE_UP;
            }
        }
        break;

    case COMPUTE_UP:
        {
            // Off-diagonal: 2 * Re( conj(b_j) * Σ_jk * b_k )
            int j = j_rom[pdx];
            int k = k_rom[pdx];

            acc_cplx_t bj((acc_fix_t)bvec[j].real(), (acc_fix_t)bvec[j].imag());
            acc_cplx_t bk((acc_fix_t)bvec[k].real(), (acc_fix_t)bvec[k].imag());
            acc_cplx_t s ((acc_fix_t)Sigma_up[pdx].real(), (acc_fix_t)Sigma_up[pdx].imag());

            acc_cplx_t t = std::conj(bj) * s * bk;       // complex triple product
            y_acc += (acc_fix_t)2 * t.real();

            ++pdx;
            if (pdx == NPAIR) {
                st = OUTPUT;
            }
        }
        break;

    case OUTPUT:
        {
            // Subtract tau for this pixel, cast to DFT_t and emit
            tau_t tau_val = tau_mem[pix];
            acc_fix_t y_sub = y_acc - (acc_fix_t)tau_val;

            bool first = (pix == 0);
            bool last  = (pix == IMG_LEN - 1);
            img_stream.write(AxisWordImg((DFT_t)y_sub, last, first));

            ++pix;
            if (pix == IMG_LEN) {
                pix = 0;
                st  = LOAD_DIAG;     // next frame's Σ
            } else {
                st  = LOAD_B;        // next steering vector for same Σ
            }
        }
        break;
    }
}
