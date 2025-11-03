#include "backproj.hpp"
#include "pair_rom_data.hpp"
#include <iostream>

// -----------------------------------------------------------------------------
// Compute y = 2 * Σ_{j<k} Re{ conj(b_j) * Σ_jk * b_k } - τ[pix]
// τ is pre-biased with the diagonal term (done in preprocessing).
// -----------------------------------------------------------------------------
void backprojection(hls::stream<AxisWordDFTc> &corr_stream,
                    hls::stream<word_t>       &param_in,
                    hls::stream<img_t>        &img_stream,
                    status_bp_t               &status)
{
    AP_CTRL_NONE;
    AXIS_IN_OUT(corr_stream);
    AXIS_IN_OUT(param_in);
    AXIS_IN_OUT(img_stream);
    AXIL_CFG(status);

    // =========================================================================
    // Persistent parameter memories
    // =========================================================================
    static b_t   b_mem[IMG_LEN][N_ELEM]; // steering vectors
    static tau_t tau_mem[IMG_LEN];       // tau compensation values

#pragma HLS BIND_STORAGE variable=b_mem  type=ram_2p impl=uram
#pragma HLS ARRAY_RESHAPE variable=b_mem complete dim=2
#pragma HLS BIND_STORAGE variable=tau_mem type=ram_1p impl=bram
#pragma HLS DEPENDENCE variable=b_mem inter false
#pragma HLS DEPENDENCE variable=tau_mem inter false


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

    // =========================================================================
    // FSM definitions
    // =========================================================================
    enum St {
        LOAD_PARAMS,  // receive b_mem + tau_mem
        LOAD_UP,      // receive Σ frame
        LOAD_BLINE,   // preload steering line
        COMPUTE_UP,   // compute upper-triangular sum
        OUTPUT        // write pixel result
    };
    static St st = LOAD_PARAMS;

    enum StL {
        READ_B,       // loading b_mem
        READ_TAU      // loading tau_mem
    };
    static StL stl = READ_B;

    // Indices / accumulators
    static int pdx = 0;
    static int pix = 0;
    static int idx = 0;
    static int elem = 0;
    static acc_fix_t y_acc = 0;
    static bool config_loaded = false;
    static bool imag_word = false;
    static b_real_t real_in;

    static word_t pixels_out = 0;
    static word_t sigmas_in = 0;

    // =========================================================================
    // Main FSM
    // =========================================================================
    switch (st)
    {
         // -------------------------------------------------------------------------
    // LOAD_PARAMS: Receive all steering vectors and tau values
    // -------------------------------------------------------------------------
    case LOAD_PARAMS: {
        if (!param_in.empty()) {
            ap_uint<32> w = param_in.read();

            switch (stl)
            {
            case READ_B: {
                // Expecting N_ELEM * IMG_LEN complex pairs:
                // - real part word (user=0x1)
                // - imag part word (user=0x2)
                // We'll interpret bits directly into ap_fixed fields.
                // user[0]=0 ⇒ real, user[0]=1 ⇒ imag

                config_loaded = false; // Reset, we are reloading everything
                status.config_loaded = false;
                status.idx = (pix<<16) | (elem<<1) | ((int)imag_word);
                if (!imag_word) {
                    // real part
                    real_in.range() = w.range();
                    imag_word = !imag_word;
                } else {
                    // imag part completes the complex sample
                    b_real_t imag_in;
                    imag_in.range() = w.range();
                    b_mem[pix][elem] = b_t(real_in, imag_in);
                    imag_word = !imag_word;
                    elem++;
                    if (elem == N_ELEM) {
                        elem = 0;
                        pix++;
                        if (pix == IMG_LEN) {
                            pix = 0;
                            stl = READ_TAU;
                        }
                    }
                }
                break;
            }

            case READ_TAU: {
                tau_mem[pix].range() = w.range();
                status.idx = pix;
                pix++;
                if (pix == IMG_LEN) {
                    pix = 0;
                    config_loaded = true;
                    status.config_loaded = true;
                    stl = READ_B;
                    // Do not necessarily go straight to LOAD_UP, maybe we'd like to reload params. Just stay in LOAD_PARAMS and go to LOAD_UP when corr_stream is no longer empty
                }
                break;
            }
            } // switch(stl)
            status.param_state = stl;
        } else if (config_loaded && !corr_stream.empty()) {
            st = LOAD_UP;
        }
        break;
    }
    // Load upper-triangular correlation matrix from stream
    case LOAD_UP:
        if (!corr_stream.empty()) {
            status.sigmas_in = ++sigmas_in;
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
        b_line[idx] = b_mem[pix][idx];
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
        acc_fix_t y_sub = y_acc - (acc_fix_t)tau_mem[pix];
        img_stream.write((img_t)y_sub);

        status.idx = pix;
        status.pixels_out = ++pixels_out;
        status.out_fifo_level = img_stream.size();
        ++pix;
        y_acc = 0;

        if (pix == IMG_LEN) {
            pix = 0;
            st  = LOAD_PARAMS; // next correlation frame, but first check if there are any params to be updated
        } else {
            st  = LOAD_BLINE; // next pixel
        }
        break;
    }
    } // switch
    status.fsm_state = st; // Write the current state to AXILITE
}
