#include "crosscor.hpp"
#include <cmath>
#include <iostream>

void crosscor(hls::stream<AxisWordDFTc> &in_stream,
              hls::stream<AxisWordDFTc> &out_stream,
              hls::stream<norm_sum_t> &norm_stream)
{
    AXIS_IN_OUT(in_stream);
    AXIS_IN_OUT(out_stream);
    AXIS_IN_OUT(norm_stream);
    AP_CTRL_NONE;
#pragma HLS PIPELINE II=1

    // ---------------- Persistent State ----------------
    enum State { COLLECT, CORRELATE, NORM_SUM, OUTPUT, CLEAR };
    static State st = COLLECT;

    static DFTc_t u[N_ELEM];          // current input vector
#pragma HLS RESOURCE variable=u core=RAM_1P_BRAM

    static corr_accum_t R[N_ELEM][N_ELEM];  // correlation accumulator
#pragma HLS RESOURCE variable=R core=RAM_2P_BRAM

    static int v_count = 0;
    static int i = 0, j = 0;
    static int frames_acc = 0;
    static int diag_idx = 0;

    static norm_sum_t power_acc = 0;
    static norm_inv_t scale = 0;

    // ---------------- FSM ----------------
    switch (st)
    {
    case COLLECT:
        if (!in_stream.empty()) {
            AxisWordDFTc w = in_stream.read();
            u[v_count] = DFTc_t(w.re, w.im);
            v_count++;
            if (v_count == N_ELEM) {
                v_count = 0;
                i = 0; j = 0;
                st = CORRELATE;
            }
        }
        break;

    case CORRELATE:
        {
            // Perform R += u^H * u
            // For each (i,j): R[i][j] += conj(u[j]) * u[i]
            corr_accum_t prod = std::conj((corr_accum_t)u[j]) * (corr_accum_t)u[i]; // Casting before calculating is neccesary
            R[i][j] += prod;

            // Step through matrix
            j++;
            if (j == N_ELEM) { j = 0; i++; }
            if (i == N_ELEM) {
                i = 0; j = 0;
                frames_acc++;
                if (frames_acc < GROUP_FRAMES) {
                    st = COLLECT;
                } else {
                    diag_idx = 0;
                    power_acc = 0;
                    st = NORM_SUM;
                }
            }
        }
        break;

    case NORM_SUM:
        {
            // Accumulate diagonal real part (signal power)
            power_acc += (norm_sum_t)(R[diag_idx][diag_idx].real());
            diag_idx++;

            if (diag_idx == N_ELEM) {
                // Correction gain = ×(1/sqrt(2))
                const norm_sum_t gain = (norm_sum_t)0.70710678118;
                norm_sum_t corrected = power_acc * gain;

                norm_stream.write(corrected);

                // Lower limit (≥1)
                if (corrected < (norm_sum_t)1.0)
                    corrected = (norm_sum_t)1.0;

                // Reciprocal (→ ufix24_En23)
                scale = (norm_inv_t)1.0 / corrected;

                // Next: normalize and output
                i = 0; j = 0;
                st = OUTPUT;
            }
        }
        break;

    case OUTPUT:
        {
            // Apply normalization (scaled correlation matrix)
            DFT_t out_re = (DFT_t)(R[i][j].real() * scale);
            DFT_t out_im = (DFT_t)(R[i][j].imag() * scale);

            bool first = (i == 0 && j == 0);
            bool last  = (i == N_ELEM - 1 && j == N_ELEM - 1);

            out_stream.write(AxisWordDFTc(out_re, out_im, last, first));

            j++;
            if (j == N_ELEM) { j = 0; i++; }
            if (i == N_ELEM) {
                i = 0; j = 0;
                st = CLEAR;
            }
        }
        break;

    case CLEAR:
        {
            // Reset accumulated correlation
            R[i][j] = DFTc_t(0, 0);

            j++;
            if (j == N_ELEM) { j = 0; i++; }
            if (i == N_ELEM) {
                i = 0; j = 0;
                frames_acc = 0;
                st = COLLECT;
            }
        }
        break;
    }
}
