#include "crosscor.hpp"
#include "pair_rom_data.hpp"
#include <cmath>
#include <iostream>

// ---------------------------------------------------------
// Cross-correlation kernel (upper-triangle output order)
// ---------------------------------------------------------
void crosscor(hls::stream<AxisWordDFTc> &in_stream,
              hls::stream<AxisWordDFTc> &out_stream,
              hls::stream<norm_sum_t>   &norm_stream)
{
    AXIS_IN_OUT(in_stream);
    AXIS_IN_OUT(out_stream);
    AXIS_IN_OUT(norm_stream);
    AP_CTRL_NONE;
#pragma HLS PIPELINE II=2

    // ---------------- Persistent State ----------------
    enum State { COLLECT, CORRELATE, NORM_SUM, OUTPUT, CLEAR };
    static State st = COLLECT;

    // Current input vector
    static DFTc_t u[N_ELEM];
#pragma HLS BIND_STORAGE variable=u type=ram_2p impl=bram

    // Correlation accumulator
    static corr_accum_t R[N_ELEM][N_ELEM];
// #pragma HLS BIND_STORAGE variable=R type=ram_2p impl=bram // Gives a warning, just let the compiler work out the required RAM type

    static int v_count = 0;
    static int i = 0, j = 0;
    static int frames_acc = 0;
    static int diag_idx = 0;
    static int pair_idx = 0;

    static norm_sum_t power_acc = 0;
    static norm_inv_t scale = 0;

    // ---- Predefined ROM lookup tables (compile-time) ----
#pragma HLS ARRAY_PARTITION variable=j_rom complete
#pragma HLS ARRAY_PARTITION variable=k_rom complete


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
        // Break false read-after-write dependency on R
#pragma HLS DEPENDENCE variable=R inter false

        // Compute only upper-triangle (j <= i)
        if (j <= i) {
            corr_accum_t prod = std::conj((corr_accum_t)u[i]) * (corr_accum_t)u[j];
            R[i][j] += prod;
        }

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
                // Apply sqrt(1/2) normalization
                const norm_sum_t gain = (norm_sum_t)0.70710678118; // 1/sqrt(2)
                norm_sum_t corrected = power_acc * gain;

                norm_stream.write(corrected);

                if (corrected < (norm_sum_t)1.0)
                    corrected = (norm_sum_t)1.0;

                scale = (norm_inv_t)1.0 / corrected;

                diag_idx = 0;
                pair_idx = 0;
                st = OUTPUT;
            }
        }
        break;

    case OUTPUT:
        {
            // Only stream the upper triangle (j<k)
            if (pair_idx < NPAIR) {
                int j_ = j_rom[pair_idx];
                int k_ = k_rom[pair_idx];
                corr_accum_t v = R[k_][j_]; // R is symmetric conjugate

                DFT_t out_re = (DFT_t)(v.real() * scale);
                DFT_t out_im = (DFT_t)(v.imag() * scale);

                bool first = (pair_idx == 0);
                bool last  = (pair_idx == NPAIR - 1);
                out_stream.write(AxisWordDFTc(out_re, out_im, last, first));

                pair_idx++;
            }
            else {
                i = j = 0;
                st = CLEAR;
            }
        }
        break;

    case CLEAR:
        {
            // Reset only upper triangle
            if (j <= i)
                R[i][j] = DFTc_t(0, 0);

            j++;
            if (j == N_ELEM) { j = 0; i++; }
            if (i == N_ELEM) {
                i = j = 0;
                frames_acc = 0;
                st = COLLECT;
            }
        }
        break;
    }
}
