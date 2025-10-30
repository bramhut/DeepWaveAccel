#include "crosscor.hpp"
#include "pair_rom_data.hpp"
#include <cmath>
#include <iostream>

// ---------------------------------------------------------
// Cross-correlation kernel (upper-triangle output order)
// - Accumulates only the upper triangle in a flat array R_flat[NPAIR]
// - Per frame: first sum diagonal power, then accumulate upper-triangle pairs
// - On the last frame of a group, launch reciprocal in parallel (outside FSM)
// ---------------------------------------------------------
void crosscor(hls::stream<AxisWordDFTc> &in_stream,
              hls::stream<AxisWordDFTc> &out_stream,
              hls::stream<norm_sum_t>   &norm_stream)
{
    AP_CTRL_NONE;
    AXIS_IN_OUT(in_stream);
    AXIS_IN_OUT(out_stream);
    AXIS_IN_OUT(norm_stream);

    // ---------------- Persistent State ----------------
    enum State { COLLECT, CORRELATE, OUTPUT };
    static State st = COLLECT;

    // Two-phase correlate sub-state
    enum CorrPhase { PHASE_DIAG, PHASE_UPPER };
    static CorrPhase phase = PHASE_DIAG;

    // Current input vector
    static DFTc_t u[N_ELEM];
#pragma HLS BIND_STORAGE variable=u type=ram_2p impl=bram

    // Accumulator for upper triangle only (flattened by pair_rom order)
    static corr_accum_t R_flat[NPAIR];
#pragma HLS BIND_STORAGE variable=R_flat type=ram_2p impl=bram

    // Counters
    static int v_count = 0;     // samples collected for u[]
    static int frames_acc = 0;  // number of frames accumulated into R_flat
    static int diag_i = 0;      // index over diagonal during PHASE_DIAG
    static int pair_idx = 0;    // index [0..NPAIR) during PHASE_UPPER/OUTPUT/CLEAR

    // Power accumulation and scaling
    static power_accum_t power_acc = 0;          // sum of |u[i]|^2 over i
    static norm_inv_t scale = 0;              // 1 / corrected_reg

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
                // Start a new per-frame correlate pass: diagonal first
                phase     = PHASE_DIAG;
                diag_i    = 0;
                pair_idx  = 0;
                st        = CORRELATE;
            }
        }
        break;

    case CORRELATE:
    {
#pragma HLS DEPENDENCE variable=R_flat inter false
        if (phase == PHASE_DIAG) {
            // ---- Phase 1: accumulate diagonal power (no BRAM touches on R_flat)
            // power += |u[i]|^2 = re^2 + im^2
            power_accum_t ur = (power_accum_t)u[diag_i].real();
            power_accum_t ui = (power_accum_t)u[diag_i].imag();
            power_acc += (ur*ur + ui*ui);

            diag_i++;
            if (diag_i == N_ELEM) {
                // We finished diagonal for this frame
                // If this is the last frame in the group, launch reciprocal now.
                if (frames_acc == (GROUP_FRAMES - 1)) {
                    const norm_sum_t gain = (norm_sum_t)0.70710678118; // 1/sqrt(2)
                    norm_sum_t corrected = (norm_sum_t)power_acc * gain;
                    norm_stream.write(corrected);
                    if (corrected < (norm_sum_t)1.0) corrected = (norm_sum_t)1.0; 
                    scale = (norm_inv_t)1.0 / corrected;
                    power_acc = 0;
                }

                // Move to Phase 2: accumulate upper triangle for this frame
                phase    = PHASE_UPPER;
                pair_idx = 0;
            }
        } else { // PHASE_UPPER
            // ---- Phase 2: accumulate upper triangle using pair_rom mapping
            int j_ = j_rom[pair_idx];
            int k_ = k_rom[pair_idx];
            power_accum_t re1 = u[k_].real();
            power_accum_t im1 = u[k_].imag();
            power_accum_t re2 = u[j_].real();
            power_accum_t im2 = u[j_].imag();
            power_accum_t reo = re1 * re2 + im1 * im2 + R_flat[pair_idx].real();
            power_accum_t imo = re1 * im2 - re2 * im1 + R_flat[pair_idx].imag();
            R_flat[pair_idx] = corr_accum_t(reo, imo);

            pair_idx++;
            if (pair_idx == NPAIR) {
                // Completed this frame's upper triangle
                frames_acc++;
                if (frames_acc < GROUP_FRAMES) {
                    // Collect next frame
                    st = COLLECT;
                } else {
                    // Completed GROUP_FRAMES frames; proceed to NORM_SUM
                    pair_idx = 0;
                    frames_acc = 0;
                    st = OUTPUT;
                }
            }
        }
        break;
    }

    case OUTPUT: {
            corr_accum_t v = R_flat[pair_idx];
            R_flat[pair_idx] = DFTc_t(0, 0);

            DFT_t out_re = (DFT_t)(v.real() * scale);
            DFT_t out_im = (DFT_t)(v.imag() * scale);

            out_stream.write(AxisWordDFTc(out_re, out_im));

            pair_idx++;
            if (pair_idx == NPAIR) {
                // Done streaming frame
                pair_idx = 0;
                st = COLLECT;
            }
        break;
    }

    // case CLEAR:
    //     // Zero R_flat sequentially
        
    //     pair_idx++;
    //     if (pair_idx == NPAIR) {
    //         pair_idx   = 0;
    //         frames_acc = 0;
    //         st         = COLLECT;
    //     }
    //     break;
    // }
    }
}
