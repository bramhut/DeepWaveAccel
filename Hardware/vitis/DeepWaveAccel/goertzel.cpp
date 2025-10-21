#include "goertzel.hpp"
#include "hann_window_data.hpp"
#include <cmath>
#include <iostream>

// ---------------------------------------------
// Host-side preparation (non-synth)
// ---------------------------------------------
void goertzel_prepare_config(goertzel_config &cfg, double fs_in, double ff) {
    double fr = fs_in / double(N_WIN);
    int bin = int(std::round(ff / fr));
    int bins[NBINS] = { bin, bin - 1 };

    // Goertzel coefficients (no *2)
    for (int b = 0; b < NBINS; ++b) {
        double omega = 2.0 * M_PI * bins[b] / double(N_WIN);
        cfg.COS_OMEGA[b]  = (coef_t)(std::cos(omega));
        cfg.COS_OMEGA2[b] = (coef_t)(2 * std::cos(omega));
        cfg.SIN_OMEGA[b]  = (coef_t)(std::sin(omega));
    }
}

// ---------------------------------------------
// Synthesizable Goertzel kernel (AXI-Lite + AXI-Stream)
// ---------------------------------------------
void goertzel(hls::stream<AxisWordSampleIn> &in_stream,
              hls::stream<AxisWordDFTc>     &out_stream,
              goertzel_config               &cfg)
{
    AXIS_IN_OUT(in_stream);
    AXIS_IN_OUT(out_stream);
    AP_CTRL_NONE;
    AXIL_CFG(cfg);

    // ---------------------------------------------------------------------
    // Constant ROM binding for Hann window (fixed-point, precomputed)
    // ---------------------------------------------------------------------
#pragma HLS BIND_STORAGE variable=hann_window type=rom_1p impl=bram

#pragma HLS ARRAY_PARTITION variable=cfg.COS_OMEGA complete
#pragma HLS ARRAY_PARTITION variable=cfg.COS_OMEGA2 complete
#pragma HLS ARRAY_PARTITION variable=cfg.SIN_OMEGA complete

    static DFT_t s_prev[NBINS];
    static DFT_t s_prev2[NBINS];
#pragma HLS ARRAY_PARTITION variable=s_prev complete
#pragma HLS ARRAY_PARTITION variable=s_prev2 complete

    static int elem   = 0;
    static int sample = 0;

    if (!in_stream.empty()) {
        AxisWordSampleIn in = in_stream.read();
        sampleIn_t xin = in.data;

        // Apply Hann window directly from ROM (fixed-point)
        win_t w = hann_window[sample];
        DFT_t xw = (DFT_t)xin * (DFT_t)w;

        DFT_t s_curr[NBINS];
        DFT_t out_real = 0;
        DFT_t out_imag = 0;
#pragma HLS ARRAY_PARTITION variable=s_curr complete

        // Goertzel recursion: s[n] = x + 2*cos(ω)*s[n-1] - s[n-2]
        for (int b = 0; b < NBINS; ++b) {
#pragma HLS UNROLL
            coef_t cosw2 = cfg.COS_OMEGA2[b];
            s_curr[b] = xw + cosw2 * s_prev[b] - s_prev2[b];
            s_prev2[b] = s_prev[b];
            s_prev[b]  = s_curr[b];
        }

        sample++;

        // End of window for this element → output complex result
        if (sample == N_WIN) {
            for (int b = 0; b < NBINS; ++b) {
#pragma HLS UNROLL
                coef_t cosw = cfg.COS_OMEGA[b];
                coef_t sinw = cfg.SIN_OMEGA[b];
                out_real += s_prev2[b] - s_prev[b] * cosw;
                out_imag += s_prev[b] * sinw;
                s_prev[b]  = (DFT_t)0;
                s_prev2[b] = (DFT_t)0;
            }

            out_stream.write(AxisWordDFTc(out_real, out_imag));
            sample = 0;
            elem++;
            if (elem == N_ELEM) elem = 0;
        }
    }
}
