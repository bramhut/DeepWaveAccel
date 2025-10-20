#include "goertzel.hpp"
#include <cmath>
#include <iostream>

// ---------------------------------------------
// Host-side preparation (non-synth)
// ---------------------------------------------
void goertzel_prepare_config(goertzel_config &cfg, double fs_in, double ff) {
    cfg.fs_in = fs_in;
    cfg.ff = ff;

    double fr = fs_in / double(N_WIN);
    int bin = int(std::round(ff / fr));
    int bins[NBINS] = { bin, bin - 1 };

    // Hann window coefficients
    for (int n = 0; n < N_WIN; ++n) {
        double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (N_WIN - 1)));
        cfg.hann[n] = (win_t)w;
    }

    // Goertzel coefficients (no *2)
    for (int b = 0; b < NBINS; ++b) {
        double omega = 2.0 * M_PI * bins[b] / double(N_WIN);
        cfg.COS_OMEGA[b]  = (coef_t)(std::cos(omega));
        cfg.COS_OMEGA2[b] = (coef_t)(2*std::cos(omega));
        cfg.SIN_OMEGA[b]  = (coef_t)(std::sin(omega));
    }
}

// ---------------------------------------------
// Synthesizable Goertzel kernel (AXI-Lite + AXI-Stream)
// ---------------------------------------------
void goertzel(hls::stream<AxisWordSampleIn> &in_stream,
              hls::stream<AxisWordDFTc> &out_stream,
              goertzel_config &cfg)
{
    AXIS_IN_OUT(in_stream);
    AXIS_IN_OUT(out_stream);
    AP_CTRL_NONE;
    AXIL_CFG(cfg);
#pragma HLS PIPELINE II=1

    static DFT_t s_prev[NBINS];
    static DFT_t s_prev2[NBINS];
#pragma HLS ARRAY_PARTITION variable=s_prev complete
#pragma HLS ARRAY_PARTITION variable=s_prev2 complete

    static int elem = 0;
    static int sample = 0;

    // const coef_t cos_omega_test[2] = {"0b00000.1100101001001","0b00000.1100111100011"};
    // const coef_t cos_omega_test2[2] = {"0b00001.1001010010010","0b00001.1001111000111"};
    // const coef_t sin_omega_test[2] = {"0b00000.1001110011101","0b00000.1001011001111"};

    if (!in_stream.empty()) {
        AxisWordSampleIn in = in_stream.read();
        sampleIn_t xin = in.data;

        // Apply Hann window
        win_t w = cfg.hann[sample];
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
            s_prev[b] = s_curr[b];
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
