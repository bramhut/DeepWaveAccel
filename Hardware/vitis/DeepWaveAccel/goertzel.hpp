#pragma once
#include "types.hpp"

// ---------------- Config ----------------
constexpr int N_WIN  = 200;     // samples per element window
constexpr int NBINS  = 2;       // number of Goertzel bins
constexpr double FF = 1666.67;  // frequency of interest

// AXI-Lite config interface structure
struct goertzel_config {
    coef_t COS_OMEGA[NBINS];
    coef_t COS_OMEGA2[NBINS];
    coef_t SIN_OMEGA[NBINS];
};

// Host-side configuration update (non-synth)
void goertzel_prepare_config(goertzel_config &cfg, double fs_in, double ff);

// Top-level synthesizable kernel (AXI-Lite + AXI-Stream)
void goertzel(hls::stream<word_t> &in_stream,
              hls::stream<AxisWordDFTc> &out_stream,
              goertzel_config &cfg,
              status_gz_t &status);
