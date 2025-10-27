#pragma once
#include "types.hpp"

// ---------------- Config ----------------
constexpr int GROUP_FRAMES = 9;

// Internal types
using norm_inv_t = ap_ufixed<24,1>;
using norm_sum_t = ap_ufixed<25,10>; // >24bits, or <=16 otherwise Vitis aligns it to 24bit AXIS (not allowed)
using power_accum_t = ap_fixed<36,10>;
using corr_accum_t = complex<power_accum_t>;


// Top-level synthesizable kernel (AXI-Lite + AXI-Stream)
void crosscor(hls::stream<AxisWordDFTc> &in_stream,
              hls::stream<AxisWordDFTc> &out_stream,
              hls::stream<norm_sum_t> &norm_stream);
