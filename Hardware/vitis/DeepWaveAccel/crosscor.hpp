#pragma once
#include "types.hpp"

// ---------------- Config ----------------
constexpr int GROUP_FRAMES = 9;

// Internal types
using norm_inv_t = ap_ufixed<24,1>;
using norm_sum_t = ap_ufixed<23,10>;
using corr_accum_t = complex<ap_fixed<36,10>>;

// Top-level synthesizable kernel (AXI-Lite + AXI-Stream)
void crosscor(hls::stream<AxisWordDFTc> &in_stream,
              hls::stream<AxisWordDFTc> &out_stream);
