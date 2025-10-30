#pragma once
// -----------------------------------------------------------------------------
// backproj.hpp – Backprojection kernel
// -----------------------------------------------------------------------------
#include "types.hpp"
#include <complex>

// ---------------- Fixed-point types for Backprojection ----------------
using acc_fix_t  = ap_fixed<24, 0>;             // sfix24_En24 (accumulator, real)
using acc_cplx_t = std::complex<acc_fix_t>;

// -----------------------------------------------------------------------------
// Top-level kernel declaration
// -----------------------------------------------------------------------------
void backprojection(hls::stream<AxisWordDFTc> &corr_stream,  // Σ upper-triangular
                    hls::stream<img_t>        &img_stream);             // output y_i - tau[i]
