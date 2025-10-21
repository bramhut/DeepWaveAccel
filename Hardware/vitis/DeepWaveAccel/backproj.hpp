#pragma once
#include "types.hpp"
#include <complex>

// ---------------- Fixed-point types for Backprojection ----------------

using acc_fix_t  = ap_fixed<24,  0>;            // sfix24_En24 (accumulator, real)
using acc_cplx_t = std::complex<acc_fix_t>;

// Top-level
void backprojection(hls::stream<AxisWordDFTc> &corr_stream,  // Σ: diag then upper (ROM order)
                    hls::stream<b_t>          &b_stream,     // b: N_ELEM complex per pixel
                    hls::stream<tau_t>        &tau_stream,   // tau: IMG_LEN values
                    hls::stream<AxisWordImg>  &img_stream);  // y_i - tau[i] (real)
