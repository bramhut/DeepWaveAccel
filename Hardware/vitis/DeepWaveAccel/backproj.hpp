#pragma once
#include "types.hpp"
#include <complex>

// -------- Types for internal math --------
using tau_t = ap_fixed<13,-3>;
using acc_fix_t  = ap_fixed<24,0>;
using acc_cplx_t = std::complex<acc_fix_t>;

// Real image word (reuse DFT_t as payload)
struct AxisWordImg {
    DFT_t      data;
    ap_uint<1> last;
    ap_uint<1> user;
    AxisWordImg() {}
    AxisWordImg(DFT_t d, bool l=false, bool u=false) : data(d), last(l), user(u) {}
};

// Top: Σ upper-triangle stream, steering-vector stream, tau preload stream, image out
void backprojection(hls::stream<AxisWordDFTc> &corr_stream,  // Σ: N diagonals then NPAIR off-diagonals
                    hls::stream<AxisWordDFTc> &b_stream,     // b_i: N_ELEM complex per pixel
                    hls::stream<tau_t>        &tau_stream,   // preload IMG_LEN tau values (once)
                    hls::stream<AxisWordImg>  &img_stream);  // y_i - tau[i] (real) per pixel
