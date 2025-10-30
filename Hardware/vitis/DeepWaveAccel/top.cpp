#include "top.hpp"
#include <iostream>

// -----------------------------------------------------------------------------
// 3. Top-level kernel entry
// -----------------------------------------------------------------------------
void deepwaveaccel(
    hls::stream<sample_t>   &in,
    hls::stream<out_axis_t> &out,
    goertzel_config         &goer_cfg,
    deblur_config           &debl_cfg)
{
    // Interfaces
    AXIS_IN_OUT(in);
    AXIS_IN_OUT(out);

    AXIL_CFG(goer_cfg);
    AXIL_CFG(debl_cfg);

#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

#pragma HLS DATAFLOW

    // Internal streams
    hls_thread_local hls::stream<AxisWordDFTc> s_goertzel("s_goertzel");
    hls_thread_local hls::stream<AxisWordDFTc> s_xcor    ("s_xcor");
    hls_thread_local hls::stream<img_t>        s_bp      ("s_bp");
    hls_thread_local hls::stream<norm_sum_t>   s_norm    ("s_norm");

#pragma HLS STREAM variable=s_goertzel depth=64
#pragma HLS STREAM variable=s_xcor     depth=64
#pragma HLS STREAM variable=s_bp       depth=64
#pragma HLS STREAM variable=s_norm     depth=4

    hls_thread_local hls::task goertzel_task (goertzel, in,         s_goertzel, goer_cfg);
    hls_thread_local hls::task crosscor_task(crosscor, s_goertzel, s_xcor,     s_norm);
    hls_thread_local hls::task backprojection_task(backprojection, s_xcor, s_bp);
    hls_thread_local hls::task deblur_task(deblur, s_bp, out, s_norm, debl_cfg);

}
