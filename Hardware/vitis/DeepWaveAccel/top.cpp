#include "top.hpp"
#include <hls_task.h>
#include <iostream>

// -----------------------------------------------------------------------------
// Top-level kernel entry
// -----------------------------------------------------------------------------
void deepwaveaccel(
    hls::stream<word_t>     &in,
    hls::stream<word_t> &param_bp,
    hls::stream<word_t> &param_db,
    hls::stream<out_axis_t>   &out,
    goertzel_config           &goer_cfg,
    deblur_config             &debl_cfg)
{
    AP_CTRL_NONE;

    // Interfaces
    AXIS_IN_OUT(in);
    AXIS_IN_OUT(param_bp);
    AXIS_IN_OUT(param_db);
    AXIS_IN_OUT(out);

    AXIL_CFG(goer_cfg);
    AXIL_CFG(debl_cfg);

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
    hls_thread_local hls::task backprojection_task(backprojection, s_xcor, param_bp, s_bp);
    hls_thread_local hls::task deblur_task(deblur, s_bp, param_db, out, s_norm, debl_cfg);

}
