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
    const goertzel_config           &goer_cfg,
    const deblur_config             &debl_cfg,
    status_gz_t               &status_gz,
    status_cc_t               &status_cc,
    status_bp_t               &status_bp,
    status_db_t               &status_db)
{
    AP_CTRL_NONE;

    // Interfaces
    AXIS_IN_OUT(in);
    AXIS_IN_OUT(param_bp);
    AXIS_IN_OUT(param_db);
    AXIS_IN_OUT(out);

    AXIL_CFG(goer_cfg);
    AXIL_CFG(debl_cfg);
    AXIL_CFG(status_gz);
    AXIL_CFG(status_cc);
    AXIL_CFG(status_bp);
    AXIL_CFG(status_db);

    AXIL_NOAGGREGATE(goer_cfg);
    AXIL_NOAGGREGATE(debl_cfg);
    AXIL_NOAGGREGATE(status_gz);
    AXIL_NOAGGREGATE(status_cc);
    AXIL_NOAGGREGATE(status_bp);
    AXIL_NOAGGREGATE(status_db);
#pragma HLS DATAFLOW

    // Internal streams
    hls_thread_local hls::stream<AxisWordDFTc, N_ELEM> s_goertzel("s_goertzel");
    hls_thread_local hls::stream<AxisWordDFTc, N_PAIR> s_xcor    ("s_xcor");
    hls_thread_local hls::stream<img_t, IMG_LEN> s_bp      ("s_bp");
    hls_thread_local hls::stream<norm_sum_t, 4> s_norm    ("s_norm");

    hls_thread_local hls::task goertzel_task (goertzel, in,         s_goertzel, goer_cfg, status_gz);
    hls_thread_local hls::task crosscor_task(crosscor, s_goertzel, s_xcor,     s_norm, status_cc);
    hls_thread_local hls::task backprojection_task(backprojection, s_xcor, param_bp, s_bp, status_bp);
    hls_thread_local hls::task deblur_task(deblur, s_bp, param_db, out, s_norm, debl_cfg, status_db);

}
