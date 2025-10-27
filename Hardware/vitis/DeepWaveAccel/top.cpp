// -----------------------------------------------------------------------------
// top.cpp  –  DeepWaveAccel top-level kernel
//             (Goertzel → Crosscor → Backproj → Deblur)
// -----------------------------------------------------------------------------
#include "top.hpp"

void deepwaveaccel(
    hls::stream<sample_t> &in,      // raw input samples
    hls::stream<b_t>              &b_in,    // steering vectors (streamed once)
    hls::stream<tau_t>            &tau_in,  // tau compensation values (streamed once)
    hls::stream<lap_t>            &lap_in,  // Laplacian coefficient stream (PS→PL)
    hls::stream<img_axis_t>       &out,     // final deblurred image output
    hls::stream<norm_sum_t>       &norm,    // crosscor normalization info
    goertzel_config               &goer_cfg,
    deblur_config                 &debl_cfg)
{
    // -------------------------------------------------------------------------
    // Interface pragmas
    // -------------------------------------------------------------------------
    AXIS_IN_OUT(in);
    AXIS_IN_OUT(b_in);
    AXIS_IN_OUT(tau_in);
    AXIS_IN_OUT(lap_in);
    AXIS_IN_OUT(out);
    AXIS_IN_OUT(norm);
    AXIL_CFG(goer_cfg);
    AXIL_CFG(debl_cfg);
    AP_CTRL_NONE;

#pragma HLS DATAFLOW

    // -------------------------------------------------------------------------
    // Intermediate streams
    // -------------------------------------------------------------------------
    static hls::stream<AxisWordDFTc> s_goertzel("s_goertzel");
    static hls::stream<AxisWordDFTc> s_xcor("s_xcor");
    static hls::stream<img_t>  s_bp("s_bp"); // backprojection output
#pragma HLS STREAM variable=s_goertzel depth=64
#pragma HLS STREAM variable=s_xcor     depth=256
#pragma HLS STREAM variable=s_bp       depth=512

    // -------------------------------------------------------------------------
    // Pipeline: Goertzel → Crosscor → Backprojection → Deblur
    // -------------------------------------------------------------------------
    goertzel      (in,         s_goertzel, goer_cfg);
    crosscor      (s_goertzel, s_xcor,     norm);
    backprojection(s_xcor,     b_in, tau_in, s_bp);
    deblur        (s_bp,       lap_in, out, debl_cfg);
}
