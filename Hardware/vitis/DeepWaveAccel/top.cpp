// top.cpp
#include "top.hpp"


void deepwaveaccel( // top kernel
    hls::stream<AxisWordSampleIn>   &in,
    hls::stream<AxisWordDFTc>       &out,
    hls::stream<norm_sum_t>         &norm,
    goertzel_config                 &goer_cfg)
    // backproj_config       &bpp_cfg,
    // deblur_config         &db_cfg)
{
    AXIS_IN_OUT(in);
    AXIS_IN_OUT(out);
    AXIS_IN_OUT(norm);
    AXIL_CFG(goer_cfg);
    // AXIL_CFG(bpp_cfg);
    // AXIL_CFG(db_cfg);
    AP_CTRL_NONE;

#pragma HLS DATAFLOW

    static hls::stream<AxisWordDFTc> s_goertzel("s_goertzel");
    static hls::stream<AxisWordDFTc> s_xcor("s_xcor");
    // static hls::stream<axis_img>  s_bpp("s_bpp");
#pragma HLS STREAM variable=s_goertzel depth=64
#pragma HLS STREAM variable=s_xcor depth=256
// #pragma HLS STREAM variable=s_bpp   depth=256

    goertzel      (in,         s_goertzel, goer_cfg);
    crosscor     (s_goertzel, out, norm        );
    // backprojection(s_xcor,     s_bpp,      bpp_cfg);
    // deblur        (s_bpp,      out,        db_cfg);
}
