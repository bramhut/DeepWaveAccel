// top.hpp
#pragma once
#include "types.hpp"
#include "goertzel.hpp"
#include "crosscor.hpp"
// #include "backproj.hpp"
// #include "deblur.hpp"

void deepwaveaccel( // top kernel
    hls::stream<AxisWordSampleIn>   &in,
    hls::stream<AxisWordDFTc>       &out,
    goertzel_config                 &goer_cfg);
    // backproj_config       &bpp_cfg,
    // deblur_config         &db_cfg)