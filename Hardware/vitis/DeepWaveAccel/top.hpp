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
    hls::stream<norm_sum_t>         &norm,
    goertzel_config                 &goer_cfg);
    // backproj_config       &bpp_cfg,
    // deblur_config         &db_cfg)