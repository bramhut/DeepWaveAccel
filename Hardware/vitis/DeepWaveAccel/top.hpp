// top.hpp
#pragma once
#include "types.hpp"
#include "goertzel.hpp"
#include "crosscor.hpp"
#include "backproj.hpp"
// #include "deblur.hpp"

void deepwaveaccel(
    hls::stream<AxisWordSampleIn> &in,     // raw input samples
    hls::stream<b_t>              &b_in,   // steering vectors (streamed once)
    hls::stream<tau_t>            &tau_in, // tau compensation values (streamed once)
    hls::stream<AxisWordImg>      &out,    // final image output
    hls::stream<norm_sum_t>       &norm,   // crosscor normalization info
    goertzel_config               &goer_cfg);