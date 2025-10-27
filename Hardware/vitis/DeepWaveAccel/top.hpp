// -----------------------------------------------------------------------------
// top.hpp  –  DeepWaveAccel top-level kernel declarations
// -----------------------------------------------------------------------------
#pragma once

#include "types.hpp"
#include "goertzel.hpp"
#include "crosscor.hpp"
#include "backproj.hpp"
#include "deblur.hpp"

// -----------------------------------------------------------------------------
// Combined top-level kernel (with Laplacian stream input)
// -----------------------------------------------------------------------------
void deepwaveaccel(
    hls::stream<sample_t> &in,      // raw input samples
    hls::stream<b_t>              &b_in,    // steering vectors (streamed once)
    hls::stream<tau_t>            &tau_in,  // tau compensation values (streamed once)
    hls::stream<lap_t>            &lap_in,  // Laplacian coefficient stream (PS→PL)
    hls::stream<img_t>      &out,     // final deblurred image output
    hls::stream<norm_sum_t>       &norm,    // crosscor normalization info
    goertzel_config               &goer_cfg,
    deblur_config                 &debl_cfg);
