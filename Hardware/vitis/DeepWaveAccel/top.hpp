// -----------------------------------------------------------------------------
// top.hpp – DeepWaveAccel top-level declarations
//             (Global parameter loading + unified 32-bit output stream)
// -----------------------------------------------------------------------------
#pragma once

#include "types.hpp"
#include "goertzel.hpp"
#include "crosscor.hpp"
#include "backproj.hpp"
#include "deblur.hpp"

// -----------------------------------------------------------------------------
// Top-level kernel prototype
// -----------------------------------------------------------------------------
void deepwaveaccel(
    hls::stream<sample_t>   &in,    // raw input samples
    hls::stream<out_axis_t> &out,   // unified 32-bit output stream
    goertzel_config         &goer_cfg,
    deblur_config           &debl_cfg);
