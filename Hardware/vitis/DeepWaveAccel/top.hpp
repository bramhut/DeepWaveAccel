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
    const b_t*              b_ddr,  // steering vectors in DDR
    const tau_t*            tau_ddr,// tau values in DDR
    const lap_t*            lap_ddr,// Laplacian coefficients in DDR
    hls::stream<out_axis_t> &out,   // unified 32-bit output stream
    goertzel_config         &goer_cfg,
    deblur_config           &debl_cfg);
