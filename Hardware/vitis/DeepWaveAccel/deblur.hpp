#pragma once
// -----------------------------------------------------------------------------
// deblur.hpp – Deblur kernel (Chebyshev filtering with sparse Laplacian)
// -----------------------------------------------------------------------------
#include "types.hpp"

// ---------------------------------------------------------
// Problem sizes (Deblur-local)
// ---------------------------------------------------------
constexpr int ND        = 6;     // # off-diagonals (excl. main)
constexpr int MAX_ORDER = 22;    // Chebyshev order (θ_0..θ_K)

// ---------------------------------------------------------
// Deblur-local types
// ---------------------------------------------------------
using acc_t   = ap_fixed<img_t::width + 6, 2>; // accumulator

// ---------------------------------------------------------
// AXI-Lite configuration
// ---------------------------------------------------------
struct deblur_config {
    ap_uint<8>  n_layers;               // # layers (e.g., 5)
};

// ---------------------------------------------------------
// Top kernel
//   - bp_stream: backprojection image (IMG_LEN per frame)
//   - lap_main, lap_rest: preloaded Laplacian data (global loader)
//   - norm_stream: ONE normalization scalar per frame (from crosscor)
//   - out_stream: 32-bit words, per frame: [norm] + IMG_LEN image pixels
// ---------------------------------------------------------
void deblur(
    hls::stream<img_t>      &bp_stream,
    hls::stream<word_t>     &param_in,
    hls::stream<out_axis_t> &out_stream,
    hls::stream<norm_sum_t> &norm_stream,
    deblur_config           &cfg,
    status_db_t             &status
);
