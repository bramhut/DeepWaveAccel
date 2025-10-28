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
using lap_t   = ap_ufixed<15, -1>;            // Laplacian coeff
using theta_t = img_t;                        // θ_k shares format with img_t
using acc_t   = ap_fixed<img_t::width + 6, 2>; // accumulator
using idx_t   = ap_uint<12>;                  // diagonal offset (>=0)

// ---------------------------------------------------------
// AXI-Lite configuration
// ---------------------------------------------------------
struct deblur_config {
    ap_uint<8>  n_layers;               // # layers (e.g., 5)
    ap_uint<6>  K;                      // Chebyshev order (≤ MAX_ORDER)
    idx_t       lap_off[ND];            // 6 offsets (positive)
    theta_t     theta[MAX_ORDER+1];     // θ_0..θ_K
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
    lap_t                    lap_main,
    const lap_t              lap_rest[ND][IMG_LEN],
    hls::stream<out_axis_t> &out_stream,
    hls::stream<norm_sum_t> &norm_stream,
    deblur_config           &cfg
);
