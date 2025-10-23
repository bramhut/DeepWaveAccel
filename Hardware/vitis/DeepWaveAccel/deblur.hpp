#pragma once
#include "types.hpp"

// ---------------------------------------------------------
// Problem sizes
// ---------------------------------------------------------
constexpr int ND        = 6;     // # off-diagonals (excl. main)
constexpr int MAX_ORDER = 22;    // default Chebyshev order (θ_0..θ_K)

// ---------------------------------------------------------
// Fixed-point types (tune to your project types)
// ---------------------------------------------------------
using lap_t     = ap_ufixed<15, -1>;  // Laplacian
using theta_t   = img_t;             // Theta
using acc_t     = ap_fixed<img_t::width+6, 2>;   // accumulator
using idx_t     = ap_uint<12>;       // diagonal offset (>=0)

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
//   - lap_stream: first 1 word = main diagonal scalar,
//                 then ND*IMG_LEN words = off-diagonal lines
// ---------------------------------------------------------
void deblur(
    hls::stream<AxisWordImg> &bp_stream,     // input backprojection image (IMG_LEN)
    hls::stream<lap_t> &lap_stream,    // Laplacian coeff stream (main, then off-diagonals)
    hls::stream<AxisWordImg> &img_stream,    // output image (IMG_LEN)
    deblur_config            &cfg            // AXI-Lite
);