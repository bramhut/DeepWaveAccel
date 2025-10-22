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
using coef_t = ap_fixed<18, 2>;   // Laplacian / Chebyshev coeff
using acc_t  = ap_fixed<24, 4>;   // accumulator
using off_t  = ap_uint<12>;       // diagonal offset (>=0)
using idx_t  = ap_uint<12>;       // pixel index

// Reuse your packed AXIS words for images; Lap stream uses same payload width
using AxisWordImg = ap_axiu<sizeof(img_t)*8, 0, 0, 0>;
using AxisWordLap = ap_axiu<sizeof(coef_t)*8, 0, 0, 0>;

// ---------------------------------------------------------
// AXI-Lite configuration
// ---------------------------------------------------------
struct deblur_config {
    ap_uint<8>  n_layers;               // # layers (e.g., 5)
    ap_uint<6>  K;                      // Chebyshev order (≤ MAX_ORDER)
    off_t       lap_off[ND];            // 6 offsets (positive)
    coef_t      theta[MAX_ORDER+1];     // θ_0..θ_K
};

// ---------------------------------------------------------
// Top kernel
//   - lap_stream: first 1 word = main diagonal scalar,
//                 then ND*IMG_LEN words = off-diagonal lines
// ---------------------------------------------------------
void deblur(
    hls::stream<AxisWordImg> &bp_stream,     // input backprojection image (IMG_LEN)
    hls::stream<AxisWordLap> &lap_stream,    // Laplacian coeff stream (main, then off-diagonals)
    hls::stream<AxisWordImg> &img_stream,    // output image (IMG_LEN)
    deblur_config            &cfg            // AXI-Lite
);
