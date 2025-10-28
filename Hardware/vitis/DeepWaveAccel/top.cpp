// -----------------------------------------------------------------------------
// top.cpp – DeepWaveAccel top-level kernel
//             (Goertzel → Crosscor → Backproj → Deblur)
//             using global DDR→BRAM loaders & unified 32-bit output
// -----------------------------------------------------------------------------
#include "top.hpp"

// -----------------------------------------------------------------------------
// Simple burst loaders (memcpy-like helpers)
// -----------------------------------------------------------------------------
template<typename T>
static void burst_load(const T* __restrict src, T* __restrict dst, int len) {
#pragma HLS INLINE off
load_loop:
    for (int i = 0; i < len; ++i) {
    #pragma HLS PIPELINE II=1
        dst[i] = src[i];
    }
}

template<typename T>
static void burst_load_2d(const T* __restrict src, T dst[][N_ELEM],
                          int rows, int cols) {
#pragma HLS INLINE off
    int idx = 0;
outer:
    for (int r = 0; r < rows; ++r) {
    inner:
        for (int c = 0; c < cols; ++c) {
        #pragma HLS PIPELINE II=1
            dst[r][c] = src[idx++];
        }
    }
}

// -----------------------------------------------------------------------------
// Top-level kernel
// -----------------------------------------------------------------------------
void deepwaveaccel(
    hls::stream<sample_t>   &in,
    const b_t*              b_ddr,
    const tau_t*            tau_ddr,
    const lap_t*            lap_ddr,
    hls::stream<out_axis_t> &out,
    goertzel_config         &goer_cfg,
    deblur_config           &debl_cfg)
{
    // -------------------------------------------------------------------------
    // Interface definitions
    // -------------------------------------------------------------------------
    AXIS_IN_OUT(in);
    AXIS_IN_OUT(out);

    M_AXI_PARAM(b_ddr,   PARAMS);
    M_AXI_PARAM(tau_ddr, PARAMS);
    M_AXI_PARAM(lap_ddr, PARAMS);

    AXIL_CFG(goer_cfg);
    AXIL_CFG(debl_cfg);
    AP_CTRL_NONE;

#pragma HLS DATAFLOW

    // -------------------------------------------------------------------------
    // Persistent global parameter storage (loaded once)
    // -------------------------------------------------------------------------
    static bool params_loaded = false;

    static b_t   b_mem[IMG_LEN][N_ELEM];
    static tau_t tau_mem[IMG_LEN];
    static lap_t lap_main;
    static lap_t lap_rest[ND][IMG_LEN];

#pragma HLS BIND_STORAGE variable=b_mem    type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=tau_mem  type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=lap_rest type=ram_2p impl=bram
#pragma HLS ARRAY_RESHAPE variable=b_mem complete dim=2

    // -------------------------------------------------------------------------
    // Internal streams
    // -------------------------------------------------------------------------
    static hls::stream<AxisWordDFTc> s_goertzel("s_goertzel");
    static hls::stream<AxisWordDFTc> s_xcor("s_xcor");
    static hls::stream<img_t>        s_bp("s_bp");
    static hls::stream<norm_sum_t>   s_norm("s_norm");

#pragma HLS STREAM variable=s_goertzel depth=64
#pragma HLS STREAM variable=s_xcor     depth=64
#pragma HLS STREAM variable=s_bp       depth=64
#pragma HLS STREAM variable=s_norm     depth=4

    // -------------------------------------------------------------------------
    // One-time parameter loading from DDR
    // -------------------------------------------------------------------------
    if (!params_loaded) {
        burst_load_2d(b_ddr,   b_mem,   IMG_LEN, N_ELEM);
        burst_load   (tau_ddr, tau_mem, IMG_LEN);

        lap_main = lap_ddr[0];
        burst_load(&lap_ddr[1], (lap_t*)lap_rest, ND * IMG_LEN);

        params_loaded = true;
    }

    // -------------------------------------------------------------------------
    // Compute pipeline
    // -------------------------------------------------------------------------
    goertzel      (in,         s_goertzel, goer_cfg);
    crosscor      (s_goertzel, s_xcor, s_norm);       // 1 norm per frame
    backprojection(s_xcor,     b_mem, tau_mem, s_bp); // uses static arrays
    deblur        (s_bp,       lap_main, lap_rest, out, s_norm, debl_cfg);
}
