#include "top.hpp"
#include <iostream>

// -----------------------------------------------------------------------------
// Persistent memories (static inside load_params for HLS binding)
// -----------------------------------------------------------------------------
static bool g_params_loaded = false;

static void load_params(const b_t* b_ddr,
                        const tau_t* tau_ddr,
                        const lap_t* lap_ddr,
                        b_t   g_b_mem[IMG_LEN][N_ELEM],
                        tau_t g_tau_mem[IMG_LEN],
                        lap_t &g_lap_main,
                        lap_t g_lap_rest[ND][IMG_LEN])
{
#pragma HLS INLINE

#pragma HLS BIND_STORAGE variable=g_b_mem    type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=g_tau_mem  type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=g_lap_rest type=ram_2p impl=bram
#pragma HLS ARRAY_RESHAPE variable=g_b_mem complete dim=2

    // ---------------------------------------------------------
    // Load steering vectors b_mem from DDR
    // ---------------------------------------------------------
    int idx_b = 0;
    for (int r = 0; r < IMG_LEN; ++r) {
#pragma HLS LOOP_FLATTEN off
        for (int c = 0; c < N_ELEM; ++c) {
#pragma HLS PIPELINE off
            g_b_mem[r][c] = b_ddr[idx_b++];
        }
    }

    // ---------------------------------------------------------
    // Load tau compensation values
    // ---------------------------------------------------------
    for (int i = 0; i < IMG_LEN; ++i) {
#pragma HLS PIPELINE off
        g_tau_mem[i] = tau_ddr[i];
    }

    // ---------------------------------------------------------
    // Load Laplacian main + off-diagonals
    // ---------------------------------------------------------
    g_lap_main = lap_ddr[0];
    int idx_lap = 1;  // start after main
    for (int d = 0; d < ND; ++d) {
#pragma HLS LOOP_FLATTEN off
        for (int i = 0; i < IMG_LEN; ++i) {
#pragma HLS PIPELINE off
            g_lap_rest[d][i] = lap_ddr[idx_lap++];
        }
    }
}


// -----------------------------------------------------------------------------
// 2. Core processing pipeline (DATAFLOW region)
// -----------------------------------------------------------------------------
static void deepwave(hls::stream<sample_t>   &in,
                     hls::stream<out_axis_t> &out,
                     goertzel_config         &goer_cfg,
                     deblur_config           &debl_cfg,
                     const b_t   g_b_mem[IMG_LEN][N_ELEM],
                     const tau_t g_tau_mem[IMG_LEN],
                     const lap_t &g_lap_main,
                     const lap_t g_lap_rest[ND][IMG_LEN])
{
#pragma HLS INLINE off

    // Internal streams
    static hls::stream<AxisWordDFTc> s_goertzel("s_goertzel");
    static hls::stream<AxisWordDFTc> s_xcor    ("s_xcor");
    static hls::stream<img_t>        s_bp      ("s_bp");
    static hls::stream<norm_sum_t>   s_norm    ("s_norm");

#pragma HLS STREAM variable=s_goertzel depth=64
#pragma HLS STREAM variable=s_xcor     depth=64
#pragma HLS STREAM variable=s_bp       depth=64
#pragma HLS STREAM variable=s_norm     depth=4

#pragma HLS STABLE variable=g_b_mem
#pragma HLS STABLE variable=g_tau_mem
#pragma HLS STABLE variable=g_lap_main
#pragma HLS STABLE variable=g_lap_rest

#pragma HLS DATAFLOW

    goertzel      (in,         s_goertzel, goer_cfg);
    crosscor      (s_goertzel, s_xcor,     s_norm);
    backprojection(s_xcor, g_b_mem, g_tau_mem, s_bp);
    deblur        (s_bp, g_lap_main, g_lap_rest, out, s_norm, debl_cfg);
}

// -----------------------------------------------------------------------------
// 3. Top-level kernel entry
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
    // Interfaces
    AXIS_IN_OUT(in);
    AXIS_IN_OUT(out);

    M_AXI_PARAM(b_ddr,   PARAMS);
    M_AXI_PARAM(tau_ddr, PARAMS);
    M_AXI_PARAM(lap_ddr, PARAMS);

    AXIL_CFG(goer_cfg);
    AXIL_CFG(debl_cfg);
    AP_CTRL_NONE;

    // Local static persistent buffers
    static b_t   g_b_mem[IMG_LEN][N_ELEM];
    static tau_t g_tau_mem[IMG_LEN];
    static lap_t g_lap_main;
    static lap_t g_lap_rest[ND][IMG_LEN];

    if (!g_params_loaded) {
        load_params(b_ddr, tau_ddr, lap_ddr,
                    g_b_mem, g_tau_mem, g_lap_main, g_lap_rest);
        g_params_loaded = true;
    }

    deepwave(in, out, goer_cfg, debl_cfg,
             g_b_mem, g_tau_mem, g_lap_main, g_lap_rest);
}
