#pragma once
// -----------------------------------------------------------------------------
// types.hpp  –  Shared types and global configuration for DeepWaveAccel
// -----------------------------------------------------------------------------
#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_task.h>
#include <complex>
#include <ap_axi_sdata.h>

// ---------------- Global Config ----------------
constexpr int N_ELEM  = 48;     // number of array elements
constexpr int IMG_LEN = 2234;   // pixels per frame

// ---------------- Shared Fixed-Point Types ----------------
using std::complex;

using sample_t      = ap_fixed<12, 1>;    // sfix12_En11 – input samples
using DFT_t         = ap_fixed<18, 5>;    // sfix18_En13 – DFT values
using DFTc_t        = complex<DFT_t>;

using b_real_t      = ap_fixed<14, -2>;   // sfix14_En16
using b_t           = std::complex<b_real_t>; // complex steering coefficients
using lap_t         = ap_ufixed<15, -1>;              // Laplacian coeff
using tau_t         = ap_fixed<13, -3>;   // sfix13_En16 (per-pixel tau)

using img_t         = ap_fixed<18,  2>;   // sfix18_En16 (image pixel)
using img_axis_t    = ap_fixed<32, 16>;   // aligned 32-bit AXIS representation

using idx_t   = ap_uint<12>;                  // diagonal offset (>=0)
using theta_t = img_t;                        // θ_k shares format with img_t

// normalization info (produced once per frame in crosscor)
using norm_sum_t    = ap_fixed<25, 10>;    // 25-bit scalar

// ---------------- Stream Word Structures ----------------
struct AxisWordDFTc {
    DFT_t re, im;
    AxisWordDFTc() {}
    AxisWordDFTc(DFT_t r, DFT_t i) : re(r), im(i) {}
    AxisWordDFTc(DFTc_t d) : re(d.real()), im(d.imag()) {}
};

// unified 32-bit AXIS output (1 norm + IMG_LEN pixels per frame)
using out_word_t = ap_uint<32>;
using out_axis_t = ap_axiu<32,0,0,0>;

// ---------------- Stringizing Helpers ----------------
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// ---------------- Default Directories ----------------
#define OUTPUT_DIR "../../../../output"
#define PARAM_DIR  "../../../../parameters"
#define WAVE_DIR   "../../../../../../../Simulation/FRIDA"

// ---------------- Interface Macros ----------------
#define AXIS_IN_OUT(NAME) \
  _Pragma(TOSTRING(HLS INTERFACE axis port=NAME))

#define AXIL_CFG(NAME) \
  _Pragma(TOSTRING(HLS INTERFACE s_axilite port=NAME bundle=CTRL_BUS))

#define M_AXI_PARAM(NAME,BUNDLE) \
  _Pragma(TOSTRING(HLS INTERFACE m_axi port=NAME offset=slave bundle=BUNDLE max_read_burst_length=256))

#define S_AXILITE_SCALAR(NAME) \
  _Pragma(TOSTRING(HLS INTERFACE s_axilite port=NAME bundle=CTRL_BUS))
