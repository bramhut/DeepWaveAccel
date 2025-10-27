#pragma once
#include <ap_fixed.h>
#include <hls_stream.h>
#include <complex>

// Shared 
using std::complex;

// ---------------- Global Config ----------------
constexpr int N_ELEM  = 48;
constexpr int IMG_LEN = 2234;

// Fixed-point types (Simulink mapping)
using sample_t      = ap_fixed<12, 1>;   // sfix12_En11
using DFT_t         = ap_fixed<18, 5>;   // sfix18_En13
using DFTc_t        = complex<DFT_t>;
using b_real_t      = ap_fixed<14, -2>;            // sfix14_En16
using b_t           = std::complex<b_real_t>;      // complex steering coefficients
using tau_t         = ap_fixed<13, -3>;            // sfix13_En16 (per-pixel)
using img_t         = ap_fixed<18,  2>;            // sfix18_En16 (output pixel)
using img_axis_t    = ap_fixed<32, 16>;            // 32 bit aligned version of img_t for AXIS


struct AxisWordDFTc {
    DFT_t re, im;
    AxisWordDFTc() {}
    AxisWordDFTc(DFT_t r, DFT_t i)
        : re(r), im(i) {}
    AxisWordDFTc(DFTc_t d) 
        : re(d.real()), im(d.imag()) {}
};
// ---- Stringizing helpers ----
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define OUTPUT_DIR "../../../../output"
#define PARAM_DIR  "../../../../parameters"
#define WAVE_DIR  "../../../../../../../Simulation/FRIDA"

// ---- Interface macros ----
#define AXIS_IN_OUT(NAME) \
  _Pragma(TOSTRING(HLS INTERFACE axis port=NAME))

#define AXIL_CFG(NAME) \
  _Pragma(TOSTRING(HLS INTERFACE s_axilite port=NAME bundle=CTRL_BUS))

#define AP_CTRL_NONE \
  _Pragma("HLS INTERFACE ap_ctrl_none port=return")

