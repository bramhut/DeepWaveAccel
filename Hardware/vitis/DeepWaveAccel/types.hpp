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

constexpr int N_PAIR = N_ELEM * (N_ELEM - 1) / 2; // number of upper triangle elements

// ---------------- Shared Fixed-Point Types ----------------
using std::complex;

// Goertzel
using coef_t        = ap_fixed<18, 2>;   // sfix18_En16
using win_t         = ap_ufixed<12, 0>;  // ufix12_En12

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
using norm_sum_t    = ap_fixed<25, 10>;    // sfix25_En15
using norm_out_t    = ap_ufixed<18, 7>;    // ufix18_En11

// ---------------- Stream Word Structures ----------------
struct AxisWordDFTc {
    DFT_t re, im;
    AxisWordDFTc() {}
    AxisWordDFTc(DFT_t r, DFT_t i) : re(r), im(i) {}
    AxisWordDFTc(DFTc_t d) : re(d.real()), im(d.imag()) {}
};

// unified 32-bit AXIS output (1 norm + IMG_LEN pixels per frame)
using word_t = ap_uint<32>;

using out_axis_t = hls::axis<ap_uint<32>, 0, 0, 0, (AXIS_ENABLE_LAST | AXIS_ENABLE_DATA)>;

// Status registers
struct status_gz_t {
    ap_uint<32> samples_in;       // total samples read from input AXIS
    ap_uint<32> sample_win;    // current sample index (0..N_WIN-1) 
    ap_uint<32> samples_out;     // number of complex outputs written 
    ap_uint<32> samples_out_fifo; // Number of samples in the output buf
};

struct status_cc_t {
    ap_uint<8>  state;          // FSM state (COLLECT=0, CORRELATE=1, OUTPUT=2)
    ap_uint<32> samples_in;     // number of complex samples read
    ap_uint<32> samples_out;    // number of complex pairs written to out_stream
    ap_uint<32> sample_idx;     // current v_count (0..N_ELEM-1)
    ap_uint<32> current_norm;   // current norm value
    ap_uint<32> norms_written;  // number of norm values written
    ap_uint<32> out_fifo;       // out_stream.size()
    ap_uint<32> norms_fifo;     // norm_stream.size()
};

struct status_bp_t {
    ap_uint<1>  config_loaded;     // true when all params loaded
    ap_uint<8>  fsm_state;         // main FSM state enum
    ap_uint<8>  param_state;       // param FSM state enum
    ap_uint<16> idx;               // index (state dependent)
    ap_uint<32> sigmas_in;         // total # Sigma_up words received in LOAD_UP
    ap_uint<32> pixels_out;       // total # frames fully processed
    ap_uint<16> out_fifo_level;    // img_stream.size()
};

struct status_db_t {
    ap_uint<1>  config_loaded;   // finished loading K, theta, offsets, lap
    ap_uint<8>  fsm_state;       // outer FSM: LOAD_PARAMS..OUTPUT
    ap_uint<8>  param_state;     // inner FSM: READ_K..READ_REST
    ap_uint<16> idx;             // meaning depends on param_state (i)
    ap_uint<32> pixels_in;       // total # frames received from backprojection
    ap_uint<32> pixels_out;     // how many deblurred frames we output
};



// ---------------- Stringizing Helpers ----------------
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// ---------------- Default Directories ----------------
#define OUTPUT_DIR "../../../../output"
#define PARAM_DIR  "../../../../parameters"
#define WAVE_DIR "../../../../../../../Simulation/FRIDA/FRIDA/recordings/20160908/data_pyramic/segmented"

// ---------------- Interface Macros ----------------
#define AXIS_IN_OUT(NAME) \
  _Pragma(TOSTRING(HLS INTERFACE axis port=NAME))

#define AXIL_CFG(NAME) \
  _Pragma(TOSTRING(HLS INTERFACE s_axilite port=NAME bundle=CTRL_BUS))

#define AXIL_NOAGGREGATE(NAME) \
  _Pragma(TOSTRING(HLS DISAGGREGATE variable=NAME))

#define M_AXI_PARAM(NAME,BUNDLE) \
  _Pragma(TOSTRING(HLS INTERFACE m_axi port=NAME offset=slave bundle=BUNDLE max_read_burst_length=256))

#define S_AXILITE_SCALAR(NAME) \
  _Pragma(TOSTRING(HLS INTERFACE s_axilite port=NAME bundle=CTRL_BUS))

#define AP_CTRL_NONE \
  _Pragma("HLS INTERFACE ap_ctrl_none port=return")