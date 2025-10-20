#pragma once
#include <ap_fixed.h>
#include <hls_stream.h>
#include <complex>

// Shared 
using std::complex;

// ---------------- Global Config ----------------
constexpr int N_ELEM  = 48;
constexpr int IMG_LEN = 2234;

// Consts
constexpr int NPAIR = (N_ELEM * (N_ELEM - 1)) / 2;

// Fixed-point types (Simulink mapping)
using sampleIn_t    = ap_fixed<12, 1>;   // sfix12_En11
using DFT_t         = ap_fixed<18, 5>;   // sfix18_En13
using DFTc_t        = complex<DFT_t>;

// Axis types
struct AxisWordSampleIn {
    sampleIn_t data;
    ap_uint<1> last;
    ap_uint<1> user;
    AxisWordSampleIn() {}
    AxisWordSampleIn(sampleIn_t d, bool l=false, bool u=false)
        : data(d), last(l), user(u) {}
};

struct AxisWordDFTc {
    DFT_t re, im;
    ap_uint<1> last;
    ap_uint<1> user;
    AxisWordDFTc() {}
    AxisWordDFTc(DFT_t r, DFT_t i, bool l=false, bool u=false)
        : re(r), im(i), last(l), user(u) {}
    AxisWordDFTc(DFTc_t d, bool l=false, bool u=false) 
        : re(d.real()), im(d.imag()), last(l), user(u) {}
};

// ROM constructors

// ---------------------------------------------------------
// Build (j,k) ROM for upper-triangle output order
// ---------------------------------------------------------
static void build_pair_rom(int j_rom[NPAIR], int k_rom[NPAIR]) {
#pragma HLS INLINE
    int p = 0;
    for (int j = 0; j < N_ELEM; ++j) {
        for (int k = j + 1; k < N_ELEM; ++k) {
#pragma HLS PIPELINE II=1
            j_rom[p] = j;
            k_rom[p] = k;
            ++p;
        }
    }
}

// ---- Stringizing helpers ----
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define OUTPUT_DIR "../../../../output"

// ---- Interface macros ----
#define AXIS_IN_OUT(NAME) \
  _Pragma(TOSTRING(HLS INTERFACE axis port=NAME))

#define AXIL_CFG(NAME) \
  _Pragma(TOSTRING(HLS INTERFACE s_axilite port=NAME bundle=CTRL_BUS))

#define AP_CTRL_NONE \
  _Pragma("HLS INTERFACE ap_ctrl_none port=return")

