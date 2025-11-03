// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.1 (64-bit)
// Tool Version Limit: 2025.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
`timescale 1ns/1ps
module deepwaveaccel_CTRL_BUS_s_axi
#(parameter
    C_S_AXI_ADDR_WIDTH = 9,
    C_S_AXI_DATA_WIDTH = 32
)(
    input  wire                          ACLK,
    input  wire                          ARESET,
    input  wire                          ACLK_EN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] AWADDR,
    input  wire                          AWVALID,
    output wire                          AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input  wire                          WVALID,
    output wire                          WREADY,
    output wire [1:0]                    BRESP,
    output wire                          BVALID,
    input  wire                          BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] ARADDR,
    input  wire                          ARVALID,
    output wire                          ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] RDATA,
    output wire [1:0]                    RRESP,
    output wire                          RVALID,
    input  wire                          RREADY,
    output wire [17:0]                   goer_cfg_COS_OMEGA_0,
    output wire [17:0]                   goer_cfg_COS_OMEGA_1,
    output wire [17:0]                   goer_cfg_COS_OMEGA2_0,
    output wire [17:0]                   goer_cfg_COS_OMEGA2_1,
    output wire [17:0]                   goer_cfg_SIN_OMEGA_0,
    output wire [17:0]                   goer_cfg_SIN_OMEGA_1,
    output wire [7:0]                    debl_cfg,
    input  wire [31:0]                   status_gz_samples_in,
    input  wire                          status_gz_samples_in_ap_vld,
    input  wire [31:0]                   status_gz_sample_win,
    input  wire                          status_gz_sample_win_ap_vld,
    input  wire [31:0]                   status_gz_samples_out,
    input  wire                          status_gz_samples_out_ap_vld,
    input  wire [31:0]                   status_gz_samples_out_fifo,
    input  wire                          status_gz_samples_out_fifo_ap_vld,
    input  wire [7:0]                    status_cc_state,
    input  wire                          status_cc_state_ap_vld,
    input  wire [31:0]                   status_cc_samples_in,
    input  wire                          status_cc_samples_in_ap_vld,
    input  wire [31:0]                   status_cc_samples_out,
    input  wire                          status_cc_samples_out_ap_vld,
    input  wire [31:0]                   status_cc_sample_idx,
    input  wire                          status_cc_sample_idx_ap_vld,
    input  wire [31:0]                   status_cc_current_norm,
    input  wire                          status_cc_current_norm_ap_vld,
    input  wire [31:0]                   status_cc_norms_written,
    input  wire                          status_cc_norms_written_ap_vld,
    input  wire [31:0]                   status_cc_out_fifo,
    input  wire                          status_cc_out_fifo_ap_vld,
    input  wire [31:0]                   status_cc_norms_fifo,
    input  wire                          status_cc_norms_fifo_ap_vld,
    input  wire [0:0]                    status_bp_config_loaded,
    input  wire                          status_bp_config_loaded_ap_vld,
    input  wire [7:0]                    status_bp_fsm_state,
    input  wire                          status_bp_fsm_state_ap_vld,
    input  wire [7:0]                    status_bp_param_state,
    input  wire                          status_bp_param_state_ap_vld,
    input  wire [15:0]                   status_bp_idx,
    input  wire                          status_bp_idx_ap_vld,
    input  wire [31:0]                   status_bp_sigmas_in,
    input  wire                          status_bp_sigmas_in_ap_vld,
    input  wire [31:0]                   status_bp_pixels_out,
    input  wire                          status_bp_pixels_out_ap_vld,
    input  wire [15:0]                   status_bp_out_fifo_level,
    input  wire                          status_bp_out_fifo_level_ap_vld,
    input  wire [0:0]                    status_db_config_loaded,
    input  wire                          status_db_config_loaded_ap_vld,
    input  wire [7:0]                    status_db_fsm_state,
    input  wire                          status_db_fsm_state_ap_vld,
    input  wire [7:0]                    status_db_param_state,
    input  wire                          status_db_param_state_ap_vld,
    input  wire [15:0]                   status_db_idx,
    input  wire                          status_db_idx_ap_vld,
    input  wire [31:0]                   status_db_pixels_in,
    input  wire                          status_db_pixels_in_ap_vld,
    input  wire [31:0]                   status_db_pixels_out,
    input  wire                          status_db_pixels_out_ap_vld
);
//------------------------Address Info-------------------
// Protocol Used: ap_ctrl_none
//
// 0x000 : reserved
// 0x004 : reserved
// 0x008 : reserved
// 0x00c : reserved
// 0x010 : Data signal of goer_cfg_COS_OMEGA_0
//         bit 17~0 - goer_cfg_COS_OMEGA_0[17:0] (Read/Write)
//         others   - reserved
// 0x014 : reserved
// 0x018 : Data signal of goer_cfg_COS_OMEGA_1
//         bit 17~0 - goer_cfg_COS_OMEGA_1[17:0] (Read/Write)
//         others   - reserved
// 0x01c : reserved
// 0x020 : Data signal of goer_cfg_COS_OMEGA2_0
//         bit 17~0 - goer_cfg_COS_OMEGA2_0[17:0] (Read/Write)
//         others   - reserved
// 0x024 : reserved
// 0x028 : Data signal of goer_cfg_COS_OMEGA2_1
//         bit 17~0 - goer_cfg_COS_OMEGA2_1[17:0] (Read/Write)
//         others   - reserved
// 0x02c : reserved
// 0x030 : Data signal of goer_cfg_SIN_OMEGA_0
//         bit 17~0 - goer_cfg_SIN_OMEGA_0[17:0] (Read/Write)
//         others   - reserved
// 0x034 : reserved
// 0x038 : Data signal of goer_cfg_SIN_OMEGA_1
//         bit 17~0 - goer_cfg_SIN_OMEGA_1[17:0] (Read/Write)
//         others   - reserved
// 0x03c : reserved
// 0x040 : Data signal of debl_cfg
//         bit 7~0 - debl_cfg[7:0] (Read/Write)
//         others  - reserved
// 0x044 : reserved
// 0x048 : Data signal of status_gz_samples_in
//         bit 31~0 - status_gz_samples_in[31:0] (Read)
// 0x04c : Control signal of status_gz_samples_in
//         bit 0  - status_gz_samples_in_ap_vld (Read/COR)
//         others - reserved
// 0x058 : Data signal of status_gz_sample_win
//         bit 31~0 - status_gz_sample_win[31:0] (Read)
// 0x05c : Control signal of status_gz_sample_win
//         bit 0  - status_gz_sample_win_ap_vld (Read/COR)
//         others - reserved
// 0x068 : Data signal of status_gz_samples_out
//         bit 31~0 - status_gz_samples_out[31:0] (Read)
// 0x06c : Control signal of status_gz_samples_out
//         bit 0  - status_gz_samples_out_ap_vld (Read/COR)
//         others - reserved
// 0x078 : Data signal of status_gz_samples_out_fifo
//         bit 31~0 - status_gz_samples_out_fifo[31:0] (Read)
// 0x07c : Control signal of status_gz_samples_out_fifo
//         bit 0  - status_gz_samples_out_fifo_ap_vld (Read/COR)
//         others - reserved
// 0x088 : Data signal of status_cc_state
//         bit 7~0 - status_cc_state[7:0] (Read)
//         others  - reserved
// 0x08c : Control signal of status_cc_state
//         bit 0  - status_cc_state_ap_vld (Read/COR)
//         others - reserved
// 0x098 : Data signal of status_cc_samples_in
//         bit 31~0 - status_cc_samples_in[31:0] (Read)
// 0x09c : Control signal of status_cc_samples_in
//         bit 0  - status_cc_samples_in_ap_vld (Read/COR)
//         others - reserved
// 0x0a8 : Data signal of status_cc_samples_out
//         bit 31~0 - status_cc_samples_out[31:0] (Read)
// 0x0ac : Control signal of status_cc_samples_out
//         bit 0  - status_cc_samples_out_ap_vld (Read/COR)
//         others - reserved
// 0x0b8 : Data signal of status_cc_sample_idx
//         bit 31~0 - status_cc_sample_idx[31:0] (Read)
// 0x0bc : Control signal of status_cc_sample_idx
//         bit 0  - status_cc_sample_idx_ap_vld (Read/COR)
//         others - reserved
// 0x0c8 : Data signal of status_cc_current_norm
//         bit 31~0 - status_cc_current_norm[31:0] (Read)
// 0x0cc : Control signal of status_cc_current_norm
//         bit 0  - status_cc_current_norm_ap_vld (Read/COR)
//         others - reserved
// 0x0d8 : Data signal of status_cc_norms_written
//         bit 31~0 - status_cc_norms_written[31:0] (Read)
// 0x0dc : Control signal of status_cc_norms_written
//         bit 0  - status_cc_norms_written_ap_vld (Read/COR)
//         others - reserved
// 0x0e8 : Data signal of status_cc_out_fifo
//         bit 31~0 - status_cc_out_fifo[31:0] (Read)
// 0x0ec : Control signal of status_cc_out_fifo
//         bit 0  - status_cc_out_fifo_ap_vld (Read/COR)
//         others - reserved
// 0x0f8 : Data signal of status_cc_norms_fifo
//         bit 31~0 - status_cc_norms_fifo[31:0] (Read)
// 0x0fc : Control signal of status_cc_norms_fifo
//         bit 0  - status_cc_norms_fifo_ap_vld (Read/COR)
//         others - reserved
// 0x108 : Data signal of status_bp_config_loaded
//         bit 0  - status_bp_config_loaded[0] (Read)
//         others - reserved
// 0x10c : Control signal of status_bp_config_loaded
//         bit 0  - status_bp_config_loaded_ap_vld (Read/COR)
//         others - reserved
// 0x118 : Data signal of status_bp_fsm_state
//         bit 7~0 - status_bp_fsm_state[7:0] (Read)
//         others  - reserved
// 0x11c : Control signal of status_bp_fsm_state
//         bit 0  - status_bp_fsm_state_ap_vld (Read/COR)
//         others - reserved
// 0x128 : Data signal of status_bp_param_state
//         bit 7~0 - status_bp_param_state[7:0] (Read)
//         others  - reserved
// 0x12c : Control signal of status_bp_param_state
//         bit 0  - status_bp_param_state_ap_vld (Read/COR)
//         others - reserved
// 0x138 : Data signal of status_bp_idx
//         bit 15~0 - status_bp_idx[15:0] (Read)
//         others   - reserved
// 0x13c : Control signal of status_bp_idx
//         bit 0  - status_bp_idx_ap_vld (Read/COR)
//         others - reserved
// 0x148 : Data signal of status_bp_sigmas_in
//         bit 31~0 - status_bp_sigmas_in[31:0] (Read)
// 0x14c : Control signal of status_bp_sigmas_in
//         bit 0  - status_bp_sigmas_in_ap_vld (Read/COR)
//         others - reserved
// 0x158 : Data signal of status_bp_pixels_out
//         bit 31~0 - status_bp_pixels_out[31:0] (Read)
// 0x15c : Control signal of status_bp_pixels_out
//         bit 0  - status_bp_pixels_out_ap_vld (Read/COR)
//         others - reserved
// 0x168 : Data signal of status_bp_out_fifo_level
//         bit 15~0 - status_bp_out_fifo_level[15:0] (Read)
//         others   - reserved
// 0x16c : Control signal of status_bp_out_fifo_level
//         bit 0  - status_bp_out_fifo_level_ap_vld (Read/COR)
//         others - reserved
// 0x178 : Data signal of status_db_config_loaded
//         bit 0  - status_db_config_loaded[0] (Read)
//         others - reserved
// 0x17c : Control signal of status_db_config_loaded
//         bit 0  - status_db_config_loaded_ap_vld (Read/COR)
//         others - reserved
// 0x188 : Data signal of status_db_fsm_state
//         bit 7~0 - status_db_fsm_state[7:0] (Read)
//         others  - reserved
// 0x18c : Control signal of status_db_fsm_state
//         bit 0  - status_db_fsm_state_ap_vld (Read/COR)
//         others - reserved
// 0x198 : Data signal of status_db_param_state
//         bit 7~0 - status_db_param_state[7:0] (Read)
//         others  - reserved
// 0x19c : Control signal of status_db_param_state
//         bit 0  - status_db_param_state_ap_vld (Read/COR)
//         others - reserved
// 0x1a8 : Data signal of status_db_idx
//         bit 15~0 - status_db_idx[15:0] (Read)
//         others   - reserved
// 0x1ac : Control signal of status_db_idx
//         bit 0  - status_db_idx_ap_vld (Read/COR)
//         others - reserved
// 0x1b8 : Data signal of status_db_pixels_in
//         bit 31~0 - status_db_pixels_in[31:0] (Read)
// 0x1bc : Control signal of status_db_pixels_in
//         bit 0  - status_db_pixels_in_ap_vld (Read/COR)
//         others - reserved
// 0x1c8 : Data signal of status_db_pixels_out
//         bit 31~0 - status_db_pixels_out[31:0] (Read)
// 0x1cc : Control signal of status_db_pixels_out
//         bit 0  - status_db_pixels_out_ap_vld (Read/COR)
//         others - reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

//------------------------Parameter----------------------
localparam
    ADDR_GOER_CFG_COS_OMEGA_0_DATA_0       = 9'h010,
    ADDR_GOER_CFG_COS_OMEGA_0_CTRL         = 9'h014,
    ADDR_GOER_CFG_COS_OMEGA_1_DATA_0       = 9'h018,
    ADDR_GOER_CFG_COS_OMEGA_1_CTRL         = 9'h01c,
    ADDR_GOER_CFG_COS_OMEGA2_0_DATA_0      = 9'h020,
    ADDR_GOER_CFG_COS_OMEGA2_0_CTRL        = 9'h024,
    ADDR_GOER_CFG_COS_OMEGA2_1_DATA_0      = 9'h028,
    ADDR_GOER_CFG_COS_OMEGA2_1_CTRL        = 9'h02c,
    ADDR_GOER_CFG_SIN_OMEGA_0_DATA_0       = 9'h030,
    ADDR_GOER_CFG_SIN_OMEGA_0_CTRL         = 9'h034,
    ADDR_GOER_CFG_SIN_OMEGA_1_DATA_0       = 9'h038,
    ADDR_GOER_CFG_SIN_OMEGA_1_CTRL         = 9'h03c,
    ADDR_DEBL_CFG_DATA_0                   = 9'h040,
    ADDR_DEBL_CFG_CTRL                     = 9'h044,
    ADDR_STATUS_GZ_SAMPLES_IN_DATA_0       = 9'h048,
    ADDR_STATUS_GZ_SAMPLES_IN_CTRL         = 9'h04c,
    ADDR_STATUS_GZ_SAMPLE_WIN_DATA_0       = 9'h058,
    ADDR_STATUS_GZ_SAMPLE_WIN_CTRL         = 9'h05c,
    ADDR_STATUS_GZ_SAMPLES_OUT_DATA_0      = 9'h068,
    ADDR_STATUS_GZ_SAMPLES_OUT_CTRL        = 9'h06c,
    ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_DATA_0 = 9'h078,
    ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_CTRL   = 9'h07c,
    ADDR_STATUS_CC_STATE_DATA_0            = 9'h088,
    ADDR_STATUS_CC_STATE_CTRL              = 9'h08c,
    ADDR_STATUS_CC_SAMPLES_IN_DATA_0       = 9'h098,
    ADDR_STATUS_CC_SAMPLES_IN_CTRL         = 9'h09c,
    ADDR_STATUS_CC_SAMPLES_OUT_DATA_0      = 9'h0a8,
    ADDR_STATUS_CC_SAMPLES_OUT_CTRL        = 9'h0ac,
    ADDR_STATUS_CC_SAMPLE_IDX_DATA_0       = 9'h0b8,
    ADDR_STATUS_CC_SAMPLE_IDX_CTRL         = 9'h0bc,
    ADDR_STATUS_CC_CURRENT_NORM_DATA_0     = 9'h0c8,
    ADDR_STATUS_CC_CURRENT_NORM_CTRL       = 9'h0cc,
    ADDR_STATUS_CC_NORMS_WRITTEN_DATA_0    = 9'h0d8,
    ADDR_STATUS_CC_NORMS_WRITTEN_CTRL      = 9'h0dc,
    ADDR_STATUS_CC_OUT_FIFO_DATA_0         = 9'h0e8,
    ADDR_STATUS_CC_OUT_FIFO_CTRL           = 9'h0ec,
    ADDR_STATUS_CC_NORMS_FIFO_DATA_0       = 9'h0f8,
    ADDR_STATUS_CC_NORMS_FIFO_CTRL         = 9'h0fc,
    ADDR_STATUS_BP_CONFIG_LOADED_DATA_0    = 9'h108,
    ADDR_STATUS_BP_CONFIG_LOADED_CTRL      = 9'h10c,
    ADDR_STATUS_BP_FSM_STATE_DATA_0        = 9'h118,
    ADDR_STATUS_BP_FSM_STATE_CTRL          = 9'h11c,
    ADDR_STATUS_BP_PARAM_STATE_DATA_0      = 9'h128,
    ADDR_STATUS_BP_PARAM_STATE_CTRL        = 9'h12c,
    ADDR_STATUS_BP_IDX_DATA_0              = 9'h138,
    ADDR_STATUS_BP_IDX_CTRL                = 9'h13c,
    ADDR_STATUS_BP_SIGMAS_IN_DATA_0        = 9'h148,
    ADDR_STATUS_BP_SIGMAS_IN_CTRL          = 9'h14c,
    ADDR_STATUS_BP_PIXELS_OUT_DATA_0       = 9'h158,
    ADDR_STATUS_BP_PIXELS_OUT_CTRL         = 9'h15c,
    ADDR_STATUS_BP_OUT_FIFO_LEVEL_DATA_0   = 9'h168,
    ADDR_STATUS_BP_OUT_FIFO_LEVEL_CTRL     = 9'h16c,
    ADDR_STATUS_DB_CONFIG_LOADED_DATA_0    = 9'h178,
    ADDR_STATUS_DB_CONFIG_LOADED_CTRL      = 9'h17c,
    ADDR_STATUS_DB_FSM_STATE_DATA_0        = 9'h188,
    ADDR_STATUS_DB_FSM_STATE_CTRL          = 9'h18c,
    ADDR_STATUS_DB_PARAM_STATE_DATA_0      = 9'h198,
    ADDR_STATUS_DB_PARAM_STATE_CTRL        = 9'h19c,
    ADDR_STATUS_DB_IDX_DATA_0              = 9'h1a8,
    ADDR_STATUS_DB_IDX_CTRL                = 9'h1ac,
    ADDR_STATUS_DB_PIXELS_IN_DATA_0        = 9'h1b8,
    ADDR_STATUS_DB_PIXELS_IN_CTRL          = 9'h1bc,
    ADDR_STATUS_DB_PIXELS_OUT_DATA_0       = 9'h1c8,
    ADDR_STATUS_DB_PIXELS_OUT_CTRL         = 9'h1cc,
    WRIDLE                                 = 2'd0,
    WRDATA                                 = 2'd1,
    WRRESP                                 = 2'd2,
    WRRESET                                = 2'd3,
    RDIDLE                                 = 2'd0,
    RDDATA                                 = 2'd1,
    RDRESET                                = 2'd2,
    ADDR_BITS                = 9;

//------------------------Local signal-------------------
    reg  [1:0]                    wstate = WRRESET;
    reg  [1:0]                    wnext;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [C_S_AXI_DATA_WIDTH-1:0] wmask;
    wire                          aw_hs;
    wire                          w_hs;
    reg  [1:0]                    rstate = RDRESET;
    reg  [1:0]                    rnext;
    reg  [C_S_AXI_DATA_WIDTH-1:0] rdata;
    wire                          ar_hs;
    wire [ADDR_BITS-1:0]          raddr;
    // internal registers
    reg  [17:0]                   int_goer_cfg_COS_OMEGA_0 = 'b0;
    reg  [17:0]                   int_goer_cfg_COS_OMEGA_1 = 'b0;
    reg  [17:0]                   int_goer_cfg_COS_OMEGA2_0 = 'b0;
    reg  [17:0]                   int_goer_cfg_COS_OMEGA2_1 = 'b0;
    reg  [17:0]                   int_goer_cfg_SIN_OMEGA_0 = 'b0;
    reg  [17:0]                   int_goer_cfg_SIN_OMEGA_1 = 'b0;
    reg  [7:0]                    int_debl_cfg = 'b0;
    reg                           int_status_gz_samples_in_ap_vld;
    reg  [31:0]                   int_status_gz_samples_in = 'b0;
    reg                           int_status_gz_sample_win_ap_vld;
    reg  [31:0]                   int_status_gz_sample_win = 'b0;
    reg                           int_status_gz_samples_out_ap_vld;
    reg  [31:0]                   int_status_gz_samples_out = 'b0;
    reg                           int_status_gz_samples_out_fifo_ap_vld;
    reg  [31:0]                   int_status_gz_samples_out_fifo = 'b0;
    reg                           int_status_cc_state_ap_vld;
    reg  [7:0]                    int_status_cc_state = 'b0;
    reg                           int_status_cc_samples_in_ap_vld;
    reg  [31:0]                   int_status_cc_samples_in = 'b0;
    reg                           int_status_cc_samples_out_ap_vld;
    reg  [31:0]                   int_status_cc_samples_out = 'b0;
    reg                           int_status_cc_sample_idx_ap_vld;
    reg  [31:0]                   int_status_cc_sample_idx = 'b0;
    reg                           int_status_cc_current_norm_ap_vld;
    reg  [31:0]                   int_status_cc_current_norm = 'b0;
    reg                           int_status_cc_norms_written_ap_vld;
    reg  [31:0]                   int_status_cc_norms_written = 'b0;
    reg                           int_status_cc_out_fifo_ap_vld;
    reg  [31:0]                   int_status_cc_out_fifo = 'b0;
    reg                           int_status_cc_norms_fifo_ap_vld;
    reg  [31:0]                   int_status_cc_norms_fifo = 'b0;
    reg                           int_status_bp_config_loaded_ap_vld;
    reg  [0:0]                    int_status_bp_config_loaded = 'b0;
    reg                           int_status_bp_fsm_state_ap_vld;
    reg  [7:0]                    int_status_bp_fsm_state = 'b0;
    reg                           int_status_bp_param_state_ap_vld;
    reg  [7:0]                    int_status_bp_param_state = 'b0;
    reg                           int_status_bp_idx_ap_vld;
    reg  [15:0]                   int_status_bp_idx = 'b0;
    reg                           int_status_bp_sigmas_in_ap_vld;
    reg  [31:0]                   int_status_bp_sigmas_in = 'b0;
    reg                           int_status_bp_pixels_out_ap_vld;
    reg  [31:0]                   int_status_bp_pixels_out = 'b0;
    reg                           int_status_bp_out_fifo_level_ap_vld;
    reg  [15:0]                   int_status_bp_out_fifo_level = 'b0;
    reg                           int_status_db_config_loaded_ap_vld;
    reg  [0:0]                    int_status_db_config_loaded = 'b0;
    reg                           int_status_db_fsm_state_ap_vld;
    reg  [7:0]                    int_status_db_fsm_state = 'b0;
    reg                           int_status_db_param_state_ap_vld;
    reg  [7:0]                    int_status_db_param_state = 'b0;
    reg                           int_status_db_idx_ap_vld;
    reg  [15:0]                   int_status_db_idx = 'b0;
    reg                           int_status_db_pixels_in_ap_vld;
    reg  [31:0]                   int_status_db_pixels_in = 'b0;
    reg                           int_status_db_pixels_out_ap_vld;
    reg  [31:0]                   int_status_db_pixels_out = 'b0;

//------------------------Instantiation------------------


//------------------------AXI write fsm------------------
assign AWREADY = (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BVALID  = (wstate == WRRESP);
assign BRESP   = 2'b00;  // OKAY
assign wmask   = { {8{WSTRB[3]}}, {8{WSTRB[2]}}, {8{WSTRB[1]}}, {8{WSTRB[0]}} };
assign aw_hs   = AWVALID & AWREADY;
assign w_hs    = WVALID & WREADY;

// wstate
always @(posedge ACLK) begin
    if (ARESET)
        wstate <= WRRESET;
    else if (ACLK_EN)
        wstate <= wnext;
end

// wnext
always @(*) begin
    case (wstate)
        WRIDLE:
            if (AWVALID)
                wnext = WRDATA;
            else
                wnext = WRIDLE;
        WRDATA:
            if (WVALID)
                wnext = WRRESP;
            else
                wnext = WRDATA;
        WRRESP:
            if (BREADY & BVALID)
                wnext = WRIDLE;
            else
                wnext = WRRESP;
        default:
            wnext = WRIDLE;
    endcase
end

// waddr
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (aw_hs)
            waddr <= {AWADDR[ADDR_BITS-1:2], {2{1'b0}}};
    end
end

//------------------------AXI read fsm-------------------
assign ARREADY = (rstate == RDIDLE);
assign RDATA   = rdata;
assign RRESP   = 2'b00;  // OKAY
assign RVALID  = (rstate == RDDATA);
assign ar_hs   = ARVALID & ARREADY;
assign raddr   = ARADDR[ADDR_BITS-1:0];

// rstate
always @(posedge ACLK) begin
    if (ARESET)
        rstate <= RDRESET;
    else if (ACLK_EN)
        rstate <= rnext;
end

// rnext
always @(*) begin
    case (rstate)
        RDIDLE:
            if (ARVALID)
                rnext = RDDATA;
            else
                rnext = RDIDLE;
        RDDATA:
            if (RREADY & RVALID)
                rnext = RDIDLE;
            else
                rnext = RDDATA;
        default:
            rnext = RDIDLE;
    endcase
end

// rdata
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (ar_hs) begin
            rdata <= 'b0;
            case (raddr)
                ADDR_GOER_CFG_COS_OMEGA_0_DATA_0: begin
                    rdata <= int_goer_cfg_COS_OMEGA_0[17:0];
                end
                ADDR_GOER_CFG_COS_OMEGA_1_DATA_0: begin
                    rdata <= int_goer_cfg_COS_OMEGA_1[17:0];
                end
                ADDR_GOER_CFG_COS_OMEGA2_0_DATA_0: begin
                    rdata <= int_goer_cfg_COS_OMEGA2_0[17:0];
                end
                ADDR_GOER_CFG_COS_OMEGA2_1_DATA_0: begin
                    rdata <= int_goer_cfg_COS_OMEGA2_1[17:0];
                end
                ADDR_GOER_CFG_SIN_OMEGA_0_DATA_0: begin
                    rdata <= int_goer_cfg_SIN_OMEGA_0[17:0];
                end
                ADDR_GOER_CFG_SIN_OMEGA_1_DATA_0: begin
                    rdata <= int_goer_cfg_SIN_OMEGA_1[17:0];
                end
                ADDR_DEBL_CFG_DATA_0: begin
                    rdata <= int_debl_cfg[7:0];
                end
                ADDR_STATUS_GZ_SAMPLES_IN_DATA_0: begin
                    rdata <= int_status_gz_samples_in[31:0];
                end
                ADDR_STATUS_GZ_SAMPLES_IN_CTRL: begin
                    rdata[0] <= int_status_gz_samples_in_ap_vld;
                end
                ADDR_STATUS_GZ_SAMPLE_WIN_DATA_0: begin
                    rdata <= int_status_gz_sample_win[31:0];
                end
                ADDR_STATUS_GZ_SAMPLE_WIN_CTRL: begin
                    rdata[0] <= int_status_gz_sample_win_ap_vld;
                end
                ADDR_STATUS_GZ_SAMPLES_OUT_DATA_0: begin
                    rdata <= int_status_gz_samples_out[31:0];
                end
                ADDR_STATUS_GZ_SAMPLES_OUT_CTRL: begin
                    rdata[0] <= int_status_gz_samples_out_ap_vld;
                end
                ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_DATA_0: begin
                    rdata <= int_status_gz_samples_out_fifo[31:0];
                end
                ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_CTRL: begin
                    rdata[0] <= int_status_gz_samples_out_fifo_ap_vld;
                end
                ADDR_STATUS_CC_STATE_DATA_0: begin
                    rdata <= int_status_cc_state[7:0];
                end
                ADDR_STATUS_CC_STATE_CTRL: begin
                    rdata[0] <= int_status_cc_state_ap_vld;
                end
                ADDR_STATUS_CC_SAMPLES_IN_DATA_0: begin
                    rdata <= int_status_cc_samples_in[31:0];
                end
                ADDR_STATUS_CC_SAMPLES_IN_CTRL: begin
                    rdata[0] <= int_status_cc_samples_in_ap_vld;
                end
                ADDR_STATUS_CC_SAMPLES_OUT_DATA_0: begin
                    rdata <= int_status_cc_samples_out[31:0];
                end
                ADDR_STATUS_CC_SAMPLES_OUT_CTRL: begin
                    rdata[0] <= int_status_cc_samples_out_ap_vld;
                end
                ADDR_STATUS_CC_SAMPLE_IDX_DATA_0: begin
                    rdata <= int_status_cc_sample_idx[31:0];
                end
                ADDR_STATUS_CC_SAMPLE_IDX_CTRL: begin
                    rdata[0] <= int_status_cc_sample_idx_ap_vld;
                end
                ADDR_STATUS_CC_CURRENT_NORM_DATA_0: begin
                    rdata <= int_status_cc_current_norm[31:0];
                end
                ADDR_STATUS_CC_CURRENT_NORM_CTRL: begin
                    rdata[0] <= int_status_cc_current_norm_ap_vld;
                end
                ADDR_STATUS_CC_NORMS_WRITTEN_DATA_0: begin
                    rdata <= int_status_cc_norms_written[31:0];
                end
                ADDR_STATUS_CC_NORMS_WRITTEN_CTRL: begin
                    rdata[0] <= int_status_cc_norms_written_ap_vld;
                end
                ADDR_STATUS_CC_OUT_FIFO_DATA_0: begin
                    rdata <= int_status_cc_out_fifo[31:0];
                end
                ADDR_STATUS_CC_OUT_FIFO_CTRL: begin
                    rdata[0] <= int_status_cc_out_fifo_ap_vld;
                end
                ADDR_STATUS_CC_NORMS_FIFO_DATA_0: begin
                    rdata <= int_status_cc_norms_fifo[31:0];
                end
                ADDR_STATUS_CC_NORMS_FIFO_CTRL: begin
                    rdata[0] <= int_status_cc_norms_fifo_ap_vld;
                end
                ADDR_STATUS_BP_CONFIG_LOADED_DATA_0: begin
                    rdata <= int_status_bp_config_loaded[0:0];
                end
                ADDR_STATUS_BP_CONFIG_LOADED_CTRL: begin
                    rdata[0] <= int_status_bp_config_loaded_ap_vld;
                end
                ADDR_STATUS_BP_FSM_STATE_DATA_0: begin
                    rdata <= int_status_bp_fsm_state[7:0];
                end
                ADDR_STATUS_BP_FSM_STATE_CTRL: begin
                    rdata[0] <= int_status_bp_fsm_state_ap_vld;
                end
                ADDR_STATUS_BP_PARAM_STATE_DATA_0: begin
                    rdata <= int_status_bp_param_state[7:0];
                end
                ADDR_STATUS_BP_PARAM_STATE_CTRL: begin
                    rdata[0] <= int_status_bp_param_state_ap_vld;
                end
                ADDR_STATUS_BP_IDX_DATA_0: begin
                    rdata <= int_status_bp_idx[15:0];
                end
                ADDR_STATUS_BP_IDX_CTRL: begin
                    rdata[0] <= int_status_bp_idx_ap_vld;
                end
                ADDR_STATUS_BP_SIGMAS_IN_DATA_0: begin
                    rdata <= int_status_bp_sigmas_in[31:0];
                end
                ADDR_STATUS_BP_SIGMAS_IN_CTRL: begin
                    rdata[0] <= int_status_bp_sigmas_in_ap_vld;
                end
                ADDR_STATUS_BP_PIXELS_OUT_DATA_0: begin
                    rdata <= int_status_bp_pixels_out[31:0];
                end
                ADDR_STATUS_BP_PIXELS_OUT_CTRL: begin
                    rdata[0] <= int_status_bp_pixels_out_ap_vld;
                end
                ADDR_STATUS_BP_OUT_FIFO_LEVEL_DATA_0: begin
                    rdata <= int_status_bp_out_fifo_level[15:0];
                end
                ADDR_STATUS_BP_OUT_FIFO_LEVEL_CTRL: begin
                    rdata[0] <= int_status_bp_out_fifo_level_ap_vld;
                end
                ADDR_STATUS_DB_CONFIG_LOADED_DATA_0: begin
                    rdata <= int_status_db_config_loaded[0:0];
                end
                ADDR_STATUS_DB_CONFIG_LOADED_CTRL: begin
                    rdata[0] <= int_status_db_config_loaded_ap_vld;
                end
                ADDR_STATUS_DB_FSM_STATE_DATA_0: begin
                    rdata <= int_status_db_fsm_state[7:0];
                end
                ADDR_STATUS_DB_FSM_STATE_CTRL: begin
                    rdata[0] <= int_status_db_fsm_state_ap_vld;
                end
                ADDR_STATUS_DB_PARAM_STATE_DATA_0: begin
                    rdata <= int_status_db_param_state[7:0];
                end
                ADDR_STATUS_DB_PARAM_STATE_CTRL: begin
                    rdata[0] <= int_status_db_param_state_ap_vld;
                end
                ADDR_STATUS_DB_IDX_DATA_0: begin
                    rdata <= int_status_db_idx[15:0];
                end
                ADDR_STATUS_DB_IDX_CTRL: begin
                    rdata[0] <= int_status_db_idx_ap_vld;
                end
                ADDR_STATUS_DB_PIXELS_IN_DATA_0: begin
                    rdata <= int_status_db_pixels_in[31:0];
                end
                ADDR_STATUS_DB_PIXELS_IN_CTRL: begin
                    rdata[0] <= int_status_db_pixels_in_ap_vld;
                end
                ADDR_STATUS_DB_PIXELS_OUT_DATA_0: begin
                    rdata <= int_status_db_pixels_out[31:0];
                end
                ADDR_STATUS_DB_PIXELS_OUT_CTRL: begin
                    rdata[0] <= int_status_db_pixels_out_ap_vld;
                end
            endcase
        end
    end
end


//------------------------Register logic-----------------
assign goer_cfg_COS_OMEGA_0  = int_goer_cfg_COS_OMEGA_0;
assign goer_cfg_COS_OMEGA_1  = int_goer_cfg_COS_OMEGA_1;
assign goer_cfg_COS_OMEGA2_0 = int_goer_cfg_COS_OMEGA2_0;
assign goer_cfg_COS_OMEGA2_1 = int_goer_cfg_COS_OMEGA2_1;
assign goer_cfg_SIN_OMEGA_0  = int_goer_cfg_SIN_OMEGA_0;
assign goer_cfg_SIN_OMEGA_1  = int_goer_cfg_SIN_OMEGA_1;
assign debl_cfg              = int_debl_cfg;
// int_goer_cfg_COS_OMEGA_0[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_goer_cfg_COS_OMEGA_0[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GOER_CFG_COS_OMEGA_0_DATA_0)
            int_goer_cfg_COS_OMEGA_0[17:0] <= (WDATA[31:0] & wmask) | (int_goer_cfg_COS_OMEGA_0[17:0] & ~wmask);
    end
end

// int_goer_cfg_COS_OMEGA_1[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_goer_cfg_COS_OMEGA_1[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GOER_CFG_COS_OMEGA_1_DATA_0)
            int_goer_cfg_COS_OMEGA_1[17:0] <= (WDATA[31:0] & wmask) | (int_goer_cfg_COS_OMEGA_1[17:0] & ~wmask);
    end
end

// int_goer_cfg_COS_OMEGA2_0[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_goer_cfg_COS_OMEGA2_0[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GOER_CFG_COS_OMEGA2_0_DATA_0)
            int_goer_cfg_COS_OMEGA2_0[17:0] <= (WDATA[31:0] & wmask) | (int_goer_cfg_COS_OMEGA2_0[17:0] & ~wmask);
    end
end

// int_goer_cfg_COS_OMEGA2_1[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_goer_cfg_COS_OMEGA2_1[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GOER_CFG_COS_OMEGA2_1_DATA_0)
            int_goer_cfg_COS_OMEGA2_1[17:0] <= (WDATA[31:0] & wmask) | (int_goer_cfg_COS_OMEGA2_1[17:0] & ~wmask);
    end
end

// int_goer_cfg_SIN_OMEGA_0[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_goer_cfg_SIN_OMEGA_0[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GOER_CFG_SIN_OMEGA_0_DATA_0)
            int_goer_cfg_SIN_OMEGA_0[17:0] <= (WDATA[31:0] & wmask) | (int_goer_cfg_SIN_OMEGA_0[17:0] & ~wmask);
    end
end

// int_goer_cfg_SIN_OMEGA_1[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_goer_cfg_SIN_OMEGA_1[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GOER_CFG_SIN_OMEGA_1_DATA_0)
            int_goer_cfg_SIN_OMEGA_1[17:0] <= (WDATA[31:0] & wmask) | (int_goer_cfg_SIN_OMEGA_1[17:0] & ~wmask);
    end
end

// int_debl_cfg[7:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg[7:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_DATA_0)
            int_debl_cfg[7:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg[7:0] & ~wmask);
    end
end

// int_status_gz_samples_in
always @(posedge ACLK) begin
    if (ARESET)
        int_status_gz_samples_in <= 0;
    else if (ACLK_EN) begin
        if (status_gz_samples_in_ap_vld)
            int_status_gz_samples_in <= status_gz_samples_in;
    end
end

// int_status_gz_samples_in_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_gz_samples_in_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_gz_samples_in_ap_vld)
            int_status_gz_samples_in_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_GZ_SAMPLES_IN_CTRL)
            int_status_gz_samples_in_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_gz_sample_win
always @(posedge ACLK) begin
    if (ARESET)
        int_status_gz_sample_win <= 0;
    else if (ACLK_EN) begin
        if (status_gz_sample_win_ap_vld)
            int_status_gz_sample_win <= status_gz_sample_win;
    end
end

// int_status_gz_sample_win_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_gz_sample_win_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_gz_sample_win_ap_vld)
            int_status_gz_sample_win_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_GZ_SAMPLE_WIN_CTRL)
            int_status_gz_sample_win_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_gz_samples_out
always @(posedge ACLK) begin
    if (ARESET)
        int_status_gz_samples_out <= 0;
    else if (ACLK_EN) begin
        if (status_gz_samples_out_ap_vld)
            int_status_gz_samples_out <= status_gz_samples_out;
    end
end

// int_status_gz_samples_out_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_gz_samples_out_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_gz_samples_out_ap_vld)
            int_status_gz_samples_out_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_GZ_SAMPLES_OUT_CTRL)
            int_status_gz_samples_out_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_gz_samples_out_fifo
always @(posedge ACLK) begin
    if (ARESET)
        int_status_gz_samples_out_fifo <= 0;
    else if (ACLK_EN) begin
        if (status_gz_samples_out_fifo_ap_vld)
            int_status_gz_samples_out_fifo <= status_gz_samples_out_fifo;
    end
end

// int_status_gz_samples_out_fifo_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_gz_samples_out_fifo_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_gz_samples_out_fifo_ap_vld)
            int_status_gz_samples_out_fifo_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_CTRL)
            int_status_gz_samples_out_fifo_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_cc_state
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_state <= 0;
    else if (ACLK_EN) begin
        if (status_cc_state_ap_vld)
            int_status_cc_state <= status_cc_state;
    end
end

// int_status_cc_state_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_state_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_cc_state_ap_vld)
            int_status_cc_state_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_CC_STATE_CTRL)
            int_status_cc_state_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_cc_samples_in
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_samples_in <= 0;
    else if (ACLK_EN) begin
        if (status_cc_samples_in_ap_vld)
            int_status_cc_samples_in <= status_cc_samples_in;
    end
end

// int_status_cc_samples_in_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_samples_in_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_cc_samples_in_ap_vld)
            int_status_cc_samples_in_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_CC_SAMPLES_IN_CTRL)
            int_status_cc_samples_in_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_cc_samples_out
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_samples_out <= 0;
    else if (ACLK_EN) begin
        if (status_cc_samples_out_ap_vld)
            int_status_cc_samples_out <= status_cc_samples_out;
    end
end

// int_status_cc_samples_out_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_samples_out_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_cc_samples_out_ap_vld)
            int_status_cc_samples_out_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_CC_SAMPLES_OUT_CTRL)
            int_status_cc_samples_out_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_cc_sample_idx
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_sample_idx <= 0;
    else if (ACLK_EN) begin
        if (status_cc_sample_idx_ap_vld)
            int_status_cc_sample_idx <= status_cc_sample_idx;
    end
end

// int_status_cc_sample_idx_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_sample_idx_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_cc_sample_idx_ap_vld)
            int_status_cc_sample_idx_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_CC_SAMPLE_IDX_CTRL)
            int_status_cc_sample_idx_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_cc_current_norm
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_current_norm <= 0;
    else if (ACLK_EN) begin
        if (status_cc_current_norm_ap_vld)
            int_status_cc_current_norm <= status_cc_current_norm;
    end
end

// int_status_cc_current_norm_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_current_norm_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_cc_current_norm_ap_vld)
            int_status_cc_current_norm_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_CC_CURRENT_NORM_CTRL)
            int_status_cc_current_norm_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_cc_norms_written
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_norms_written <= 0;
    else if (ACLK_EN) begin
        if (status_cc_norms_written_ap_vld)
            int_status_cc_norms_written <= status_cc_norms_written;
    end
end

// int_status_cc_norms_written_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_norms_written_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_cc_norms_written_ap_vld)
            int_status_cc_norms_written_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_CC_NORMS_WRITTEN_CTRL)
            int_status_cc_norms_written_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_cc_out_fifo
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_out_fifo <= 0;
    else if (ACLK_EN) begin
        if (status_cc_out_fifo_ap_vld)
            int_status_cc_out_fifo <= status_cc_out_fifo;
    end
end

// int_status_cc_out_fifo_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_out_fifo_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_cc_out_fifo_ap_vld)
            int_status_cc_out_fifo_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_CC_OUT_FIFO_CTRL)
            int_status_cc_out_fifo_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_cc_norms_fifo
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_norms_fifo <= 0;
    else if (ACLK_EN) begin
        if (status_cc_norms_fifo_ap_vld)
            int_status_cc_norms_fifo <= status_cc_norms_fifo;
    end
end

// int_status_cc_norms_fifo_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_cc_norms_fifo_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_cc_norms_fifo_ap_vld)
            int_status_cc_norms_fifo_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_CC_NORMS_FIFO_CTRL)
            int_status_cc_norms_fifo_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_bp_config_loaded
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_config_loaded <= 0;
    else if (ACLK_EN) begin
        if (status_bp_config_loaded_ap_vld)
            int_status_bp_config_loaded <= status_bp_config_loaded;
    end
end

// int_status_bp_config_loaded_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_config_loaded_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_bp_config_loaded_ap_vld)
            int_status_bp_config_loaded_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_BP_CONFIG_LOADED_CTRL)
            int_status_bp_config_loaded_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_bp_fsm_state
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_fsm_state <= 0;
    else if (ACLK_EN) begin
        if (status_bp_fsm_state_ap_vld)
            int_status_bp_fsm_state <= status_bp_fsm_state;
    end
end

// int_status_bp_fsm_state_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_fsm_state_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_bp_fsm_state_ap_vld)
            int_status_bp_fsm_state_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_BP_FSM_STATE_CTRL)
            int_status_bp_fsm_state_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_bp_param_state
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_param_state <= 0;
    else if (ACLK_EN) begin
        if (status_bp_param_state_ap_vld)
            int_status_bp_param_state <= status_bp_param_state;
    end
end

// int_status_bp_param_state_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_param_state_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_bp_param_state_ap_vld)
            int_status_bp_param_state_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_BP_PARAM_STATE_CTRL)
            int_status_bp_param_state_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_bp_idx
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_idx <= 0;
    else if (ACLK_EN) begin
        if (status_bp_idx_ap_vld)
            int_status_bp_idx <= status_bp_idx;
    end
end

// int_status_bp_idx_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_idx_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_bp_idx_ap_vld)
            int_status_bp_idx_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_BP_IDX_CTRL)
            int_status_bp_idx_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_bp_sigmas_in
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_sigmas_in <= 0;
    else if (ACLK_EN) begin
        if (status_bp_sigmas_in_ap_vld)
            int_status_bp_sigmas_in <= status_bp_sigmas_in;
    end
end

// int_status_bp_sigmas_in_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_sigmas_in_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_bp_sigmas_in_ap_vld)
            int_status_bp_sigmas_in_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_BP_SIGMAS_IN_CTRL)
            int_status_bp_sigmas_in_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_bp_pixels_out
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_pixels_out <= 0;
    else if (ACLK_EN) begin
        if (status_bp_pixels_out_ap_vld)
            int_status_bp_pixels_out <= status_bp_pixels_out;
    end
end

// int_status_bp_pixels_out_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_pixels_out_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_bp_pixels_out_ap_vld)
            int_status_bp_pixels_out_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_BP_PIXELS_OUT_CTRL)
            int_status_bp_pixels_out_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_bp_out_fifo_level
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_out_fifo_level <= 0;
    else if (ACLK_EN) begin
        if (status_bp_out_fifo_level_ap_vld)
            int_status_bp_out_fifo_level <= status_bp_out_fifo_level;
    end
end

// int_status_bp_out_fifo_level_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_bp_out_fifo_level_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_bp_out_fifo_level_ap_vld)
            int_status_bp_out_fifo_level_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_BP_OUT_FIFO_LEVEL_CTRL)
            int_status_bp_out_fifo_level_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_db_config_loaded
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_config_loaded <= 0;
    else if (ACLK_EN) begin
        if (status_db_config_loaded_ap_vld)
            int_status_db_config_loaded <= status_db_config_loaded;
    end
end

// int_status_db_config_loaded_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_config_loaded_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_db_config_loaded_ap_vld)
            int_status_db_config_loaded_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_DB_CONFIG_LOADED_CTRL)
            int_status_db_config_loaded_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_db_fsm_state
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_fsm_state <= 0;
    else if (ACLK_EN) begin
        if (status_db_fsm_state_ap_vld)
            int_status_db_fsm_state <= status_db_fsm_state;
    end
end

// int_status_db_fsm_state_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_fsm_state_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_db_fsm_state_ap_vld)
            int_status_db_fsm_state_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_DB_FSM_STATE_CTRL)
            int_status_db_fsm_state_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_db_param_state
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_param_state <= 0;
    else if (ACLK_EN) begin
        if (status_db_param_state_ap_vld)
            int_status_db_param_state <= status_db_param_state;
    end
end

// int_status_db_param_state_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_param_state_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_db_param_state_ap_vld)
            int_status_db_param_state_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_DB_PARAM_STATE_CTRL)
            int_status_db_param_state_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_db_idx
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_idx <= 0;
    else if (ACLK_EN) begin
        if (status_db_idx_ap_vld)
            int_status_db_idx <= status_db_idx;
    end
end

// int_status_db_idx_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_idx_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_db_idx_ap_vld)
            int_status_db_idx_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_DB_IDX_CTRL)
            int_status_db_idx_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_db_pixels_in
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_pixels_in <= 0;
    else if (ACLK_EN) begin
        if (status_db_pixels_in_ap_vld)
            int_status_db_pixels_in <= status_db_pixels_in;
    end
end

// int_status_db_pixels_in_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_pixels_in_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_db_pixels_in_ap_vld)
            int_status_db_pixels_in_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_DB_PIXELS_IN_CTRL)
            int_status_db_pixels_in_ap_vld <= 1'b0; // clear on read
    end
end

// int_status_db_pixels_out
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_pixels_out <= 0;
    else if (ACLK_EN) begin
        if (status_db_pixels_out_ap_vld)
            int_status_db_pixels_out <= status_db_pixels_out;
    end
end

// int_status_db_pixels_out_ap_vld
always @(posedge ACLK) begin
    if (ARESET)
        int_status_db_pixels_out_ap_vld <= 1'b0;
    else if (ACLK_EN) begin
        if (status_db_pixels_out_ap_vld)
            int_status_db_pixels_out_ap_vld <= 1'b1;
        else if (ar_hs && raddr == ADDR_STATUS_DB_PIXELS_OUT_CTRL)
            int_status_db_pixels_out_ap_vld <= 1'b0; // clear on read
    end
end


//------------------------Memory logic-------------------

endmodule
