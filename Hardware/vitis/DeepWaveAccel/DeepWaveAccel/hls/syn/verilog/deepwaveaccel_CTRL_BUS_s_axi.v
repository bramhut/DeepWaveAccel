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
    output wire [7:0]                    debl_cfg_n_layers,
    output wire [5:0]                    debl_cfg_K,
    output wire [11:0]                   debl_cfg_lap_off_0,
    output wire [11:0]                   debl_cfg_lap_off_1,
    output wire [11:0]                   debl_cfg_lap_off_2,
    output wire [11:0]                   debl_cfg_lap_off_3,
    output wire [11:0]                   debl_cfg_lap_off_4,
    output wire [11:0]                   debl_cfg_lap_off_5,
    output wire [17:0]                   debl_cfg_theta_0,
    output wire [17:0]                   debl_cfg_theta_1,
    output wire [17:0]                   debl_cfg_theta_2,
    output wire [17:0]                   debl_cfg_theta_3,
    output wire [17:0]                   debl_cfg_theta_4,
    output wire [17:0]                   debl_cfg_theta_5,
    output wire [17:0]                   debl_cfg_theta_6,
    output wire [17:0]                   debl_cfg_theta_7,
    output wire [17:0]                   debl_cfg_theta_8,
    output wire [17:0]                   debl_cfg_theta_9,
    output wire [17:0]                   debl_cfg_theta_10,
    output wire [17:0]                   debl_cfg_theta_11,
    output wire [17:0]                   debl_cfg_theta_12,
    output wire [17:0]                   debl_cfg_theta_13,
    output wire [17:0]                   debl_cfg_theta_14,
    output wire [17:0]                   debl_cfg_theta_15,
    output wire [17:0]                   debl_cfg_theta_16,
    output wire [17:0]                   debl_cfg_theta_17,
    output wire [17:0]                   debl_cfg_theta_18,
    output wire [17:0]                   debl_cfg_theta_19,
    output wire [17:0]                   debl_cfg_theta_20,
    output wire [17:0]                   debl_cfg_theta_21,
    output wire [17:0]                   debl_cfg_theta_22
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
// 0x040 : Data signal of debl_cfg_n_layers
//         bit 7~0 - debl_cfg_n_layers[7:0] (Read/Write)
//         others  - reserved
// 0x044 : reserved
// 0x048 : Data signal of debl_cfg_K
//         bit 5~0 - debl_cfg_K[5:0] (Read/Write)
//         others  - reserved
// 0x04c : reserved
// 0x050 : Data signal of debl_cfg_lap_off_0
//         bit 11~0 - debl_cfg_lap_off_0[11:0] (Read/Write)
//         others   - reserved
// 0x054 : reserved
// 0x058 : Data signal of debl_cfg_lap_off_1
//         bit 11~0 - debl_cfg_lap_off_1[11:0] (Read/Write)
//         others   - reserved
// 0x05c : reserved
// 0x060 : Data signal of debl_cfg_lap_off_2
//         bit 11~0 - debl_cfg_lap_off_2[11:0] (Read/Write)
//         others   - reserved
// 0x064 : reserved
// 0x068 : Data signal of debl_cfg_lap_off_3
//         bit 11~0 - debl_cfg_lap_off_3[11:0] (Read/Write)
//         others   - reserved
// 0x06c : reserved
// 0x070 : Data signal of debl_cfg_lap_off_4
//         bit 11~0 - debl_cfg_lap_off_4[11:0] (Read/Write)
//         others   - reserved
// 0x074 : reserved
// 0x078 : Data signal of debl_cfg_lap_off_5
//         bit 11~0 - debl_cfg_lap_off_5[11:0] (Read/Write)
//         others   - reserved
// 0x07c : reserved
// 0x080 : Data signal of debl_cfg_theta_0
//         bit 17~0 - debl_cfg_theta_0[17:0] (Read/Write)
//         others   - reserved
// 0x084 : reserved
// 0x088 : Data signal of debl_cfg_theta_1
//         bit 17~0 - debl_cfg_theta_1[17:0] (Read/Write)
//         others   - reserved
// 0x08c : reserved
// 0x090 : Data signal of debl_cfg_theta_2
//         bit 17~0 - debl_cfg_theta_2[17:0] (Read/Write)
//         others   - reserved
// 0x094 : reserved
// 0x098 : Data signal of debl_cfg_theta_3
//         bit 17~0 - debl_cfg_theta_3[17:0] (Read/Write)
//         others   - reserved
// 0x09c : reserved
// 0x0a0 : Data signal of debl_cfg_theta_4
//         bit 17~0 - debl_cfg_theta_4[17:0] (Read/Write)
//         others   - reserved
// 0x0a4 : reserved
// 0x0a8 : Data signal of debl_cfg_theta_5
//         bit 17~0 - debl_cfg_theta_5[17:0] (Read/Write)
//         others   - reserved
// 0x0ac : reserved
// 0x0b0 : Data signal of debl_cfg_theta_6
//         bit 17~0 - debl_cfg_theta_6[17:0] (Read/Write)
//         others   - reserved
// 0x0b4 : reserved
// 0x0b8 : Data signal of debl_cfg_theta_7
//         bit 17~0 - debl_cfg_theta_7[17:0] (Read/Write)
//         others   - reserved
// 0x0bc : reserved
// 0x0c0 : Data signal of debl_cfg_theta_8
//         bit 17~0 - debl_cfg_theta_8[17:0] (Read/Write)
//         others   - reserved
// 0x0c4 : reserved
// 0x0c8 : Data signal of debl_cfg_theta_9
//         bit 17~0 - debl_cfg_theta_9[17:0] (Read/Write)
//         others   - reserved
// 0x0cc : reserved
// 0x0d0 : Data signal of debl_cfg_theta_10
//         bit 17~0 - debl_cfg_theta_10[17:0] (Read/Write)
//         others   - reserved
// 0x0d4 : reserved
// 0x0d8 : Data signal of debl_cfg_theta_11
//         bit 17~0 - debl_cfg_theta_11[17:0] (Read/Write)
//         others   - reserved
// 0x0dc : reserved
// 0x0e0 : Data signal of debl_cfg_theta_12
//         bit 17~0 - debl_cfg_theta_12[17:0] (Read/Write)
//         others   - reserved
// 0x0e4 : reserved
// 0x0e8 : Data signal of debl_cfg_theta_13
//         bit 17~0 - debl_cfg_theta_13[17:0] (Read/Write)
//         others   - reserved
// 0x0ec : reserved
// 0x0f0 : Data signal of debl_cfg_theta_14
//         bit 17~0 - debl_cfg_theta_14[17:0] (Read/Write)
//         others   - reserved
// 0x0f4 : reserved
// 0x0f8 : Data signal of debl_cfg_theta_15
//         bit 17~0 - debl_cfg_theta_15[17:0] (Read/Write)
//         others   - reserved
// 0x0fc : reserved
// 0x100 : Data signal of debl_cfg_theta_16
//         bit 17~0 - debl_cfg_theta_16[17:0] (Read/Write)
//         others   - reserved
// 0x104 : reserved
// 0x108 : Data signal of debl_cfg_theta_17
//         bit 17~0 - debl_cfg_theta_17[17:0] (Read/Write)
//         others   - reserved
// 0x10c : reserved
// 0x110 : Data signal of debl_cfg_theta_18
//         bit 17~0 - debl_cfg_theta_18[17:0] (Read/Write)
//         others   - reserved
// 0x114 : reserved
// 0x118 : Data signal of debl_cfg_theta_19
//         bit 17~0 - debl_cfg_theta_19[17:0] (Read/Write)
//         others   - reserved
// 0x11c : reserved
// 0x120 : Data signal of debl_cfg_theta_20
//         bit 17~0 - debl_cfg_theta_20[17:0] (Read/Write)
//         others   - reserved
// 0x124 : reserved
// 0x128 : Data signal of debl_cfg_theta_21
//         bit 17~0 - debl_cfg_theta_21[17:0] (Read/Write)
//         others   - reserved
// 0x12c : reserved
// 0x130 : Data signal of debl_cfg_theta_22
//         bit 17~0 - debl_cfg_theta_22[17:0] (Read/Write)
//         others   - reserved
// 0x134 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

//------------------------Parameter----------------------
localparam
    ADDR_GOER_CFG_COS_OMEGA_0_DATA_0  = 9'h010,
    ADDR_GOER_CFG_COS_OMEGA_0_CTRL    = 9'h014,
    ADDR_GOER_CFG_COS_OMEGA_1_DATA_0  = 9'h018,
    ADDR_GOER_CFG_COS_OMEGA_1_CTRL    = 9'h01c,
    ADDR_GOER_CFG_COS_OMEGA2_0_DATA_0 = 9'h020,
    ADDR_GOER_CFG_COS_OMEGA2_0_CTRL   = 9'h024,
    ADDR_GOER_CFG_COS_OMEGA2_1_DATA_0 = 9'h028,
    ADDR_GOER_CFG_COS_OMEGA2_1_CTRL   = 9'h02c,
    ADDR_GOER_CFG_SIN_OMEGA_0_DATA_0  = 9'h030,
    ADDR_GOER_CFG_SIN_OMEGA_0_CTRL    = 9'h034,
    ADDR_GOER_CFG_SIN_OMEGA_1_DATA_0  = 9'h038,
    ADDR_GOER_CFG_SIN_OMEGA_1_CTRL    = 9'h03c,
    ADDR_DEBL_CFG_N_LAYERS_DATA_0     = 9'h040,
    ADDR_DEBL_CFG_N_LAYERS_CTRL       = 9'h044,
    ADDR_DEBL_CFG_K_DATA_0            = 9'h048,
    ADDR_DEBL_CFG_K_CTRL              = 9'h04c,
    ADDR_DEBL_CFG_LAP_OFF_0_DATA_0    = 9'h050,
    ADDR_DEBL_CFG_LAP_OFF_0_CTRL      = 9'h054,
    ADDR_DEBL_CFG_LAP_OFF_1_DATA_0    = 9'h058,
    ADDR_DEBL_CFG_LAP_OFF_1_CTRL      = 9'h05c,
    ADDR_DEBL_CFG_LAP_OFF_2_DATA_0    = 9'h060,
    ADDR_DEBL_CFG_LAP_OFF_2_CTRL      = 9'h064,
    ADDR_DEBL_CFG_LAP_OFF_3_DATA_0    = 9'h068,
    ADDR_DEBL_CFG_LAP_OFF_3_CTRL      = 9'h06c,
    ADDR_DEBL_CFG_LAP_OFF_4_DATA_0    = 9'h070,
    ADDR_DEBL_CFG_LAP_OFF_4_CTRL      = 9'h074,
    ADDR_DEBL_CFG_LAP_OFF_5_DATA_0    = 9'h078,
    ADDR_DEBL_CFG_LAP_OFF_5_CTRL      = 9'h07c,
    ADDR_DEBL_CFG_THETA_0_DATA_0      = 9'h080,
    ADDR_DEBL_CFG_THETA_0_CTRL        = 9'h084,
    ADDR_DEBL_CFG_THETA_1_DATA_0      = 9'h088,
    ADDR_DEBL_CFG_THETA_1_CTRL        = 9'h08c,
    ADDR_DEBL_CFG_THETA_2_DATA_0      = 9'h090,
    ADDR_DEBL_CFG_THETA_2_CTRL        = 9'h094,
    ADDR_DEBL_CFG_THETA_3_DATA_0      = 9'h098,
    ADDR_DEBL_CFG_THETA_3_CTRL        = 9'h09c,
    ADDR_DEBL_CFG_THETA_4_DATA_0      = 9'h0a0,
    ADDR_DEBL_CFG_THETA_4_CTRL        = 9'h0a4,
    ADDR_DEBL_CFG_THETA_5_DATA_0      = 9'h0a8,
    ADDR_DEBL_CFG_THETA_5_CTRL        = 9'h0ac,
    ADDR_DEBL_CFG_THETA_6_DATA_0      = 9'h0b0,
    ADDR_DEBL_CFG_THETA_6_CTRL        = 9'h0b4,
    ADDR_DEBL_CFG_THETA_7_DATA_0      = 9'h0b8,
    ADDR_DEBL_CFG_THETA_7_CTRL        = 9'h0bc,
    ADDR_DEBL_CFG_THETA_8_DATA_0      = 9'h0c0,
    ADDR_DEBL_CFG_THETA_8_CTRL        = 9'h0c4,
    ADDR_DEBL_CFG_THETA_9_DATA_0      = 9'h0c8,
    ADDR_DEBL_CFG_THETA_9_CTRL        = 9'h0cc,
    ADDR_DEBL_CFG_THETA_10_DATA_0     = 9'h0d0,
    ADDR_DEBL_CFG_THETA_10_CTRL       = 9'h0d4,
    ADDR_DEBL_CFG_THETA_11_DATA_0     = 9'h0d8,
    ADDR_DEBL_CFG_THETA_11_CTRL       = 9'h0dc,
    ADDR_DEBL_CFG_THETA_12_DATA_0     = 9'h0e0,
    ADDR_DEBL_CFG_THETA_12_CTRL       = 9'h0e4,
    ADDR_DEBL_CFG_THETA_13_DATA_0     = 9'h0e8,
    ADDR_DEBL_CFG_THETA_13_CTRL       = 9'h0ec,
    ADDR_DEBL_CFG_THETA_14_DATA_0     = 9'h0f0,
    ADDR_DEBL_CFG_THETA_14_CTRL       = 9'h0f4,
    ADDR_DEBL_CFG_THETA_15_DATA_0     = 9'h0f8,
    ADDR_DEBL_CFG_THETA_15_CTRL       = 9'h0fc,
    ADDR_DEBL_CFG_THETA_16_DATA_0     = 9'h100,
    ADDR_DEBL_CFG_THETA_16_CTRL       = 9'h104,
    ADDR_DEBL_CFG_THETA_17_DATA_0     = 9'h108,
    ADDR_DEBL_CFG_THETA_17_CTRL       = 9'h10c,
    ADDR_DEBL_CFG_THETA_18_DATA_0     = 9'h110,
    ADDR_DEBL_CFG_THETA_18_CTRL       = 9'h114,
    ADDR_DEBL_CFG_THETA_19_DATA_0     = 9'h118,
    ADDR_DEBL_CFG_THETA_19_CTRL       = 9'h11c,
    ADDR_DEBL_CFG_THETA_20_DATA_0     = 9'h120,
    ADDR_DEBL_CFG_THETA_20_CTRL       = 9'h124,
    ADDR_DEBL_CFG_THETA_21_DATA_0     = 9'h128,
    ADDR_DEBL_CFG_THETA_21_CTRL       = 9'h12c,
    ADDR_DEBL_CFG_THETA_22_DATA_0     = 9'h130,
    ADDR_DEBL_CFG_THETA_22_CTRL       = 9'h134,
    WRIDLE                            = 2'd0,
    WRDATA                            = 2'd1,
    WRRESP                            = 2'd2,
    WRRESET                           = 2'd3,
    RDIDLE                            = 2'd0,
    RDDATA                            = 2'd1,
    RDRESET                           = 2'd2,
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
    reg  [7:0]                    int_debl_cfg_n_layers = 'b0;
    reg  [5:0]                    int_debl_cfg_K = 'b0;
    reg  [11:0]                   int_debl_cfg_lap_off_0 = 'b0;
    reg  [11:0]                   int_debl_cfg_lap_off_1 = 'b0;
    reg  [11:0]                   int_debl_cfg_lap_off_2 = 'b0;
    reg  [11:0]                   int_debl_cfg_lap_off_3 = 'b0;
    reg  [11:0]                   int_debl_cfg_lap_off_4 = 'b0;
    reg  [11:0]                   int_debl_cfg_lap_off_5 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_0 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_1 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_2 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_3 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_4 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_5 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_6 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_7 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_8 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_9 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_10 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_11 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_12 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_13 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_14 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_15 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_16 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_17 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_18 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_19 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_20 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_21 = 'b0;
    reg  [17:0]                   int_debl_cfg_theta_22 = 'b0;

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
                ADDR_DEBL_CFG_N_LAYERS_DATA_0: begin
                    rdata <= int_debl_cfg_n_layers[7:0];
                end
                ADDR_DEBL_CFG_K_DATA_0: begin
                    rdata <= int_debl_cfg_K[5:0];
                end
                ADDR_DEBL_CFG_LAP_OFF_0_DATA_0: begin
                    rdata <= int_debl_cfg_lap_off_0[11:0];
                end
                ADDR_DEBL_CFG_LAP_OFF_1_DATA_0: begin
                    rdata <= int_debl_cfg_lap_off_1[11:0];
                end
                ADDR_DEBL_CFG_LAP_OFF_2_DATA_0: begin
                    rdata <= int_debl_cfg_lap_off_2[11:0];
                end
                ADDR_DEBL_CFG_LAP_OFF_3_DATA_0: begin
                    rdata <= int_debl_cfg_lap_off_3[11:0];
                end
                ADDR_DEBL_CFG_LAP_OFF_4_DATA_0: begin
                    rdata <= int_debl_cfg_lap_off_4[11:0];
                end
                ADDR_DEBL_CFG_LAP_OFF_5_DATA_0: begin
                    rdata <= int_debl_cfg_lap_off_5[11:0];
                end
                ADDR_DEBL_CFG_THETA_0_DATA_0: begin
                    rdata <= int_debl_cfg_theta_0[17:0];
                end
                ADDR_DEBL_CFG_THETA_1_DATA_0: begin
                    rdata <= int_debl_cfg_theta_1[17:0];
                end
                ADDR_DEBL_CFG_THETA_2_DATA_0: begin
                    rdata <= int_debl_cfg_theta_2[17:0];
                end
                ADDR_DEBL_CFG_THETA_3_DATA_0: begin
                    rdata <= int_debl_cfg_theta_3[17:0];
                end
                ADDR_DEBL_CFG_THETA_4_DATA_0: begin
                    rdata <= int_debl_cfg_theta_4[17:0];
                end
                ADDR_DEBL_CFG_THETA_5_DATA_0: begin
                    rdata <= int_debl_cfg_theta_5[17:0];
                end
                ADDR_DEBL_CFG_THETA_6_DATA_0: begin
                    rdata <= int_debl_cfg_theta_6[17:0];
                end
                ADDR_DEBL_CFG_THETA_7_DATA_0: begin
                    rdata <= int_debl_cfg_theta_7[17:0];
                end
                ADDR_DEBL_CFG_THETA_8_DATA_0: begin
                    rdata <= int_debl_cfg_theta_8[17:0];
                end
                ADDR_DEBL_CFG_THETA_9_DATA_0: begin
                    rdata <= int_debl_cfg_theta_9[17:0];
                end
                ADDR_DEBL_CFG_THETA_10_DATA_0: begin
                    rdata <= int_debl_cfg_theta_10[17:0];
                end
                ADDR_DEBL_CFG_THETA_11_DATA_0: begin
                    rdata <= int_debl_cfg_theta_11[17:0];
                end
                ADDR_DEBL_CFG_THETA_12_DATA_0: begin
                    rdata <= int_debl_cfg_theta_12[17:0];
                end
                ADDR_DEBL_CFG_THETA_13_DATA_0: begin
                    rdata <= int_debl_cfg_theta_13[17:0];
                end
                ADDR_DEBL_CFG_THETA_14_DATA_0: begin
                    rdata <= int_debl_cfg_theta_14[17:0];
                end
                ADDR_DEBL_CFG_THETA_15_DATA_0: begin
                    rdata <= int_debl_cfg_theta_15[17:0];
                end
                ADDR_DEBL_CFG_THETA_16_DATA_0: begin
                    rdata <= int_debl_cfg_theta_16[17:0];
                end
                ADDR_DEBL_CFG_THETA_17_DATA_0: begin
                    rdata <= int_debl_cfg_theta_17[17:0];
                end
                ADDR_DEBL_CFG_THETA_18_DATA_0: begin
                    rdata <= int_debl_cfg_theta_18[17:0];
                end
                ADDR_DEBL_CFG_THETA_19_DATA_0: begin
                    rdata <= int_debl_cfg_theta_19[17:0];
                end
                ADDR_DEBL_CFG_THETA_20_DATA_0: begin
                    rdata <= int_debl_cfg_theta_20[17:0];
                end
                ADDR_DEBL_CFG_THETA_21_DATA_0: begin
                    rdata <= int_debl_cfg_theta_21[17:0];
                end
                ADDR_DEBL_CFG_THETA_22_DATA_0: begin
                    rdata <= int_debl_cfg_theta_22[17:0];
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
assign debl_cfg_n_layers     = int_debl_cfg_n_layers;
assign debl_cfg_K            = int_debl_cfg_K;
assign debl_cfg_lap_off_0    = int_debl_cfg_lap_off_0;
assign debl_cfg_lap_off_1    = int_debl_cfg_lap_off_1;
assign debl_cfg_lap_off_2    = int_debl_cfg_lap_off_2;
assign debl_cfg_lap_off_3    = int_debl_cfg_lap_off_3;
assign debl_cfg_lap_off_4    = int_debl_cfg_lap_off_4;
assign debl_cfg_lap_off_5    = int_debl_cfg_lap_off_5;
assign debl_cfg_theta_0      = int_debl_cfg_theta_0;
assign debl_cfg_theta_1      = int_debl_cfg_theta_1;
assign debl_cfg_theta_2      = int_debl_cfg_theta_2;
assign debl_cfg_theta_3      = int_debl_cfg_theta_3;
assign debl_cfg_theta_4      = int_debl_cfg_theta_4;
assign debl_cfg_theta_5      = int_debl_cfg_theta_5;
assign debl_cfg_theta_6      = int_debl_cfg_theta_6;
assign debl_cfg_theta_7      = int_debl_cfg_theta_7;
assign debl_cfg_theta_8      = int_debl_cfg_theta_8;
assign debl_cfg_theta_9      = int_debl_cfg_theta_9;
assign debl_cfg_theta_10     = int_debl_cfg_theta_10;
assign debl_cfg_theta_11     = int_debl_cfg_theta_11;
assign debl_cfg_theta_12     = int_debl_cfg_theta_12;
assign debl_cfg_theta_13     = int_debl_cfg_theta_13;
assign debl_cfg_theta_14     = int_debl_cfg_theta_14;
assign debl_cfg_theta_15     = int_debl_cfg_theta_15;
assign debl_cfg_theta_16     = int_debl_cfg_theta_16;
assign debl_cfg_theta_17     = int_debl_cfg_theta_17;
assign debl_cfg_theta_18     = int_debl_cfg_theta_18;
assign debl_cfg_theta_19     = int_debl_cfg_theta_19;
assign debl_cfg_theta_20     = int_debl_cfg_theta_20;
assign debl_cfg_theta_21     = int_debl_cfg_theta_21;
assign debl_cfg_theta_22     = int_debl_cfg_theta_22;
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

// int_debl_cfg_n_layers[7:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_n_layers[7:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_N_LAYERS_DATA_0)
            int_debl_cfg_n_layers[7:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_n_layers[7:0] & ~wmask);
    end
end

// int_debl_cfg_K[5:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_K[5:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_K_DATA_0)
            int_debl_cfg_K[5:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_K[5:0] & ~wmask);
    end
end

// int_debl_cfg_lap_off_0[11:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_lap_off_0[11:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_LAP_OFF_0_DATA_0)
            int_debl_cfg_lap_off_0[11:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_lap_off_0[11:0] & ~wmask);
    end
end

// int_debl_cfg_lap_off_1[11:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_lap_off_1[11:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_LAP_OFF_1_DATA_0)
            int_debl_cfg_lap_off_1[11:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_lap_off_1[11:0] & ~wmask);
    end
end

// int_debl_cfg_lap_off_2[11:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_lap_off_2[11:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_LAP_OFF_2_DATA_0)
            int_debl_cfg_lap_off_2[11:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_lap_off_2[11:0] & ~wmask);
    end
end

// int_debl_cfg_lap_off_3[11:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_lap_off_3[11:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_LAP_OFF_3_DATA_0)
            int_debl_cfg_lap_off_3[11:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_lap_off_3[11:0] & ~wmask);
    end
end

// int_debl_cfg_lap_off_4[11:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_lap_off_4[11:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_LAP_OFF_4_DATA_0)
            int_debl_cfg_lap_off_4[11:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_lap_off_4[11:0] & ~wmask);
    end
end

// int_debl_cfg_lap_off_5[11:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_lap_off_5[11:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_LAP_OFF_5_DATA_0)
            int_debl_cfg_lap_off_5[11:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_lap_off_5[11:0] & ~wmask);
    end
end

// int_debl_cfg_theta_0[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_0[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_0_DATA_0)
            int_debl_cfg_theta_0[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_0[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_1[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_1[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_1_DATA_0)
            int_debl_cfg_theta_1[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_1[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_2[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_2[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_2_DATA_0)
            int_debl_cfg_theta_2[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_2[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_3[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_3[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_3_DATA_0)
            int_debl_cfg_theta_3[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_3[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_4[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_4[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_4_DATA_0)
            int_debl_cfg_theta_4[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_4[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_5[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_5[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_5_DATA_0)
            int_debl_cfg_theta_5[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_5[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_6[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_6[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_6_DATA_0)
            int_debl_cfg_theta_6[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_6[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_7[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_7[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_7_DATA_0)
            int_debl_cfg_theta_7[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_7[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_8[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_8[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_8_DATA_0)
            int_debl_cfg_theta_8[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_8[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_9[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_9[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_9_DATA_0)
            int_debl_cfg_theta_9[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_9[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_10[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_10[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_10_DATA_0)
            int_debl_cfg_theta_10[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_10[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_11[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_11[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_11_DATA_0)
            int_debl_cfg_theta_11[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_11[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_12[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_12[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_12_DATA_0)
            int_debl_cfg_theta_12[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_12[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_13[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_13[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_13_DATA_0)
            int_debl_cfg_theta_13[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_13[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_14[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_14[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_14_DATA_0)
            int_debl_cfg_theta_14[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_14[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_15[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_15[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_15_DATA_0)
            int_debl_cfg_theta_15[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_15[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_16[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_16[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_16_DATA_0)
            int_debl_cfg_theta_16[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_16[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_17[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_17[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_17_DATA_0)
            int_debl_cfg_theta_17[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_17[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_18[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_18[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_18_DATA_0)
            int_debl_cfg_theta_18[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_18[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_19[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_19[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_19_DATA_0)
            int_debl_cfg_theta_19[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_19[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_20[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_20[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_20_DATA_0)
            int_debl_cfg_theta_20[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_20[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_21[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_21[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_21_DATA_0)
            int_debl_cfg_theta_21[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_21[17:0] & ~wmask);
    end
end

// int_debl_cfg_theta_22[17:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_debl_cfg_theta_22[17:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_DEBL_CFG_THETA_22_DATA_0)
            int_debl_cfg_theta_22[17:0] <= (WDATA[31:0] & wmask) | (int_debl_cfg_theta_22[17:0] & ~wmask);
    end
end


//------------------------Memory logic-------------------

endmodule
