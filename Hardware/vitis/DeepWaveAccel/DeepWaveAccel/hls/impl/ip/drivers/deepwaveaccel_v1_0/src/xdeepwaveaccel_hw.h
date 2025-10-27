// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.1 (64-bit)
// Tool Version Limit: 2025.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// CTRL_BUS
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

#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA_0_DATA  0x010
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA_0_DATA  18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA_1_DATA  0x018
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA_1_DATA  18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA2_0_DATA 0x020
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA2_0_DATA 18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA2_1_DATA 0x028
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA2_1_DATA 18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_SIN_OMEGA_0_DATA  0x030
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_SIN_OMEGA_0_DATA  18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_SIN_OMEGA_1_DATA  0x038
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_SIN_OMEGA_1_DATA  18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_N_LAYERS_DATA     0x040
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_N_LAYERS_DATA     8
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_K_DATA            0x048
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_K_DATA            6
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_0_DATA    0x050
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_LAP_OFF_0_DATA    12
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_1_DATA    0x058
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_LAP_OFF_1_DATA    12
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_2_DATA    0x060
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_LAP_OFF_2_DATA    12
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_3_DATA    0x068
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_LAP_OFF_3_DATA    12
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_4_DATA    0x070
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_LAP_OFF_4_DATA    12
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_5_DATA    0x078
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_LAP_OFF_5_DATA    12
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_0_DATA      0x080
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_0_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_1_DATA      0x088
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_1_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_2_DATA      0x090
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_2_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_3_DATA      0x098
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_3_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_4_DATA      0x0a0
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_4_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_5_DATA      0x0a8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_5_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_6_DATA      0x0b0
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_6_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_7_DATA      0x0b8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_7_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_8_DATA      0x0c0
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_8_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_9_DATA      0x0c8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_9_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_10_DATA     0x0d0
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_10_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_11_DATA     0x0d8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_11_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_12_DATA     0x0e0
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_12_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_13_DATA     0x0e8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_13_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_14_DATA     0x0f0
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_14_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_15_DATA     0x0f8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_15_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_16_DATA     0x100
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_16_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_17_DATA     0x108
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_17_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_18_DATA     0x110
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_18_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_19_DATA     0x118
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_19_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_20_DATA     0x120
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_20_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_21_DATA     0x128
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_21_DATA     18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_22_DATA     0x130
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_THETA_22_DATA     18

