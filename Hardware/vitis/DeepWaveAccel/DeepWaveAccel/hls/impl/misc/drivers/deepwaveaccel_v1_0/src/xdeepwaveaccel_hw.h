// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.1 (64-bit)
// Tool Version Limit: 2025.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// CTRL_BUS
// 0x00 : reserved
// 0x04 : reserved
// 0x08 : reserved
// 0x0c : reserved
// 0x10 : Data signal of goer_cfg_COS_OMEGA_0
//        bit 17~0 - goer_cfg_COS_OMEGA_0[17:0] (Read/Write)
//        others   - reserved
// 0x14 : reserved
// 0x18 : Data signal of goer_cfg_COS_OMEGA_1
//        bit 17~0 - goer_cfg_COS_OMEGA_1[17:0] (Read/Write)
//        others   - reserved
// 0x1c : reserved
// 0x20 : Data signal of goer_cfg_COS_OMEGA2_0
//        bit 17~0 - goer_cfg_COS_OMEGA2_0[17:0] (Read/Write)
//        others   - reserved
// 0x24 : reserved
// 0x28 : Data signal of goer_cfg_COS_OMEGA2_1
//        bit 17~0 - goer_cfg_COS_OMEGA2_1[17:0] (Read/Write)
//        others   - reserved
// 0x2c : reserved
// 0x30 : Data signal of goer_cfg_SIN_OMEGA_0
//        bit 17~0 - goer_cfg_SIN_OMEGA_0[17:0] (Read/Write)
//        others   - reserved
// 0x34 : reserved
// 0x38 : Data signal of goer_cfg_SIN_OMEGA_1
//        bit 17~0 - goer_cfg_SIN_OMEGA_1[17:0] (Read/Write)
//        others   - reserved
// 0x3c : reserved
// 0x40 : Data signal of debl_cfg
//        bit 7~0 - debl_cfg[7:0] (Read/Write)
//        others  - reserved
// 0x44 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA_0_DATA  0x10
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA_0_DATA  18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA_1_DATA  0x18
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA_1_DATA  18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA2_0_DATA 0x20
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA2_0_DATA 18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA2_1_DATA 0x28
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA2_1_DATA 18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_SIN_OMEGA_0_DATA  0x30
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_SIN_OMEGA_0_DATA  18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_SIN_OMEGA_1_DATA  0x38
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_SIN_OMEGA_1_DATA  18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_DATA              0x40
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_DATA              8

