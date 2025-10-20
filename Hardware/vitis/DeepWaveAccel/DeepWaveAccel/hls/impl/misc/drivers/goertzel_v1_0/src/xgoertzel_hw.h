// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.1 (64-bit)
// Tool Version Limit: 2025.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// CTRL_BUS
// 0x200 ~
// 0x3ff : Memory 'cfg' (54 * 64b)
//         Word 2n   : bit [31:0] - cfg[n][31: 0]
//         Word 2n+1 : bit [31:0] - cfg[n][63:32]
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE 0x200
#define XGOERTZEL_CTRL_BUS_ADDR_CFG_HIGH 0x3ff
#define XGOERTZEL_CTRL_BUS_WIDTH_CFG     64
#define XGOERTZEL_CTRL_BUS_DEPTH_CFG     54

