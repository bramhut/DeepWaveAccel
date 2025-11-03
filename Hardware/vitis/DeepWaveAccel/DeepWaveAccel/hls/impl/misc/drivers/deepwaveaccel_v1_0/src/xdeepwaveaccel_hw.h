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

#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA_0_DATA       0x010
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA_0_DATA       18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA_1_DATA       0x018
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA_1_DATA       18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA2_0_DATA      0x020
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA2_0_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA2_1_DATA      0x028
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_COS_OMEGA2_1_DATA      18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_SIN_OMEGA_0_DATA       0x030
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_SIN_OMEGA_0_DATA       18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_SIN_OMEGA_1_DATA       0x038
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_GOER_CFG_SIN_OMEGA_1_DATA       18
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_DATA                   0x040
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_DEBL_CFG_DATA                   8
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_IN_DATA       0x048
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_GZ_SAMPLES_IN_DATA       32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_IN_CTRL       0x04c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLE_WIN_DATA       0x058
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_GZ_SAMPLE_WIN_DATA       32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLE_WIN_CTRL       0x05c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_OUT_DATA      0x068
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_GZ_SAMPLES_OUT_DATA      32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_OUT_CTRL      0x06c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_DATA 0x078
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_GZ_SAMPLES_OUT_FIFO_DATA 32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_CTRL 0x07c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_STATE_DATA            0x088
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_CC_STATE_DATA            8
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_STATE_CTRL            0x08c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLES_IN_DATA       0x098
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_CC_SAMPLES_IN_DATA       32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLES_IN_CTRL       0x09c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLES_OUT_DATA      0x0a8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_CC_SAMPLES_OUT_DATA      32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLES_OUT_CTRL      0x0ac
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLE_IDX_DATA       0x0b8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_CC_SAMPLE_IDX_DATA       32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLE_IDX_CTRL       0x0bc
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_CURRENT_NORM_DATA     0x0c8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_CC_CURRENT_NORM_DATA     32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_CURRENT_NORM_CTRL     0x0cc
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_NORMS_WRITTEN_DATA    0x0d8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_CC_NORMS_WRITTEN_DATA    32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_NORMS_WRITTEN_CTRL    0x0dc
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_OUT_FIFO_DATA         0x0e8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_CC_OUT_FIFO_DATA         32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_OUT_FIFO_CTRL         0x0ec
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_NORMS_FIFO_DATA       0x0f8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_CC_NORMS_FIFO_DATA       32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_NORMS_FIFO_CTRL       0x0fc
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_CONFIG_LOADED_DATA    0x108
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_BP_CONFIG_LOADED_DATA    1
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_CONFIG_LOADED_CTRL    0x10c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_FSM_STATE_DATA        0x118
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_BP_FSM_STATE_DATA        8
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_FSM_STATE_CTRL        0x11c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_PARAM_STATE_DATA      0x128
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_BP_PARAM_STATE_DATA      8
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_PARAM_STATE_CTRL      0x12c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_IDX_DATA              0x138
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_BP_IDX_DATA              16
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_IDX_CTRL              0x13c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_SIGMAS_IN_DATA        0x148
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_BP_SIGMAS_IN_DATA        32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_SIGMAS_IN_CTRL        0x14c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_PIXELS_OUT_DATA       0x158
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_BP_PIXELS_OUT_DATA       32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_PIXELS_OUT_CTRL       0x15c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_OUT_FIFO_LEVEL_DATA   0x168
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_BP_OUT_FIFO_LEVEL_DATA   16
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_OUT_FIFO_LEVEL_CTRL   0x16c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_CONFIG_LOADED_DATA    0x178
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_DB_CONFIG_LOADED_DATA    1
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_CONFIG_LOADED_CTRL    0x17c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_FSM_STATE_DATA        0x188
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_DB_FSM_STATE_DATA        8
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_FSM_STATE_CTRL        0x18c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PARAM_STATE_DATA      0x198
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_DB_PARAM_STATE_DATA      8
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PARAM_STATE_CTRL      0x19c
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_IDX_DATA              0x1a8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_DB_IDX_DATA              16
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_IDX_CTRL              0x1ac
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PIXELS_IN_DATA        0x1b8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_DB_PIXELS_IN_DATA        32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PIXELS_IN_CTRL        0x1bc
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PIXELS_OUT_DATA       0x1c8
#define XDEEPWAVEACCEL_CTRL_BUS_BITS_STATUS_DB_PIXELS_OUT_DATA       32
#define XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PIXELS_OUT_CTRL       0x1cc

