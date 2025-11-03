// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.1 (64-bit)
// Tool Version Limit: 2025.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xdeepwaveaccel.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XDeepwaveaccel_CfgInitialize(XDeepwaveaccel *InstancePtr, XDeepwaveaccel_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Ctrl_bus_BaseAddress = ConfigPtr->Ctrl_bus_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XDeepwaveaccel_Set_goer_cfg_COS_OMEGA_0(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA_0_DATA, Data);
}

u32 XDeepwaveaccel_Get_goer_cfg_COS_OMEGA_0(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA_0_DATA);
    return Data;
}

void XDeepwaveaccel_Set_goer_cfg_COS_OMEGA_1(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA_1_DATA, Data);
}

u32 XDeepwaveaccel_Get_goer_cfg_COS_OMEGA_1(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA_1_DATA);
    return Data;
}

void XDeepwaveaccel_Set_goer_cfg_COS_OMEGA2_0(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA2_0_DATA, Data);
}

u32 XDeepwaveaccel_Get_goer_cfg_COS_OMEGA2_0(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA2_0_DATA);
    return Data;
}

void XDeepwaveaccel_Set_goer_cfg_COS_OMEGA2_1(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA2_1_DATA, Data);
}

u32 XDeepwaveaccel_Get_goer_cfg_COS_OMEGA2_1(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_COS_OMEGA2_1_DATA);
    return Data;
}

void XDeepwaveaccel_Set_goer_cfg_SIN_OMEGA_0(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_SIN_OMEGA_0_DATA, Data);
}

u32 XDeepwaveaccel_Get_goer_cfg_SIN_OMEGA_0(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_SIN_OMEGA_0_DATA);
    return Data;
}

void XDeepwaveaccel_Set_goer_cfg_SIN_OMEGA_1(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_SIN_OMEGA_1_DATA, Data);
}

u32 XDeepwaveaccel_Get_goer_cfg_SIN_OMEGA_1(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_SIN_OMEGA_1_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_gz_samples_in(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_IN_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_gz_samples_in_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_IN_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_gz_sample_win(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLE_WIN_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_gz_sample_win_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLE_WIN_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_gz_samples_out(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_OUT_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_gz_samples_out_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_OUT_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_gz_samples_out_fifo(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_gz_samples_out_fifo_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_cc_state(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_STATE_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_cc_state_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_STATE_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_cc_samples_in(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLES_IN_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_cc_samples_in_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLES_IN_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_cc_samples_out(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLES_OUT_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_cc_samples_out_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLES_OUT_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_cc_sample_idx(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLE_IDX_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_cc_sample_idx_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_SAMPLE_IDX_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_cc_current_norm(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_CURRENT_NORM_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_cc_current_norm_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_CURRENT_NORM_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_cc_norms_written(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_NORMS_WRITTEN_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_cc_norms_written_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_NORMS_WRITTEN_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_cc_out_fifo(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_OUT_FIFO_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_cc_out_fifo_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_OUT_FIFO_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_cc_norms_fifo(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_NORMS_FIFO_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_cc_norms_fifo_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_CC_NORMS_FIFO_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_bp_config_loaded(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_CONFIG_LOADED_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_bp_config_loaded_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_CONFIG_LOADED_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_bp_fsm_state(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_FSM_STATE_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_bp_fsm_state_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_FSM_STATE_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_bp_param_state(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_PARAM_STATE_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_bp_param_state_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_PARAM_STATE_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_bp_idx(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_IDX_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_bp_idx_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_IDX_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_bp_sigmas_in(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_SIGMAS_IN_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_bp_sigmas_in_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_SIGMAS_IN_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_bp_pixels_out(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_PIXELS_OUT_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_bp_pixels_out_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_PIXELS_OUT_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_bp_out_fifo_level(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_OUT_FIFO_LEVEL_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_bp_out_fifo_level_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_BP_OUT_FIFO_LEVEL_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_db_config_loaded(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_CONFIG_LOADED_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_db_config_loaded_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_CONFIG_LOADED_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_db_fsm_state(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_FSM_STATE_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_db_fsm_state_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_FSM_STATE_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_db_param_state(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PARAM_STATE_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_db_param_state_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PARAM_STATE_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_db_idx(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_IDX_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_db_idx_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_IDX_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_db_pixels_in(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PIXELS_IN_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_db_pixels_in_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PIXELS_IN_CTRL);
    return Data & 0x1;
}

u32 XDeepwaveaccel_Get_status_db_pixels_out(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PIXELS_OUT_DATA);
    return Data;
}

u32 XDeepwaveaccel_Get_status_db_pixels_out_vld(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_STATUS_DB_PIXELS_OUT_CTRL);
    return Data & 0x1;
}

