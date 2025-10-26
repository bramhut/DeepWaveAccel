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

void XDeepwaveaccel_Set_debl_cfg_n_layers(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_N_LAYERS_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_n_layers(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_N_LAYERS_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_K(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_K_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_K(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_K_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_lap_off_0(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_0_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_lap_off_0(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_0_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_lap_off_1(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_1_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_lap_off_1(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_1_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_lap_off_2(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_2_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_lap_off_2(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_2_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_lap_off_3(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_3_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_lap_off_3(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_3_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_lap_off_4(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_4_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_lap_off_4(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_4_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_lap_off_5(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_5_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_lap_off_5(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_LAP_OFF_5_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_0(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_0_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_0(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_0_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_1(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_1_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_1(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_1_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_2(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_2_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_2(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_2_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_3(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_3_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_3(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_3_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_4(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_4_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_4(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_4_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_5(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_5_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_5(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_5_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_6(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_6_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_6(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_6_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_7(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_7_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_7(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_7_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_8(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_8_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_8(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_8_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_9(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_9_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_9(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_9_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_10(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_10_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_10(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_10_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_11(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_11_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_11(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_11_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_12(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_12_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_12(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_12_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_13(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_13_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_13(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_13_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_14(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_14_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_14(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_14_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_15(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_15_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_15(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_15_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_16(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_16_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_16(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_16_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_17(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_17_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_17(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_17_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_18(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_18_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_18(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_18_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_19(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_19_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_19(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_19_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_20(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_20_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_20(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_20_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_21(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_21_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_21(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_21_DATA);
    return Data;
}

void XDeepwaveaccel_Set_debl_cfg_theta_22(XDeepwaveaccel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XDeepwaveaccel_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_22_DATA, Data);
}

u32 XDeepwaveaccel_Get_debl_cfg_theta_22(XDeepwaveaccel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XDeepwaveaccel_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XDEEPWAVEACCEL_CTRL_BUS_ADDR_DEBL_CFG_THETA_22_DATA);
    return Data;
}

