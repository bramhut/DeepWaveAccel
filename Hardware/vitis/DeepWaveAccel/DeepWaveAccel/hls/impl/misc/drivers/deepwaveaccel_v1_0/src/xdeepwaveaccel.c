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

u32 XDeepwaveaccel_Get_goer_cfg_BaseAddress(XDeepwaveaccel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return (InstancePtr->Ctrl_bus_BaseAddress + XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_BASE);
}

u32 XDeepwaveaccel_Get_goer_cfg_HighAddress(XDeepwaveaccel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return (InstancePtr->Ctrl_bus_BaseAddress + XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_HIGH);
}

u32 XDeepwaveaccel_Get_goer_cfg_TotalBytes(XDeepwaveaccel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return (XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_HIGH - XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_BASE + 1);
}

u32 XDeepwaveaccel_Get_goer_cfg_BitWidth(XDeepwaveaccel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XDEEPWAVEACCEL_CTRL_BUS_WIDTH_GOER_CFG;
}

u32 XDeepwaveaccel_Get_goer_cfg_Depth(XDeepwaveaccel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XDEEPWAVEACCEL_CTRL_BUS_DEPTH_GOER_CFG;
}

u32 XDeepwaveaccel_Write_goer_cfg_Words(XDeepwaveaccel *InstancePtr, int offset, word_type *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length)*4 > (XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_HIGH - XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(int *)(InstancePtr->Ctrl_bus_BaseAddress + XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_BASE + (offset + i)*4) = *(data + i);
    }
    return length;
}

u32 XDeepwaveaccel_Read_goer_cfg_Words(XDeepwaveaccel *InstancePtr, int offset, word_type *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length)*4 > (XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_HIGH - XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(data + i) = *(int *)(InstancePtr->Ctrl_bus_BaseAddress + XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_BASE + (offset + i)*4);
    }
    return length;
}

u32 XDeepwaveaccel_Write_goer_cfg_Bytes(XDeepwaveaccel *InstancePtr, int offset, char *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length) > (XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_HIGH - XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(char *)(InstancePtr->Ctrl_bus_BaseAddress + XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_BASE + offset + i) = *(data + i);
    }
    return length;
}

u32 XDeepwaveaccel_Read_goer_cfg_Bytes(XDeepwaveaccel *InstancePtr, int offset, char *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length) > (XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_HIGH - XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(data + i) = *(char *)(InstancePtr->Ctrl_bus_BaseAddress + XDEEPWAVEACCEL_CTRL_BUS_ADDR_GOER_CFG_BASE + offset + i);
    }
    return length;
}

