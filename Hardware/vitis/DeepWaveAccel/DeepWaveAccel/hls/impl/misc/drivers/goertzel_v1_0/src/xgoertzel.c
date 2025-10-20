// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.1 (64-bit)
// Tool Version Limit: 2025.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xgoertzel.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XGoertzel_CfgInitialize(XGoertzel *InstancePtr, XGoertzel_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Ctrl_bus_BaseAddress = ConfigPtr->Ctrl_bus_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

u32 XGoertzel_Get_cfg_BaseAddress(XGoertzel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return (InstancePtr->Ctrl_bus_BaseAddress + XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE);
}

u32 XGoertzel_Get_cfg_HighAddress(XGoertzel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return (InstancePtr->Ctrl_bus_BaseAddress + XGOERTZEL_CTRL_BUS_ADDR_CFG_HIGH);
}

u32 XGoertzel_Get_cfg_TotalBytes(XGoertzel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return (XGOERTZEL_CTRL_BUS_ADDR_CFG_HIGH - XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE + 1);
}

u32 XGoertzel_Get_cfg_BitWidth(XGoertzel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XGOERTZEL_CTRL_BUS_WIDTH_CFG;
}

u32 XGoertzel_Get_cfg_Depth(XGoertzel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XGOERTZEL_CTRL_BUS_DEPTH_CFG;
}

u32 XGoertzel_Write_cfg_Words(XGoertzel *InstancePtr, int offset, word_type *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length)*4 > (XGOERTZEL_CTRL_BUS_ADDR_CFG_HIGH - XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(int *)(InstancePtr->Ctrl_bus_BaseAddress + XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE + (offset + i)*4) = *(data + i);
    }
    return length;
}

u32 XGoertzel_Read_cfg_Words(XGoertzel *InstancePtr, int offset, word_type *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length)*4 > (XGOERTZEL_CTRL_BUS_ADDR_CFG_HIGH - XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(data + i) = *(int *)(InstancePtr->Ctrl_bus_BaseAddress + XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE + (offset + i)*4);
    }
    return length;
}

u32 XGoertzel_Write_cfg_Bytes(XGoertzel *InstancePtr, int offset, char *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length) > (XGOERTZEL_CTRL_BUS_ADDR_CFG_HIGH - XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(char *)(InstancePtr->Ctrl_bus_BaseAddress + XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE + offset + i) = *(data + i);
    }
    return length;
}

u32 XGoertzel_Read_cfg_Bytes(XGoertzel *InstancePtr, int offset, char *data, int length) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr -> IsReady == XIL_COMPONENT_IS_READY);

    int i;

    if ((offset + length) > (XGOERTZEL_CTRL_BUS_ADDR_CFG_HIGH - XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE + 1))
        return 0;

    for (i = 0; i < length; i++) {
        *(data + i) = *(char *)(InstancePtr->Ctrl_bus_BaseAddress + XGOERTZEL_CTRL_BUS_ADDR_CFG_BASE + offset + i);
    }
    return length;
}

