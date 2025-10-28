// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.1 (64-bit)
// Tool Version Limit: 2025.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xdeepwaveaccel.h"

extern XDeepwaveaccel_Config XDeepwaveaccel_ConfigTable[];

#ifdef SDT
XDeepwaveaccel_Config *XDeepwaveaccel_LookupConfig(UINTPTR BaseAddress) {
	XDeepwaveaccel_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XDeepwaveaccel_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XDeepwaveaccel_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XDeepwaveaccel_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XDeepwaveaccel_Initialize(XDeepwaveaccel *InstancePtr, UINTPTR BaseAddress) {
	XDeepwaveaccel_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XDeepwaveaccel_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XDeepwaveaccel_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XDeepwaveaccel_Config *XDeepwaveaccel_LookupConfig(u16 DeviceId) {
	XDeepwaveaccel_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XDEEPWAVEACCEL_NUM_INSTANCES; Index++) {
		if (XDeepwaveaccel_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XDeepwaveaccel_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XDeepwaveaccel_Initialize(XDeepwaveaccel *InstancePtr, u16 DeviceId) {
	XDeepwaveaccel_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XDeepwaveaccel_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XDeepwaveaccel_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

