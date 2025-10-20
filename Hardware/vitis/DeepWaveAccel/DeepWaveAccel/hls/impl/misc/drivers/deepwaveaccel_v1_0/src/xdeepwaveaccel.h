// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.1 (64-bit)
// Tool Version Limit: 2025.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XDEEPWAVEACCEL_H
#define XDEEPWAVEACCEL_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xdeepwaveaccel_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Ctrl_bus_BaseAddress;
} XDeepwaveaccel_Config;
#endif

typedef struct {
    u64 Ctrl_bus_BaseAddress;
    u32 IsReady;
} XDeepwaveaccel;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XDeepwaveaccel_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XDeepwaveaccel_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XDeepwaveaccel_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XDeepwaveaccel_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XDeepwaveaccel_Initialize(XDeepwaveaccel *InstancePtr, UINTPTR BaseAddress);
XDeepwaveaccel_Config* XDeepwaveaccel_LookupConfig(UINTPTR BaseAddress);
#else
int XDeepwaveaccel_Initialize(XDeepwaveaccel *InstancePtr, u16 DeviceId);
XDeepwaveaccel_Config* XDeepwaveaccel_LookupConfig(u16 DeviceId);
#endif
int XDeepwaveaccel_CfgInitialize(XDeepwaveaccel *InstancePtr, XDeepwaveaccel_Config *ConfigPtr);
#else
int XDeepwaveaccel_Initialize(XDeepwaveaccel *InstancePtr, const char* InstanceName);
int XDeepwaveaccel_Release(XDeepwaveaccel *InstancePtr);
#endif


u32 XDeepwaveaccel_Get_goer_cfg_BaseAddress(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_goer_cfg_HighAddress(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_goer_cfg_TotalBytes(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_goer_cfg_BitWidth(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_goer_cfg_Depth(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Write_goer_cfg_Words(XDeepwaveaccel *InstancePtr, int offset, word_type *data, int length);
u32 XDeepwaveaccel_Read_goer_cfg_Words(XDeepwaveaccel *InstancePtr, int offset, word_type *data, int length);
u32 XDeepwaveaccel_Write_goer_cfg_Bytes(XDeepwaveaccel *InstancePtr, int offset, char *data, int length);
u32 XDeepwaveaccel_Read_goer_cfg_Bytes(XDeepwaveaccel *InstancePtr, int offset, char *data, int length);

#ifdef __cplusplus
}
#endif

#endif
