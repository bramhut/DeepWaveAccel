// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.1 (64-bit)
// Tool Version Limit: 2025.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XGOERTZEL_H
#define XGOERTZEL_H

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
#include "xgoertzel_hw.h"

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
} XGoertzel_Config;
#endif

typedef struct {
    u64 Ctrl_bus_BaseAddress;
    u32 IsReady;
} XGoertzel;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XGoertzel_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XGoertzel_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XGoertzel_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XGoertzel_ReadReg(BaseAddress, RegOffset) \
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
int XGoertzel_Initialize(XGoertzel *InstancePtr, UINTPTR BaseAddress);
XGoertzel_Config* XGoertzel_LookupConfig(UINTPTR BaseAddress);
#else
int XGoertzel_Initialize(XGoertzel *InstancePtr, u16 DeviceId);
XGoertzel_Config* XGoertzel_LookupConfig(u16 DeviceId);
#endif
int XGoertzel_CfgInitialize(XGoertzel *InstancePtr, XGoertzel_Config *ConfigPtr);
#else
int XGoertzel_Initialize(XGoertzel *InstancePtr, const char* InstanceName);
int XGoertzel_Release(XGoertzel *InstancePtr);
#endif


u32 XGoertzel_Get_cfg_BaseAddress(XGoertzel *InstancePtr);
u32 XGoertzel_Get_cfg_HighAddress(XGoertzel *InstancePtr);
u32 XGoertzel_Get_cfg_TotalBytes(XGoertzel *InstancePtr);
u32 XGoertzel_Get_cfg_BitWidth(XGoertzel *InstancePtr);
u32 XGoertzel_Get_cfg_Depth(XGoertzel *InstancePtr);
u32 XGoertzel_Write_cfg_Words(XGoertzel *InstancePtr, int offset, word_type *data, int length);
u32 XGoertzel_Read_cfg_Words(XGoertzel *InstancePtr, int offset, word_type *data, int length);
u32 XGoertzel_Write_cfg_Bytes(XGoertzel *InstancePtr, int offset, char *data, int length);
u32 XGoertzel_Read_cfg_Bytes(XGoertzel *InstancePtr, int offset, char *data, int length);

#ifdef __cplusplus
}
#endif

#endif
