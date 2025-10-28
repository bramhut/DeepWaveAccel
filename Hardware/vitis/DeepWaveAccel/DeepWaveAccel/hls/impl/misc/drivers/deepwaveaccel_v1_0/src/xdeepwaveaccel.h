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
    u64 Control_BaseAddress;
    u64 Ctrl_bus_BaseAddress;
} XDeepwaveaccel_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
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


void XDeepwaveaccel_Set_b_ddr(XDeepwaveaccel *InstancePtr, u64 Data);
u64 XDeepwaveaccel_Get_b_ddr(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_tau_ddr(XDeepwaveaccel *InstancePtr, u64 Data);
u64 XDeepwaveaccel_Get_tau_ddr(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_lap_ddr(XDeepwaveaccel *InstancePtr, u64 Data);
u64 XDeepwaveaccel_Get_lap_ddr(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_goer_cfg_COS_OMEGA_0(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_goer_cfg_COS_OMEGA_0(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_goer_cfg_COS_OMEGA_1(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_goer_cfg_COS_OMEGA_1(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_goer_cfg_COS_OMEGA2_0(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_goer_cfg_COS_OMEGA2_0(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_goer_cfg_COS_OMEGA2_1(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_goer_cfg_COS_OMEGA2_1(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_goer_cfg_SIN_OMEGA_0(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_goer_cfg_SIN_OMEGA_0(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_goer_cfg_SIN_OMEGA_1(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_goer_cfg_SIN_OMEGA_1(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_n_layers(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_n_layers(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_K(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_K(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_lap_off_0(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_lap_off_0(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_lap_off_1(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_lap_off_1(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_lap_off_2(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_lap_off_2(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_lap_off_3(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_lap_off_3(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_lap_off_4(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_lap_off_4(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_lap_off_5(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_lap_off_5(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_0(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_0(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_1(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_1(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_2(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_2(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_3(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_3(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_4(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_4(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_5(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_5(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_6(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_6(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_7(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_7(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_8(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_8(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_9(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_9(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_10(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_10(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_11(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_11(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_12(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_12(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_13(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_13(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_14(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_14(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_15(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_15(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_16(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_16(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_17(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_17(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_18(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_18(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_19(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_19(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_20(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_20(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_21(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_21(XDeepwaveaccel *InstancePtr);
void XDeepwaveaccel_Set_debl_cfg_theta_22(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg_theta_22(XDeepwaveaccel *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
