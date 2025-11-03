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
void XDeepwaveaccel_Set_debl_cfg(XDeepwaveaccel *InstancePtr, u32 Data);
u32 XDeepwaveaccel_Get_debl_cfg(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_gz_samples_in(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_gz_samples_in_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_gz_sample_win(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_gz_sample_win_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_gz_samples_out(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_gz_samples_out_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_gz_samples_out_fifo(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_gz_samples_out_fifo_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_state(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_state_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_samples_in(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_samples_in_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_samples_out(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_samples_out_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_sample_idx(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_sample_idx_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_current_norm(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_current_norm_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_norms_written(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_norms_written_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_out_fifo(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_out_fifo_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_norms_fifo(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_cc_norms_fifo_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_config_loaded(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_config_loaded_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_fsm_state(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_fsm_state_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_param_state(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_param_state_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_idx(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_idx_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_sigmas_in(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_sigmas_in_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_pixels_out(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_pixels_out_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_out_fifo_level(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_bp_out_fifo_level_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_config_loaded(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_config_loaded_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_fsm_state(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_fsm_state_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_param_state(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_param_state_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_idx(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_idx_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_pixels_in(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_pixels_in_vld(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_pixels_out(XDeepwaveaccel *InstancePtr);
u32 XDeepwaveaccel_Get_status_db_pixels_out_vld(XDeepwaveaccel *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
