-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.1 (64-bit)
-- Tool Version Limit: 2025.05
-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
-- 
-- ==============================================================
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

entity deepwaveaccel_CTRL_BUS_s_axi is
generic (
    C_S_AXI_ADDR_WIDTH    : INTEGER := 9;
    C_S_AXI_DATA_WIDTH    : INTEGER := 32);
port (
    ACLK                  :in   STD_LOGIC;
    ARESET                :in   STD_LOGIC;
    ACLK_EN               :in   STD_LOGIC;
    AWADDR                :in   STD_LOGIC_VECTOR(C_S_AXI_ADDR_WIDTH-1 downto 0);
    AWVALID               :in   STD_LOGIC;
    AWREADY               :out  STD_LOGIC;
    WDATA                 :in   STD_LOGIC_VECTOR(C_S_AXI_DATA_WIDTH-1 downto 0);
    WSTRB                 :in   STD_LOGIC_VECTOR(C_S_AXI_DATA_WIDTH/8-1 downto 0);
    WVALID                :in   STD_LOGIC;
    WREADY                :out  STD_LOGIC;
    BRESP                 :out  STD_LOGIC_VECTOR(1 downto 0);
    BVALID                :out  STD_LOGIC;
    BREADY                :in   STD_LOGIC;
    ARADDR                :in   STD_LOGIC_VECTOR(C_S_AXI_ADDR_WIDTH-1 downto 0);
    ARVALID               :in   STD_LOGIC;
    ARREADY               :out  STD_LOGIC;
    RDATA                 :out  STD_LOGIC_VECTOR(C_S_AXI_DATA_WIDTH-1 downto 0);
    RRESP                 :out  STD_LOGIC_VECTOR(1 downto 0);
    RVALID                :out  STD_LOGIC;
    RREADY                :in   STD_LOGIC;
    goer_cfg_COS_OMEGA_0  :out  STD_LOGIC_VECTOR(17 downto 0);
    goer_cfg_COS_OMEGA_1  :out  STD_LOGIC_VECTOR(17 downto 0);
    goer_cfg_COS_OMEGA2_0 :out  STD_LOGIC_VECTOR(17 downto 0);
    goer_cfg_COS_OMEGA2_1 :out  STD_LOGIC_VECTOR(17 downto 0);
    goer_cfg_SIN_OMEGA_0  :out  STD_LOGIC_VECTOR(17 downto 0);
    goer_cfg_SIN_OMEGA_1  :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg              :out  STD_LOGIC_VECTOR(7 downto 0);
    status_gz_samples_in  :in   STD_LOGIC_VECTOR(31 downto 0);
    status_gz_samples_in_ap_vld :in   STD_LOGIC;
    status_gz_sample_win  :in   STD_LOGIC_VECTOR(31 downto 0);
    status_gz_sample_win_ap_vld :in   STD_LOGIC;
    status_gz_samples_out :in   STD_LOGIC_VECTOR(31 downto 0);
    status_gz_samples_out_ap_vld :in   STD_LOGIC;
    status_gz_samples_out_fifo :in   STD_LOGIC_VECTOR(31 downto 0);
    status_gz_samples_out_fifo_ap_vld :in   STD_LOGIC;
    status_cc_state       :in   STD_LOGIC_VECTOR(7 downto 0);
    status_cc_state_ap_vld :in   STD_LOGIC;
    status_cc_samples_in  :in   STD_LOGIC_VECTOR(31 downto 0);
    status_cc_samples_in_ap_vld :in   STD_LOGIC;
    status_cc_samples_out :in   STD_LOGIC_VECTOR(31 downto 0);
    status_cc_samples_out_ap_vld :in   STD_LOGIC;
    status_cc_sample_idx  :in   STD_LOGIC_VECTOR(31 downto 0);
    status_cc_sample_idx_ap_vld :in   STD_LOGIC;
    status_cc_current_norm :in   STD_LOGIC_VECTOR(31 downto 0);
    status_cc_current_norm_ap_vld :in   STD_LOGIC;
    status_cc_norms_written :in   STD_LOGIC_VECTOR(31 downto 0);
    status_cc_norms_written_ap_vld :in   STD_LOGIC;
    status_cc_out_fifo    :in   STD_LOGIC_VECTOR(31 downto 0);
    status_cc_out_fifo_ap_vld :in   STD_LOGIC;
    status_cc_norms_fifo  :in   STD_LOGIC_VECTOR(31 downto 0);
    status_cc_norms_fifo_ap_vld :in   STD_LOGIC;
    status_bp_config_loaded :in   STD_LOGIC_VECTOR(0 downto 0);
    status_bp_config_loaded_ap_vld :in   STD_LOGIC;
    status_bp_fsm_state   :in   STD_LOGIC_VECTOR(7 downto 0);
    status_bp_fsm_state_ap_vld :in   STD_LOGIC;
    status_bp_param_state :in   STD_LOGIC_VECTOR(7 downto 0);
    status_bp_param_state_ap_vld :in   STD_LOGIC;
    status_bp_idx         :in   STD_LOGIC_VECTOR(15 downto 0);
    status_bp_idx_ap_vld  :in   STD_LOGIC;
    status_bp_sigmas_in   :in   STD_LOGIC_VECTOR(31 downto 0);
    status_bp_sigmas_in_ap_vld :in   STD_LOGIC;
    status_bp_pixels_out  :in   STD_LOGIC_VECTOR(31 downto 0);
    status_bp_pixels_out_ap_vld :in   STD_LOGIC;
    status_bp_out_fifo_level :in   STD_LOGIC_VECTOR(15 downto 0);
    status_bp_out_fifo_level_ap_vld :in   STD_LOGIC;
    status_db_config_loaded :in   STD_LOGIC_VECTOR(0 downto 0);
    status_db_config_loaded_ap_vld :in   STD_LOGIC;
    status_db_fsm_state   :in   STD_LOGIC_VECTOR(7 downto 0);
    status_db_fsm_state_ap_vld :in   STD_LOGIC;
    status_db_param_state :in   STD_LOGIC_VECTOR(7 downto 0);
    status_db_param_state_ap_vld :in   STD_LOGIC;
    status_db_idx         :in   STD_LOGIC_VECTOR(15 downto 0);
    status_db_idx_ap_vld  :in   STD_LOGIC;
    status_db_pixels_in   :in   STD_LOGIC_VECTOR(31 downto 0);
    status_db_pixels_in_ap_vld :in   STD_LOGIC;
    status_db_pixels_out  :in   STD_LOGIC_VECTOR(31 downto 0);
    status_db_pixels_out_ap_vld :in   STD_LOGIC
);
end entity deepwaveaccel_CTRL_BUS_s_axi;

-- ------------------------Address Info-------------------
-- Protocol Used: ap_ctrl_none
--
-- 0x000 : reserved
-- 0x004 : reserved
-- 0x008 : reserved
-- 0x00c : reserved
-- 0x010 : Data signal of goer_cfg_COS_OMEGA_0
--         bit 17~0 - goer_cfg_COS_OMEGA_0[17:0] (Read/Write)
--         others   - reserved
-- 0x014 : reserved
-- 0x018 : Data signal of goer_cfg_COS_OMEGA_1
--         bit 17~0 - goer_cfg_COS_OMEGA_1[17:0] (Read/Write)
--         others   - reserved
-- 0x01c : reserved
-- 0x020 : Data signal of goer_cfg_COS_OMEGA2_0
--         bit 17~0 - goer_cfg_COS_OMEGA2_0[17:0] (Read/Write)
--         others   - reserved
-- 0x024 : reserved
-- 0x028 : Data signal of goer_cfg_COS_OMEGA2_1
--         bit 17~0 - goer_cfg_COS_OMEGA2_1[17:0] (Read/Write)
--         others   - reserved
-- 0x02c : reserved
-- 0x030 : Data signal of goer_cfg_SIN_OMEGA_0
--         bit 17~0 - goer_cfg_SIN_OMEGA_0[17:0] (Read/Write)
--         others   - reserved
-- 0x034 : reserved
-- 0x038 : Data signal of goer_cfg_SIN_OMEGA_1
--         bit 17~0 - goer_cfg_SIN_OMEGA_1[17:0] (Read/Write)
--         others   - reserved
-- 0x03c : reserved
-- 0x040 : Data signal of debl_cfg
--         bit 7~0 - debl_cfg[7:0] (Read/Write)
--         others  - reserved
-- 0x044 : reserved
-- 0x048 : Data signal of status_gz_samples_in
--         bit 31~0 - status_gz_samples_in[31:0] (Read)
-- 0x04c : Control signal of status_gz_samples_in
--         bit 0  - status_gz_samples_in_ap_vld (Read/COR)
--         others - reserved
-- 0x058 : Data signal of status_gz_sample_win
--         bit 31~0 - status_gz_sample_win[31:0] (Read)
-- 0x05c : Control signal of status_gz_sample_win
--         bit 0  - status_gz_sample_win_ap_vld (Read/COR)
--         others - reserved
-- 0x068 : Data signal of status_gz_samples_out
--         bit 31~0 - status_gz_samples_out[31:0] (Read)
-- 0x06c : Control signal of status_gz_samples_out
--         bit 0  - status_gz_samples_out_ap_vld (Read/COR)
--         others - reserved
-- 0x078 : Data signal of status_gz_samples_out_fifo
--         bit 31~0 - status_gz_samples_out_fifo[31:0] (Read)
-- 0x07c : Control signal of status_gz_samples_out_fifo
--         bit 0  - status_gz_samples_out_fifo_ap_vld (Read/COR)
--         others - reserved
-- 0x088 : Data signal of status_cc_state
--         bit 7~0 - status_cc_state[7:0] (Read)
--         others  - reserved
-- 0x08c : Control signal of status_cc_state
--         bit 0  - status_cc_state_ap_vld (Read/COR)
--         others - reserved
-- 0x098 : Data signal of status_cc_samples_in
--         bit 31~0 - status_cc_samples_in[31:0] (Read)
-- 0x09c : Control signal of status_cc_samples_in
--         bit 0  - status_cc_samples_in_ap_vld (Read/COR)
--         others - reserved
-- 0x0a8 : Data signal of status_cc_samples_out
--         bit 31~0 - status_cc_samples_out[31:0] (Read)
-- 0x0ac : Control signal of status_cc_samples_out
--         bit 0  - status_cc_samples_out_ap_vld (Read/COR)
--         others - reserved
-- 0x0b8 : Data signal of status_cc_sample_idx
--         bit 31~0 - status_cc_sample_idx[31:0] (Read)
-- 0x0bc : Control signal of status_cc_sample_idx
--         bit 0  - status_cc_sample_idx_ap_vld (Read/COR)
--         others - reserved
-- 0x0c8 : Data signal of status_cc_current_norm
--         bit 31~0 - status_cc_current_norm[31:0] (Read)
-- 0x0cc : Control signal of status_cc_current_norm
--         bit 0  - status_cc_current_norm_ap_vld (Read/COR)
--         others - reserved
-- 0x0d8 : Data signal of status_cc_norms_written
--         bit 31~0 - status_cc_norms_written[31:0] (Read)
-- 0x0dc : Control signal of status_cc_norms_written
--         bit 0  - status_cc_norms_written_ap_vld (Read/COR)
--         others - reserved
-- 0x0e8 : Data signal of status_cc_out_fifo
--         bit 31~0 - status_cc_out_fifo[31:0] (Read)
-- 0x0ec : Control signal of status_cc_out_fifo
--         bit 0  - status_cc_out_fifo_ap_vld (Read/COR)
--         others - reserved
-- 0x0f8 : Data signal of status_cc_norms_fifo
--         bit 31~0 - status_cc_norms_fifo[31:0] (Read)
-- 0x0fc : Control signal of status_cc_norms_fifo
--         bit 0  - status_cc_norms_fifo_ap_vld (Read/COR)
--         others - reserved
-- 0x108 : Data signal of status_bp_config_loaded
--         bit 0  - status_bp_config_loaded[0] (Read)
--         others - reserved
-- 0x10c : Control signal of status_bp_config_loaded
--         bit 0  - status_bp_config_loaded_ap_vld (Read/COR)
--         others - reserved
-- 0x118 : Data signal of status_bp_fsm_state
--         bit 7~0 - status_bp_fsm_state[7:0] (Read)
--         others  - reserved
-- 0x11c : Control signal of status_bp_fsm_state
--         bit 0  - status_bp_fsm_state_ap_vld (Read/COR)
--         others - reserved
-- 0x128 : Data signal of status_bp_param_state
--         bit 7~0 - status_bp_param_state[7:0] (Read)
--         others  - reserved
-- 0x12c : Control signal of status_bp_param_state
--         bit 0  - status_bp_param_state_ap_vld (Read/COR)
--         others - reserved
-- 0x138 : Data signal of status_bp_idx
--         bit 15~0 - status_bp_idx[15:0] (Read)
--         others   - reserved
-- 0x13c : Control signal of status_bp_idx
--         bit 0  - status_bp_idx_ap_vld (Read/COR)
--         others - reserved
-- 0x148 : Data signal of status_bp_sigmas_in
--         bit 31~0 - status_bp_sigmas_in[31:0] (Read)
-- 0x14c : Control signal of status_bp_sigmas_in
--         bit 0  - status_bp_sigmas_in_ap_vld (Read/COR)
--         others - reserved
-- 0x158 : Data signal of status_bp_pixels_out
--         bit 31~0 - status_bp_pixels_out[31:0] (Read)
-- 0x15c : Control signal of status_bp_pixels_out
--         bit 0  - status_bp_pixels_out_ap_vld (Read/COR)
--         others - reserved
-- 0x168 : Data signal of status_bp_out_fifo_level
--         bit 15~0 - status_bp_out_fifo_level[15:0] (Read)
--         others   - reserved
-- 0x16c : Control signal of status_bp_out_fifo_level
--         bit 0  - status_bp_out_fifo_level_ap_vld (Read/COR)
--         others - reserved
-- 0x178 : Data signal of status_db_config_loaded
--         bit 0  - status_db_config_loaded[0] (Read)
--         others - reserved
-- 0x17c : Control signal of status_db_config_loaded
--         bit 0  - status_db_config_loaded_ap_vld (Read/COR)
--         others - reserved
-- 0x188 : Data signal of status_db_fsm_state
--         bit 7~0 - status_db_fsm_state[7:0] (Read)
--         others  - reserved
-- 0x18c : Control signal of status_db_fsm_state
--         bit 0  - status_db_fsm_state_ap_vld (Read/COR)
--         others - reserved
-- 0x198 : Data signal of status_db_param_state
--         bit 7~0 - status_db_param_state[7:0] (Read)
--         others  - reserved
-- 0x19c : Control signal of status_db_param_state
--         bit 0  - status_db_param_state_ap_vld (Read/COR)
--         others - reserved
-- 0x1a8 : Data signal of status_db_idx
--         bit 15~0 - status_db_idx[15:0] (Read)
--         others   - reserved
-- 0x1ac : Control signal of status_db_idx
--         bit 0  - status_db_idx_ap_vld (Read/COR)
--         others - reserved
-- 0x1b8 : Data signal of status_db_pixels_in
--         bit 31~0 - status_db_pixels_in[31:0] (Read)
-- 0x1bc : Control signal of status_db_pixels_in
--         bit 0  - status_db_pixels_in_ap_vld (Read/COR)
--         others - reserved
-- 0x1c8 : Data signal of status_db_pixels_out
--         bit 31~0 - status_db_pixels_out[31:0] (Read)
-- 0x1cc : Control signal of status_db_pixels_out
--         bit 0  - status_db_pixels_out_ap_vld (Read/COR)
--         others - reserved
-- (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

architecture behave of deepwaveaccel_CTRL_BUS_s_axi is
    type states is (wridle, wrdata, wrresp, wrreset, rdidle, rddata, rdreset);  -- read and write fsm states
    signal wstate  : states := wrreset;
    signal rstate  : states := rdreset;
    signal wnext, rnext: states;
    constant ADDR_GOER_CFG_COS_OMEGA_0_DATA_0       : INTEGER := 16#010#;
    constant ADDR_GOER_CFG_COS_OMEGA_0_CTRL         : INTEGER := 16#014#;
    constant ADDR_GOER_CFG_COS_OMEGA_1_DATA_0       : INTEGER := 16#018#;
    constant ADDR_GOER_CFG_COS_OMEGA_1_CTRL         : INTEGER := 16#01c#;
    constant ADDR_GOER_CFG_COS_OMEGA2_0_DATA_0      : INTEGER := 16#020#;
    constant ADDR_GOER_CFG_COS_OMEGA2_0_CTRL        : INTEGER := 16#024#;
    constant ADDR_GOER_CFG_COS_OMEGA2_1_DATA_0      : INTEGER := 16#028#;
    constant ADDR_GOER_CFG_COS_OMEGA2_1_CTRL        : INTEGER := 16#02c#;
    constant ADDR_GOER_CFG_SIN_OMEGA_0_DATA_0       : INTEGER := 16#030#;
    constant ADDR_GOER_CFG_SIN_OMEGA_0_CTRL         : INTEGER := 16#034#;
    constant ADDR_GOER_CFG_SIN_OMEGA_1_DATA_0       : INTEGER := 16#038#;
    constant ADDR_GOER_CFG_SIN_OMEGA_1_CTRL         : INTEGER := 16#03c#;
    constant ADDR_DEBL_CFG_DATA_0                   : INTEGER := 16#040#;
    constant ADDR_DEBL_CFG_CTRL                     : INTEGER := 16#044#;
    constant ADDR_STATUS_GZ_SAMPLES_IN_DATA_0       : INTEGER := 16#048#;
    constant ADDR_STATUS_GZ_SAMPLES_IN_CTRL         : INTEGER := 16#04c#;
    constant ADDR_STATUS_GZ_SAMPLE_WIN_DATA_0       : INTEGER := 16#058#;
    constant ADDR_STATUS_GZ_SAMPLE_WIN_CTRL         : INTEGER := 16#05c#;
    constant ADDR_STATUS_GZ_SAMPLES_OUT_DATA_0      : INTEGER := 16#068#;
    constant ADDR_STATUS_GZ_SAMPLES_OUT_CTRL        : INTEGER := 16#06c#;
    constant ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_DATA_0 : INTEGER := 16#078#;
    constant ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_CTRL   : INTEGER := 16#07c#;
    constant ADDR_STATUS_CC_STATE_DATA_0            : INTEGER := 16#088#;
    constant ADDR_STATUS_CC_STATE_CTRL              : INTEGER := 16#08c#;
    constant ADDR_STATUS_CC_SAMPLES_IN_DATA_0       : INTEGER := 16#098#;
    constant ADDR_STATUS_CC_SAMPLES_IN_CTRL         : INTEGER := 16#09c#;
    constant ADDR_STATUS_CC_SAMPLES_OUT_DATA_0      : INTEGER := 16#0a8#;
    constant ADDR_STATUS_CC_SAMPLES_OUT_CTRL        : INTEGER := 16#0ac#;
    constant ADDR_STATUS_CC_SAMPLE_IDX_DATA_0       : INTEGER := 16#0b8#;
    constant ADDR_STATUS_CC_SAMPLE_IDX_CTRL         : INTEGER := 16#0bc#;
    constant ADDR_STATUS_CC_CURRENT_NORM_DATA_0     : INTEGER := 16#0c8#;
    constant ADDR_STATUS_CC_CURRENT_NORM_CTRL       : INTEGER := 16#0cc#;
    constant ADDR_STATUS_CC_NORMS_WRITTEN_DATA_0    : INTEGER := 16#0d8#;
    constant ADDR_STATUS_CC_NORMS_WRITTEN_CTRL      : INTEGER := 16#0dc#;
    constant ADDR_STATUS_CC_OUT_FIFO_DATA_0         : INTEGER := 16#0e8#;
    constant ADDR_STATUS_CC_OUT_FIFO_CTRL           : INTEGER := 16#0ec#;
    constant ADDR_STATUS_CC_NORMS_FIFO_DATA_0       : INTEGER := 16#0f8#;
    constant ADDR_STATUS_CC_NORMS_FIFO_CTRL         : INTEGER := 16#0fc#;
    constant ADDR_STATUS_BP_CONFIG_LOADED_DATA_0    : INTEGER := 16#108#;
    constant ADDR_STATUS_BP_CONFIG_LOADED_CTRL      : INTEGER := 16#10c#;
    constant ADDR_STATUS_BP_FSM_STATE_DATA_0        : INTEGER := 16#118#;
    constant ADDR_STATUS_BP_FSM_STATE_CTRL          : INTEGER := 16#11c#;
    constant ADDR_STATUS_BP_PARAM_STATE_DATA_0      : INTEGER := 16#128#;
    constant ADDR_STATUS_BP_PARAM_STATE_CTRL        : INTEGER := 16#12c#;
    constant ADDR_STATUS_BP_IDX_DATA_0              : INTEGER := 16#138#;
    constant ADDR_STATUS_BP_IDX_CTRL                : INTEGER := 16#13c#;
    constant ADDR_STATUS_BP_SIGMAS_IN_DATA_0        : INTEGER := 16#148#;
    constant ADDR_STATUS_BP_SIGMAS_IN_CTRL          : INTEGER := 16#14c#;
    constant ADDR_STATUS_BP_PIXELS_OUT_DATA_0       : INTEGER := 16#158#;
    constant ADDR_STATUS_BP_PIXELS_OUT_CTRL         : INTEGER := 16#15c#;
    constant ADDR_STATUS_BP_OUT_FIFO_LEVEL_DATA_0   : INTEGER := 16#168#;
    constant ADDR_STATUS_BP_OUT_FIFO_LEVEL_CTRL     : INTEGER := 16#16c#;
    constant ADDR_STATUS_DB_CONFIG_LOADED_DATA_0    : INTEGER := 16#178#;
    constant ADDR_STATUS_DB_CONFIG_LOADED_CTRL      : INTEGER := 16#17c#;
    constant ADDR_STATUS_DB_FSM_STATE_DATA_0        : INTEGER := 16#188#;
    constant ADDR_STATUS_DB_FSM_STATE_CTRL          : INTEGER := 16#18c#;
    constant ADDR_STATUS_DB_PARAM_STATE_DATA_0      : INTEGER := 16#198#;
    constant ADDR_STATUS_DB_PARAM_STATE_CTRL        : INTEGER := 16#19c#;
    constant ADDR_STATUS_DB_IDX_DATA_0              : INTEGER := 16#1a8#;
    constant ADDR_STATUS_DB_IDX_CTRL                : INTEGER := 16#1ac#;
    constant ADDR_STATUS_DB_PIXELS_IN_DATA_0        : INTEGER := 16#1b8#;
    constant ADDR_STATUS_DB_PIXELS_IN_CTRL          : INTEGER := 16#1bc#;
    constant ADDR_STATUS_DB_PIXELS_OUT_DATA_0       : INTEGER := 16#1c8#;
    constant ADDR_STATUS_DB_PIXELS_OUT_CTRL         : INTEGER := 16#1cc#;
    constant ADDR_BITS         : INTEGER := 9;

    signal AWREADY_t           : STD_LOGIC;
    signal WREADY_t            : STD_LOGIC;
    signal ARREADY_t           : STD_LOGIC;
    signal RVALID_t            : STD_LOGIC;
    signal BVALID_t            : STD_LOGIC;
    signal waddr               : UNSIGNED(ADDR_BITS-1 downto 0);
    signal wmask               : UNSIGNED(C_S_AXI_DATA_WIDTH-1 downto 0);
    signal aw_hs               : STD_LOGIC;
    signal w_hs                : STD_LOGIC;
    signal rdata_data          : UNSIGNED(C_S_AXI_DATA_WIDTH-1 downto 0);
    signal ar_hs               : STD_LOGIC;
    signal raddr               : UNSIGNED(ADDR_BITS-1 downto 0);
    -- internal registers
    signal int_goer_cfg_COS_OMEGA_0 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_goer_cfg_COS_OMEGA_1 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_goer_cfg_COS_OMEGA2_0 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_goer_cfg_COS_OMEGA2_1 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_goer_cfg_SIN_OMEGA_0 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_goer_cfg_SIN_OMEGA_1 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg        : UNSIGNED(7 downto 0) := (others => '0');
    signal int_status_gz_samples_in_ap_vld : STD_LOGIC;
    signal int_status_gz_samples_in : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_gz_sample_win_ap_vld : STD_LOGIC;
    signal int_status_gz_sample_win : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_gz_samples_out_ap_vld : STD_LOGIC;
    signal int_status_gz_samples_out : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_gz_samples_out_fifo_ap_vld : STD_LOGIC;
    signal int_status_gz_samples_out_fifo : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_cc_state_ap_vld : STD_LOGIC;
    signal int_status_cc_state : UNSIGNED(7 downto 0) := (others => '0');
    signal int_status_cc_samples_in_ap_vld : STD_LOGIC;
    signal int_status_cc_samples_in : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_cc_samples_out_ap_vld : STD_LOGIC;
    signal int_status_cc_samples_out : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_cc_sample_idx_ap_vld : STD_LOGIC;
    signal int_status_cc_sample_idx : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_cc_current_norm_ap_vld : STD_LOGIC;
    signal int_status_cc_current_norm : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_cc_norms_written_ap_vld : STD_LOGIC;
    signal int_status_cc_norms_written : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_cc_out_fifo_ap_vld : STD_LOGIC;
    signal int_status_cc_out_fifo : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_cc_norms_fifo_ap_vld : STD_LOGIC;
    signal int_status_cc_norms_fifo : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_bp_config_loaded_ap_vld : STD_LOGIC;
    signal int_status_bp_config_loaded : UNSIGNED(0 downto 0) := (others => '0');
    signal int_status_bp_fsm_state_ap_vld : STD_LOGIC;
    signal int_status_bp_fsm_state : UNSIGNED(7 downto 0) := (others => '0');
    signal int_status_bp_param_state_ap_vld : STD_LOGIC;
    signal int_status_bp_param_state : UNSIGNED(7 downto 0) := (others => '0');
    signal int_status_bp_idx_ap_vld : STD_LOGIC;
    signal int_status_bp_idx   : UNSIGNED(15 downto 0) := (others => '0');
    signal int_status_bp_sigmas_in_ap_vld : STD_LOGIC;
    signal int_status_bp_sigmas_in : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_bp_pixels_out_ap_vld : STD_LOGIC;
    signal int_status_bp_pixels_out : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_bp_out_fifo_level_ap_vld : STD_LOGIC;
    signal int_status_bp_out_fifo_level : UNSIGNED(15 downto 0) := (others => '0');
    signal int_status_db_config_loaded_ap_vld : STD_LOGIC;
    signal int_status_db_config_loaded : UNSIGNED(0 downto 0) := (others => '0');
    signal int_status_db_fsm_state_ap_vld : STD_LOGIC;
    signal int_status_db_fsm_state : UNSIGNED(7 downto 0) := (others => '0');
    signal int_status_db_param_state_ap_vld : STD_LOGIC;
    signal int_status_db_param_state : UNSIGNED(7 downto 0) := (others => '0');
    signal int_status_db_idx_ap_vld : STD_LOGIC;
    signal int_status_db_idx   : UNSIGNED(15 downto 0) := (others => '0');
    signal int_status_db_pixels_in_ap_vld : STD_LOGIC;
    signal int_status_db_pixels_in : UNSIGNED(31 downto 0) := (others => '0');
    signal int_status_db_pixels_out_ap_vld : STD_LOGIC;
    signal int_status_db_pixels_out : UNSIGNED(31 downto 0) := (others => '0');


begin
-- ----------------------- Instantiation------------------


-- ----------------------- AXI WRITE ---------------------
    AWREADY_t <=  '1' when wstate = wridle else '0';
    AWREADY   <=  AWREADY_t;
    WREADY_t  <=  '1' when wstate = wrdata else '0';
    WREADY    <=  WREADY_t;
    BVALID_t  <=  '1' when wstate = wrresp else '0';
    BVALID    <=  BVALID_t;
    BRESP     <=  "00";  -- OKAY
    wmask     <=  (31 downto 24 => WSTRB(3), 23 downto 16 => WSTRB(2), 15 downto 8 => WSTRB(1), 7 downto 0 => WSTRB(0));
    aw_hs     <=  AWVALID and AWREADY_t;
    w_hs      <=  WVALID and WREADY_t;

    -- write FSM
    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                wstate <= wrreset;
            elsif (ACLK_EN = '1') then
                wstate <= wnext;
            end if;
        end if;
    end process;

    process (wstate, AWVALID, WVALID, BREADY, BVALID_t)
    begin
        case (wstate) is
        when wridle =>
            if (AWVALID = '1') then
                wnext <= wrdata;
            else
                wnext <= wridle;
            end if;
        when wrdata =>
            if (WVALID = '1') then
                wnext <= wrresp;
            else
                wnext <= wrdata;
            end if;
        when wrresp =>
            if (BREADY = '1' and BVALID_t = '1') then
                wnext <= wridle;
            else
                wnext <= wrresp;
            end if;
        when others =>
            wnext <= wridle;
        end case;
    end process;

    waddr_proc : process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (aw_hs = '1') then
                    waddr <= UNSIGNED(AWADDR(ADDR_BITS-1 downto 2) & (1 downto 0 => '0'));
                end if;
            end if;
        end if;
    end process;

-- ----------------------- AXI READ ----------------------
    ARREADY_t <= '1' when (rstate = rdidle) else '0';
    ARREADY <= ARREADY_t;
    RDATA   <= STD_LOGIC_VECTOR(rdata_data);
    RRESP   <= "00";  -- OKAY
    RVALID_t  <= '1' when (rstate = rddata) else '0';
    RVALID    <= RVALID_t;
    ar_hs   <= ARVALID and ARREADY_t;
    raddr   <= UNSIGNED(ARADDR(ADDR_BITS-1 downto 0));

    -- read FSM
    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                rstate <= rdreset;
            elsif (ACLK_EN = '1') then
                rstate <= rnext;
            end if;
        end if;
    end process;

    process (rstate, ARVALID, RREADY, RVALID_t)
    begin
        case (rstate) is
        when rdidle =>
            if (ARVALID = '1') then
                rnext <= rddata;
            else
                rnext <= rdidle;
            end if;
        when rddata =>
            if (RREADY = '1' and RVALID_t = '1') then
                rnext <= rdidle;
            else
                rnext <= rddata;
            end if;
        when others =>
            rnext <= rdidle;
        end case;
    end process;

    rdata_proc : process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ACLK_EN = '1') then
                if (ar_hs = '1') then
                    rdata_data <= (others => '0');
                    case (TO_INTEGER(raddr)) is
                    when ADDR_GOER_CFG_COS_OMEGA_0_DATA_0 =>
                        rdata_data <= RESIZE(int_goer_cfg_COS_OMEGA_0(17 downto 0), 32);
                    when ADDR_GOER_CFG_COS_OMEGA_1_DATA_0 =>
                        rdata_data <= RESIZE(int_goer_cfg_COS_OMEGA_1(17 downto 0), 32);
                    when ADDR_GOER_CFG_COS_OMEGA2_0_DATA_0 =>
                        rdata_data <= RESIZE(int_goer_cfg_COS_OMEGA2_0(17 downto 0), 32);
                    when ADDR_GOER_CFG_COS_OMEGA2_1_DATA_0 =>
                        rdata_data <= RESIZE(int_goer_cfg_COS_OMEGA2_1(17 downto 0), 32);
                    when ADDR_GOER_CFG_SIN_OMEGA_0_DATA_0 =>
                        rdata_data <= RESIZE(int_goer_cfg_SIN_OMEGA_0(17 downto 0), 32);
                    when ADDR_GOER_CFG_SIN_OMEGA_1_DATA_0 =>
                        rdata_data <= RESIZE(int_goer_cfg_SIN_OMEGA_1(17 downto 0), 32);
                    when ADDR_DEBL_CFG_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg(7 downto 0), 32);
                    when ADDR_STATUS_GZ_SAMPLES_IN_DATA_0 =>
                        rdata_data <= RESIZE(int_status_gz_samples_in(31 downto 0), 32);
                    when ADDR_STATUS_GZ_SAMPLES_IN_CTRL =>
                        rdata_data(0) <= int_status_gz_samples_in_ap_vld;
                    when ADDR_STATUS_GZ_SAMPLE_WIN_DATA_0 =>
                        rdata_data <= RESIZE(int_status_gz_sample_win(31 downto 0), 32);
                    when ADDR_STATUS_GZ_SAMPLE_WIN_CTRL =>
                        rdata_data(0) <= int_status_gz_sample_win_ap_vld;
                    when ADDR_STATUS_GZ_SAMPLES_OUT_DATA_0 =>
                        rdata_data <= RESIZE(int_status_gz_samples_out(31 downto 0), 32);
                    when ADDR_STATUS_GZ_SAMPLES_OUT_CTRL =>
                        rdata_data(0) <= int_status_gz_samples_out_ap_vld;
                    when ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_DATA_0 =>
                        rdata_data <= RESIZE(int_status_gz_samples_out_fifo(31 downto 0), 32);
                    when ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_CTRL =>
                        rdata_data(0) <= int_status_gz_samples_out_fifo_ap_vld;
                    when ADDR_STATUS_CC_STATE_DATA_0 =>
                        rdata_data <= RESIZE(int_status_cc_state(7 downto 0), 32);
                    when ADDR_STATUS_CC_STATE_CTRL =>
                        rdata_data(0) <= int_status_cc_state_ap_vld;
                    when ADDR_STATUS_CC_SAMPLES_IN_DATA_0 =>
                        rdata_data <= RESIZE(int_status_cc_samples_in(31 downto 0), 32);
                    when ADDR_STATUS_CC_SAMPLES_IN_CTRL =>
                        rdata_data(0) <= int_status_cc_samples_in_ap_vld;
                    when ADDR_STATUS_CC_SAMPLES_OUT_DATA_0 =>
                        rdata_data <= RESIZE(int_status_cc_samples_out(31 downto 0), 32);
                    when ADDR_STATUS_CC_SAMPLES_OUT_CTRL =>
                        rdata_data(0) <= int_status_cc_samples_out_ap_vld;
                    when ADDR_STATUS_CC_SAMPLE_IDX_DATA_0 =>
                        rdata_data <= RESIZE(int_status_cc_sample_idx(31 downto 0), 32);
                    when ADDR_STATUS_CC_SAMPLE_IDX_CTRL =>
                        rdata_data(0) <= int_status_cc_sample_idx_ap_vld;
                    when ADDR_STATUS_CC_CURRENT_NORM_DATA_0 =>
                        rdata_data <= RESIZE(int_status_cc_current_norm(31 downto 0), 32);
                    when ADDR_STATUS_CC_CURRENT_NORM_CTRL =>
                        rdata_data(0) <= int_status_cc_current_norm_ap_vld;
                    when ADDR_STATUS_CC_NORMS_WRITTEN_DATA_0 =>
                        rdata_data <= RESIZE(int_status_cc_norms_written(31 downto 0), 32);
                    when ADDR_STATUS_CC_NORMS_WRITTEN_CTRL =>
                        rdata_data(0) <= int_status_cc_norms_written_ap_vld;
                    when ADDR_STATUS_CC_OUT_FIFO_DATA_0 =>
                        rdata_data <= RESIZE(int_status_cc_out_fifo(31 downto 0), 32);
                    when ADDR_STATUS_CC_OUT_FIFO_CTRL =>
                        rdata_data(0) <= int_status_cc_out_fifo_ap_vld;
                    when ADDR_STATUS_CC_NORMS_FIFO_DATA_0 =>
                        rdata_data <= RESIZE(int_status_cc_norms_fifo(31 downto 0), 32);
                    when ADDR_STATUS_CC_NORMS_FIFO_CTRL =>
                        rdata_data(0) <= int_status_cc_norms_fifo_ap_vld;
                    when ADDR_STATUS_BP_CONFIG_LOADED_DATA_0 =>
                        rdata_data <= RESIZE(int_status_bp_config_loaded(0 downto 0), 32);
                    when ADDR_STATUS_BP_CONFIG_LOADED_CTRL =>
                        rdata_data(0) <= int_status_bp_config_loaded_ap_vld;
                    when ADDR_STATUS_BP_FSM_STATE_DATA_0 =>
                        rdata_data <= RESIZE(int_status_bp_fsm_state(7 downto 0), 32);
                    when ADDR_STATUS_BP_FSM_STATE_CTRL =>
                        rdata_data(0) <= int_status_bp_fsm_state_ap_vld;
                    when ADDR_STATUS_BP_PARAM_STATE_DATA_0 =>
                        rdata_data <= RESIZE(int_status_bp_param_state(7 downto 0), 32);
                    when ADDR_STATUS_BP_PARAM_STATE_CTRL =>
                        rdata_data(0) <= int_status_bp_param_state_ap_vld;
                    when ADDR_STATUS_BP_IDX_DATA_0 =>
                        rdata_data <= RESIZE(int_status_bp_idx(15 downto 0), 32);
                    when ADDR_STATUS_BP_IDX_CTRL =>
                        rdata_data(0) <= int_status_bp_idx_ap_vld;
                    when ADDR_STATUS_BP_SIGMAS_IN_DATA_0 =>
                        rdata_data <= RESIZE(int_status_bp_sigmas_in(31 downto 0), 32);
                    when ADDR_STATUS_BP_SIGMAS_IN_CTRL =>
                        rdata_data(0) <= int_status_bp_sigmas_in_ap_vld;
                    when ADDR_STATUS_BP_PIXELS_OUT_DATA_0 =>
                        rdata_data <= RESIZE(int_status_bp_pixels_out(31 downto 0), 32);
                    when ADDR_STATUS_BP_PIXELS_OUT_CTRL =>
                        rdata_data(0) <= int_status_bp_pixels_out_ap_vld;
                    when ADDR_STATUS_BP_OUT_FIFO_LEVEL_DATA_0 =>
                        rdata_data <= RESIZE(int_status_bp_out_fifo_level(15 downto 0), 32);
                    when ADDR_STATUS_BP_OUT_FIFO_LEVEL_CTRL =>
                        rdata_data(0) <= int_status_bp_out_fifo_level_ap_vld;
                    when ADDR_STATUS_DB_CONFIG_LOADED_DATA_0 =>
                        rdata_data <= RESIZE(int_status_db_config_loaded(0 downto 0), 32);
                    when ADDR_STATUS_DB_CONFIG_LOADED_CTRL =>
                        rdata_data(0) <= int_status_db_config_loaded_ap_vld;
                    when ADDR_STATUS_DB_FSM_STATE_DATA_0 =>
                        rdata_data <= RESIZE(int_status_db_fsm_state(7 downto 0), 32);
                    when ADDR_STATUS_DB_FSM_STATE_CTRL =>
                        rdata_data(0) <= int_status_db_fsm_state_ap_vld;
                    when ADDR_STATUS_DB_PARAM_STATE_DATA_0 =>
                        rdata_data <= RESIZE(int_status_db_param_state(7 downto 0), 32);
                    when ADDR_STATUS_DB_PARAM_STATE_CTRL =>
                        rdata_data(0) <= int_status_db_param_state_ap_vld;
                    when ADDR_STATUS_DB_IDX_DATA_0 =>
                        rdata_data <= RESIZE(int_status_db_idx(15 downto 0), 32);
                    when ADDR_STATUS_DB_IDX_CTRL =>
                        rdata_data(0) <= int_status_db_idx_ap_vld;
                    when ADDR_STATUS_DB_PIXELS_IN_DATA_0 =>
                        rdata_data <= RESIZE(int_status_db_pixels_in(31 downto 0), 32);
                    when ADDR_STATUS_DB_PIXELS_IN_CTRL =>
                        rdata_data(0) <= int_status_db_pixels_in_ap_vld;
                    when ADDR_STATUS_DB_PIXELS_OUT_DATA_0 =>
                        rdata_data <= RESIZE(int_status_db_pixels_out(31 downto 0), 32);
                    when ADDR_STATUS_DB_PIXELS_OUT_CTRL =>
                        rdata_data(0) <= int_status_db_pixels_out_ap_vld;
                    when others =>
                        NULL;
                    end case;
                end if;
            end if;
        end if;
    end process;

-- ----------------------- Register logic ----------------
    goer_cfg_COS_OMEGA_0 <= STD_LOGIC_VECTOR(int_goer_cfg_COS_OMEGA_0);
    goer_cfg_COS_OMEGA_1 <= STD_LOGIC_VECTOR(int_goer_cfg_COS_OMEGA_1);
    goer_cfg_COS_OMEGA2_0 <= STD_LOGIC_VECTOR(int_goer_cfg_COS_OMEGA2_0);
    goer_cfg_COS_OMEGA2_1 <= STD_LOGIC_VECTOR(int_goer_cfg_COS_OMEGA2_1);
    goer_cfg_SIN_OMEGA_0 <= STD_LOGIC_VECTOR(int_goer_cfg_SIN_OMEGA_0);
    goer_cfg_SIN_OMEGA_1 <= STD_LOGIC_VECTOR(int_goer_cfg_SIN_OMEGA_1);
    debl_cfg             <= STD_LOGIC_VECTOR(int_debl_cfg);

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_goer_cfg_COS_OMEGA_0(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_GOER_CFG_COS_OMEGA_0_DATA_0) then
                    int_goer_cfg_COS_OMEGA_0(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_goer_cfg_COS_OMEGA_0(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_goer_cfg_COS_OMEGA_1(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_GOER_CFG_COS_OMEGA_1_DATA_0) then
                    int_goer_cfg_COS_OMEGA_1(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_goer_cfg_COS_OMEGA_1(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_goer_cfg_COS_OMEGA2_0(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_GOER_CFG_COS_OMEGA2_0_DATA_0) then
                    int_goer_cfg_COS_OMEGA2_0(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_goer_cfg_COS_OMEGA2_0(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_goer_cfg_COS_OMEGA2_1(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_GOER_CFG_COS_OMEGA2_1_DATA_0) then
                    int_goer_cfg_COS_OMEGA2_1(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_goer_cfg_COS_OMEGA2_1(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_goer_cfg_SIN_OMEGA_0(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_GOER_CFG_SIN_OMEGA_0_DATA_0) then
                    int_goer_cfg_SIN_OMEGA_0(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_goer_cfg_SIN_OMEGA_0(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_goer_cfg_SIN_OMEGA_1(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_GOER_CFG_SIN_OMEGA_1_DATA_0) then
                    int_goer_cfg_SIN_OMEGA_1(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_goer_cfg_SIN_OMEGA_1(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg(7 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_DATA_0) then
                    int_debl_cfg(7 downto 0) <= (UNSIGNED(WDATA(7 downto 0)) and wmask(7 downto 0)) or ((not wmask(7 downto 0)) and int_debl_cfg(7 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_gz_samples_in <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_gz_samples_in_ap_vld = '1') then
                    int_status_gz_samples_in <= UNSIGNED(status_gz_samples_in);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_gz_samples_in_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_gz_samples_in_ap_vld = '1') then
                    int_status_gz_samples_in_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_GZ_SAMPLES_IN_CTRL) then
                    int_status_gz_samples_in_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_gz_sample_win <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_gz_sample_win_ap_vld = '1') then
                    int_status_gz_sample_win <= UNSIGNED(status_gz_sample_win);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_gz_sample_win_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_gz_sample_win_ap_vld = '1') then
                    int_status_gz_sample_win_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_GZ_SAMPLE_WIN_CTRL) then
                    int_status_gz_sample_win_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_gz_samples_out <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_gz_samples_out_ap_vld = '1') then
                    int_status_gz_samples_out <= UNSIGNED(status_gz_samples_out);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_gz_samples_out_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_gz_samples_out_ap_vld = '1') then
                    int_status_gz_samples_out_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_GZ_SAMPLES_OUT_CTRL) then
                    int_status_gz_samples_out_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_gz_samples_out_fifo <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_gz_samples_out_fifo_ap_vld = '1') then
                    int_status_gz_samples_out_fifo <= UNSIGNED(status_gz_samples_out_fifo);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_gz_samples_out_fifo_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_gz_samples_out_fifo_ap_vld = '1') then
                    int_status_gz_samples_out_fifo_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_GZ_SAMPLES_OUT_FIFO_CTRL) then
                    int_status_gz_samples_out_fifo_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_state <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_cc_state_ap_vld = '1') then
                    int_status_cc_state <= UNSIGNED(status_cc_state);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_state_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_cc_state_ap_vld = '1') then
                    int_status_cc_state_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_CC_STATE_CTRL) then
                    int_status_cc_state_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_samples_in <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_cc_samples_in_ap_vld = '1') then
                    int_status_cc_samples_in <= UNSIGNED(status_cc_samples_in);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_samples_in_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_cc_samples_in_ap_vld = '1') then
                    int_status_cc_samples_in_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_CC_SAMPLES_IN_CTRL) then
                    int_status_cc_samples_in_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_samples_out <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_cc_samples_out_ap_vld = '1') then
                    int_status_cc_samples_out <= UNSIGNED(status_cc_samples_out);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_samples_out_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_cc_samples_out_ap_vld = '1') then
                    int_status_cc_samples_out_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_CC_SAMPLES_OUT_CTRL) then
                    int_status_cc_samples_out_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_sample_idx <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_cc_sample_idx_ap_vld = '1') then
                    int_status_cc_sample_idx <= UNSIGNED(status_cc_sample_idx);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_sample_idx_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_cc_sample_idx_ap_vld = '1') then
                    int_status_cc_sample_idx_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_CC_SAMPLE_IDX_CTRL) then
                    int_status_cc_sample_idx_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_current_norm <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_cc_current_norm_ap_vld = '1') then
                    int_status_cc_current_norm <= UNSIGNED(status_cc_current_norm);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_current_norm_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_cc_current_norm_ap_vld = '1') then
                    int_status_cc_current_norm_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_CC_CURRENT_NORM_CTRL) then
                    int_status_cc_current_norm_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_norms_written <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_cc_norms_written_ap_vld = '1') then
                    int_status_cc_norms_written <= UNSIGNED(status_cc_norms_written);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_norms_written_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_cc_norms_written_ap_vld = '1') then
                    int_status_cc_norms_written_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_CC_NORMS_WRITTEN_CTRL) then
                    int_status_cc_norms_written_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_out_fifo <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_cc_out_fifo_ap_vld = '1') then
                    int_status_cc_out_fifo <= UNSIGNED(status_cc_out_fifo);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_out_fifo_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_cc_out_fifo_ap_vld = '1') then
                    int_status_cc_out_fifo_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_CC_OUT_FIFO_CTRL) then
                    int_status_cc_out_fifo_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_norms_fifo <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_cc_norms_fifo_ap_vld = '1') then
                    int_status_cc_norms_fifo <= UNSIGNED(status_cc_norms_fifo);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_cc_norms_fifo_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_cc_norms_fifo_ap_vld = '1') then
                    int_status_cc_norms_fifo_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_CC_NORMS_FIFO_CTRL) then
                    int_status_cc_norms_fifo_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_config_loaded <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_bp_config_loaded_ap_vld = '1') then
                    int_status_bp_config_loaded <= UNSIGNED(status_bp_config_loaded);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_config_loaded_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_bp_config_loaded_ap_vld = '1') then
                    int_status_bp_config_loaded_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_BP_CONFIG_LOADED_CTRL) then
                    int_status_bp_config_loaded_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_fsm_state <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_bp_fsm_state_ap_vld = '1') then
                    int_status_bp_fsm_state <= UNSIGNED(status_bp_fsm_state);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_fsm_state_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_bp_fsm_state_ap_vld = '1') then
                    int_status_bp_fsm_state_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_BP_FSM_STATE_CTRL) then
                    int_status_bp_fsm_state_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_param_state <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_bp_param_state_ap_vld = '1') then
                    int_status_bp_param_state <= UNSIGNED(status_bp_param_state);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_param_state_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_bp_param_state_ap_vld = '1') then
                    int_status_bp_param_state_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_BP_PARAM_STATE_CTRL) then
                    int_status_bp_param_state_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_idx <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_bp_idx_ap_vld = '1') then
                    int_status_bp_idx <= UNSIGNED(status_bp_idx);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_idx_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_bp_idx_ap_vld = '1') then
                    int_status_bp_idx_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_BP_IDX_CTRL) then
                    int_status_bp_idx_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_sigmas_in <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_bp_sigmas_in_ap_vld = '1') then
                    int_status_bp_sigmas_in <= UNSIGNED(status_bp_sigmas_in);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_sigmas_in_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_bp_sigmas_in_ap_vld = '1') then
                    int_status_bp_sigmas_in_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_BP_SIGMAS_IN_CTRL) then
                    int_status_bp_sigmas_in_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_pixels_out <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_bp_pixels_out_ap_vld = '1') then
                    int_status_bp_pixels_out <= UNSIGNED(status_bp_pixels_out);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_pixels_out_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_bp_pixels_out_ap_vld = '1') then
                    int_status_bp_pixels_out_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_BP_PIXELS_OUT_CTRL) then
                    int_status_bp_pixels_out_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_out_fifo_level <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_bp_out_fifo_level_ap_vld = '1') then
                    int_status_bp_out_fifo_level <= UNSIGNED(status_bp_out_fifo_level);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_bp_out_fifo_level_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_bp_out_fifo_level_ap_vld = '1') then
                    int_status_bp_out_fifo_level_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_BP_OUT_FIFO_LEVEL_CTRL) then
                    int_status_bp_out_fifo_level_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_config_loaded <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_db_config_loaded_ap_vld = '1') then
                    int_status_db_config_loaded <= UNSIGNED(status_db_config_loaded);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_config_loaded_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_db_config_loaded_ap_vld = '1') then
                    int_status_db_config_loaded_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_DB_CONFIG_LOADED_CTRL) then
                    int_status_db_config_loaded_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_fsm_state <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_db_fsm_state_ap_vld = '1') then
                    int_status_db_fsm_state <= UNSIGNED(status_db_fsm_state);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_fsm_state_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_db_fsm_state_ap_vld = '1') then
                    int_status_db_fsm_state_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_DB_FSM_STATE_CTRL) then
                    int_status_db_fsm_state_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_param_state <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_db_param_state_ap_vld = '1') then
                    int_status_db_param_state <= UNSIGNED(status_db_param_state);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_param_state_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_db_param_state_ap_vld = '1') then
                    int_status_db_param_state_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_DB_PARAM_STATE_CTRL) then
                    int_status_db_param_state_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_idx <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_db_idx_ap_vld = '1') then
                    int_status_db_idx <= UNSIGNED(status_db_idx);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_idx_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_db_idx_ap_vld = '1') then
                    int_status_db_idx_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_DB_IDX_CTRL) then
                    int_status_db_idx_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_pixels_in <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_db_pixels_in_ap_vld = '1') then
                    int_status_db_pixels_in <= UNSIGNED(status_db_pixels_in);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_pixels_in_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_db_pixels_in_ap_vld = '1') then
                    int_status_db_pixels_in_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_DB_PIXELS_IN_CTRL) then
                    int_status_db_pixels_in_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_pixels_out <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (status_db_pixels_out_ap_vld = '1') then
                    int_status_db_pixels_out <= UNSIGNED(status_db_pixels_out);
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_status_db_pixels_out_ap_vld <= '0';
            elsif (ACLK_EN = '1') then
                if (status_db_pixels_out_ap_vld = '1') then
                    int_status_db_pixels_out_ap_vld <= '1';
                elsif (ar_hs = '1' and raddr = ADDR_STATUS_DB_PIXELS_OUT_CTRL) then
                    int_status_db_pixels_out_ap_vld <= '0'; -- clear on read
                end if;
            end if;
        end if;
    end process;


-- ----------------------- Memory logic ------------------

end architecture behave;
