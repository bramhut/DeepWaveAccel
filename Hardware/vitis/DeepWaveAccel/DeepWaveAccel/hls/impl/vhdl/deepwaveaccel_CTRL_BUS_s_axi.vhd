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
    debl_cfg_n_layers     :out  STD_LOGIC_VECTOR(7 downto 0);
    debl_cfg_K            :out  STD_LOGIC_VECTOR(5 downto 0);
    debl_cfg_lap_off_0    :out  STD_LOGIC_VECTOR(11 downto 0);
    debl_cfg_lap_off_1    :out  STD_LOGIC_VECTOR(11 downto 0);
    debl_cfg_lap_off_2    :out  STD_LOGIC_VECTOR(11 downto 0);
    debl_cfg_lap_off_3    :out  STD_LOGIC_VECTOR(11 downto 0);
    debl_cfg_lap_off_4    :out  STD_LOGIC_VECTOR(11 downto 0);
    debl_cfg_lap_off_5    :out  STD_LOGIC_VECTOR(11 downto 0);
    debl_cfg_theta_0      :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_1      :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_2      :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_3      :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_4      :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_5      :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_6      :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_7      :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_8      :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_9      :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_10     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_11     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_12     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_13     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_14     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_15     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_16     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_17     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_18     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_19     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_20     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_21     :out  STD_LOGIC_VECTOR(17 downto 0);
    debl_cfg_theta_22     :out  STD_LOGIC_VECTOR(17 downto 0)
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
-- 0x040 : Data signal of debl_cfg_n_layers
--         bit 7~0 - debl_cfg_n_layers[7:0] (Read/Write)
--         others  - reserved
-- 0x044 : reserved
-- 0x048 : Data signal of debl_cfg_K
--         bit 5~0 - debl_cfg_K[5:0] (Read/Write)
--         others  - reserved
-- 0x04c : reserved
-- 0x050 : Data signal of debl_cfg_lap_off_0
--         bit 11~0 - debl_cfg_lap_off_0[11:0] (Read/Write)
--         others   - reserved
-- 0x054 : reserved
-- 0x058 : Data signal of debl_cfg_lap_off_1
--         bit 11~0 - debl_cfg_lap_off_1[11:0] (Read/Write)
--         others   - reserved
-- 0x05c : reserved
-- 0x060 : Data signal of debl_cfg_lap_off_2
--         bit 11~0 - debl_cfg_lap_off_2[11:0] (Read/Write)
--         others   - reserved
-- 0x064 : reserved
-- 0x068 : Data signal of debl_cfg_lap_off_3
--         bit 11~0 - debl_cfg_lap_off_3[11:0] (Read/Write)
--         others   - reserved
-- 0x06c : reserved
-- 0x070 : Data signal of debl_cfg_lap_off_4
--         bit 11~0 - debl_cfg_lap_off_4[11:0] (Read/Write)
--         others   - reserved
-- 0x074 : reserved
-- 0x078 : Data signal of debl_cfg_lap_off_5
--         bit 11~0 - debl_cfg_lap_off_5[11:0] (Read/Write)
--         others   - reserved
-- 0x07c : reserved
-- 0x080 : Data signal of debl_cfg_theta_0
--         bit 17~0 - debl_cfg_theta_0[17:0] (Read/Write)
--         others   - reserved
-- 0x084 : reserved
-- 0x088 : Data signal of debl_cfg_theta_1
--         bit 17~0 - debl_cfg_theta_1[17:0] (Read/Write)
--         others   - reserved
-- 0x08c : reserved
-- 0x090 : Data signal of debl_cfg_theta_2
--         bit 17~0 - debl_cfg_theta_2[17:0] (Read/Write)
--         others   - reserved
-- 0x094 : reserved
-- 0x098 : Data signal of debl_cfg_theta_3
--         bit 17~0 - debl_cfg_theta_3[17:0] (Read/Write)
--         others   - reserved
-- 0x09c : reserved
-- 0x0a0 : Data signal of debl_cfg_theta_4
--         bit 17~0 - debl_cfg_theta_4[17:0] (Read/Write)
--         others   - reserved
-- 0x0a4 : reserved
-- 0x0a8 : Data signal of debl_cfg_theta_5
--         bit 17~0 - debl_cfg_theta_5[17:0] (Read/Write)
--         others   - reserved
-- 0x0ac : reserved
-- 0x0b0 : Data signal of debl_cfg_theta_6
--         bit 17~0 - debl_cfg_theta_6[17:0] (Read/Write)
--         others   - reserved
-- 0x0b4 : reserved
-- 0x0b8 : Data signal of debl_cfg_theta_7
--         bit 17~0 - debl_cfg_theta_7[17:0] (Read/Write)
--         others   - reserved
-- 0x0bc : reserved
-- 0x0c0 : Data signal of debl_cfg_theta_8
--         bit 17~0 - debl_cfg_theta_8[17:0] (Read/Write)
--         others   - reserved
-- 0x0c4 : reserved
-- 0x0c8 : Data signal of debl_cfg_theta_9
--         bit 17~0 - debl_cfg_theta_9[17:0] (Read/Write)
--         others   - reserved
-- 0x0cc : reserved
-- 0x0d0 : Data signal of debl_cfg_theta_10
--         bit 17~0 - debl_cfg_theta_10[17:0] (Read/Write)
--         others   - reserved
-- 0x0d4 : reserved
-- 0x0d8 : Data signal of debl_cfg_theta_11
--         bit 17~0 - debl_cfg_theta_11[17:0] (Read/Write)
--         others   - reserved
-- 0x0dc : reserved
-- 0x0e0 : Data signal of debl_cfg_theta_12
--         bit 17~0 - debl_cfg_theta_12[17:0] (Read/Write)
--         others   - reserved
-- 0x0e4 : reserved
-- 0x0e8 : Data signal of debl_cfg_theta_13
--         bit 17~0 - debl_cfg_theta_13[17:0] (Read/Write)
--         others   - reserved
-- 0x0ec : reserved
-- 0x0f0 : Data signal of debl_cfg_theta_14
--         bit 17~0 - debl_cfg_theta_14[17:0] (Read/Write)
--         others   - reserved
-- 0x0f4 : reserved
-- 0x0f8 : Data signal of debl_cfg_theta_15
--         bit 17~0 - debl_cfg_theta_15[17:0] (Read/Write)
--         others   - reserved
-- 0x0fc : reserved
-- 0x100 : Data signal of debl_cfg_theta_16
--         bit 17~0 - debl_cfg_theta_16[17:0] (Read/Write)
--         others   - reserved
-- 0x104 : reserved
-- 0x108 : Data signal of debl_cfg_theta_17
--         bit 17~0 - debl_cfg_theta_17[17:0] (Read/Write)
--         others   - reserved
-- 0x10c : reserved
-- 0x110 : Data signal of debl_cfg_theta_18
--         bit 17~0 - debl_cfg_theta_18[17:0] (Read/Write)
--         others   - reserved
-- 0x114 : reserved
-- 0x118 : Data signal of debl_cfg_theta_19
--         bit 17~0 - debl_cfg_theta_19[17:0] (Read/Write)
--         others   - reserved
-- 0x11c : reserved
-- 0x120 : Data signal of debl_cfg_theta_20
--         bit 17~0 - debl_cfg_theta_20[17:0] (Read/Write)
--         others   - reserved
-- 0x124 : reserved
-- 0x128 : Data signal of debl_cfg_theta_21
--         bit 17~0 - debl_cfg_theta_21[17:0] (Read/Write)
--         others   - reserved
-- 0x12c : reserved
-- 0x130 : Data signal of debl_cfg_theta_22
--         bit 17~0 - debl_cfg_theta_22[17:0] (Read/Write)
--         others   - reserved
-- 0x134 : reserved
-- (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

architecture behave of deepwaveaccel_CTRL_BUS_s_axi is
    type states is (wridle, wrdata, wrresp, wrreset, rdidle, rddata, rdreset);  -- read and write fsm states
    signal wstate  : states := wrreset;
    signal rstate  : states := rdreset;
    signal wnext, rnext: states;
    constant ADDR_GOER_CFG_COS_OMEGA_0_DATA_0  : INTEGER := 16#010#;
    constant ADDR_GOER_CFG_COS_OMEGA_0_CTRL    : INTEGER := 16#014#;
    constant ADDR_GOER_CFG_COS_OMEGA_1_DATA_0  : INTEGER := 16#018#;
    constant ADDR_GOER_CFG_COS_OMEGA_1_CTRL    : INTEGER := 16#01c#;
    constant ADDR_GOER_CFG_COS_OMEGA2_0_DATA_0 : INTEGER := 16#020#;
    constant ADDR_GOER_CFG_COS_OMEGA2_0_CTRL   : INTEGER := 16#024#;
    constant ADDR_GOER_CFG_COS_OMEGA2_1_DATA_0 : INTEGER := 16#028#;
    constant ADDR_GOER_CFG_COS_OMEGA2_1_CTRL   : INTEGER := 16#02c#;
    constant ADDR_GOER_CFG_SIN_OMEGA_0_DATA_0  : INTEGER := 16#030#;
    constant ADDR_GOER_CFG_SIN_OMEGA_0_CTRL    : INTEGER := 16#034#;
    constant ADDR_GOER_CFG_SIN_OMEGA_1_DATA_0  : INTEGER := 16#038#;
    constant ADDR_GOER_CFG_SIN_OMEGA_1_CTRL    : INTEGER := 16#03c#;
    constant ADDR_DEBL_CFG_N_LAYERS_DATA_0     : INTEGER := 16#040#;
    constant ADDR_DEBL_CFG_N_LAYERS_CTRL       : INTEGER := 16#044#;
    constant ADDR_DEBL_CFG_K_DATA_0            : INTEGER := 16#048#;
    constant ADDR_DEBL_CFG_K_CTRL              : INTEGER := 16#04c#;
    constant ADDR_DEBL_CFG_LAP_OFF_0_DATA_0    : INTEGER := 16#050#;
    constant ADDR_DEBL_CFG_LAP_OFF_0_CTRL      : INTEGER := 16#054#;
    constant ADDR_DEBL_CFG_LAP_OFF_1_DATA_0    : INTEGER := 16#058#;
    constant ADDR_DEBL_CFG_LAP_OFF_1_CTRL      : INTEGER := 16#05c#;
    constant ADDR_DEBL_CFG_LAP_OFF_2_DATA_0    : INTEGER := 16#060#;
    constant ADDR_DEBL_CFG_LAP_OFF_2_CTRL      : INTEGER := 16#064#;
    constant ADDR_DEBL_CFG_LAP_OFF_3_DATA_0    : INTEGER := 16#068#;
    constant ADDR_DEBL_CFG_LAP_OFF_3_CTRL      : INTEGER := 16#06c#;
    constant ADDR_DEBL_CFG_LAP_OFF_4_DATA_0    : INTEGER := 16#070#;
    constant ADDR_DEBL_CFG_LAP_OFF_4_CTRL      : INTEGER := 16#074#;
    constant ADDR_DEBL_CFG_LAP_OFF_5_DATA_0    : INTEGER := 16#078#;
    constant ADDR_DEBL_CFG_LAP_OFF_5_CTRL      : INTEGER := 16#07c#;
    constant ADDR_DEBL_CFG_THETA_0_DATA_0      : INTEGER := 16#080#;
    constant ADDR_DEBL_CFG_THETA_0_CTRL        : INTEGER := 16#084#;
    constant ADDR_DEBL_CFG_THETA_1_DATA_0      : INTEGER := 16#088#;
    constant ADDR_DEBL_CFG_THETA_1_CTRL        : INTEGER := 16#08c#;
    constant ADDR_DEBL_CFG_THETA_2_DATA_0      : INTEGER := 16#090#;
    constant ADDR_DEBL_CFG_THETA_2_CTRL        : INTEGER := 16#094#;
    constant ADDR_DEBL_CFG_THETA_3_DATA_0      : INTEGER := 16#098#;
    constant ADDR_DEBL_CFG_THETA_3_CTRL        : INTEGER := 16#09c#;
    constant ADDR_DEBL_CFG_THETA_4_DATA_0      : INTEGER := 16#0a0#;
    constant ADDR_DEBL_CFG_THETA_4_CTRL        : INTEGER := 16#0a4#;
    constant ADDR_DEBL_CFG_THETA_5_DATA_0      : INTEGER := 16#0a8#;
    constant ADDR_DEBL_CFG_THETA_5_CTRL        : INTEGER := 16#0ac#;
    constant ADDR_DEBL_CFG_THETA_6_DATA_0      : INTEGER := 16#0b0#;
    constant ADDR_DEBL_CFG_THETA_6_CTRL        : INTEGER := 16#0b4#;
    constant ADDR_DEBL_CFG_THETA_7_DATA_0      : INTEGER := 16#0b8#;
    constant ADDR_DEBL_CFG_THETA_7_CTRL        : INTEGER := 16#0bc#;
    constant ADDR_DEBL_CFG_THETA_8_DATA_0      : INTEGER := 16#0c0#;
    constant ADDR_DEBL_CFG_THETA_8_CTRL        : INTEGER := 16#0c4#;
    constant ADDR_DEBL_CFG_THETA_9_DATA_0      : INTEGER := 16#0c8#;
    constant ADDR_DEBL_CFG_THETA_9_CTRL        : INTEGER := 16#0cc#;
    constant ADDR_DEBL_CFG_THETA_10_DATA_0     : INTEGER := 16#0d0#;
    constant ADDR_DEBL_CFG_THETA_10_CTRL       : INTEGER := 16#0d4#;
    constant ADDR_DEBL_CFG_THETA_11_DATA_0     : INTEGER := 16#0d8#;
    constant ADDR_DEBL_CFG_THETA_11_CTRL       : INTEGER := 16#0dc#;
    constant ADDR_DEBL_CFG_THETA_12_DATA_0     : INTEGER := 16#0e0#;
    constant ADDR_DEBL_CFG_THETA_12_CTRL       : INTEGER := 16#0e4#;
    constant ADDR_DEBL_CFG_THETA_13_DATA_0     : INTEGER := 16#0e8#;
    constant ADDR_DEBL_CFG_THETA_13_CTRL       : INTEGER := 16#0ec#;
    constant ADDR_DEBL_CFG_THETA_14_DATA_0     : INTEGER := 16#0f0#;
    constant ADDR_DEBL_CFG_THETA_14_CTRL       : INTEGER := 16#0f4#;
    constant ADDR_DEBL_CFG_THETA_15_DATA_0     : INTEGER := 16#0f8#;
    constant ADDR_DEBL_CFG_THETA_15_CTRL       : INTEGER := 16#0fc#;
    constant ADDR_DEBL_CFG_THETA_16_DATA_0     : INTEGER := 16#100#;
    constant ADDR_DEBL_CFG_THETA_16_CTRL       : INTEGER := 16#104#;
    constant ADDR_DEBL_CFG_THETA_17_DATA_0     : INTEGER := 16#108#;
    constant ADDR_DEBL_CFG_THETA_17_CTRL       : INTEGER := 16#10c#;
    constant ADDR_DEBL_CFG_THETA_18_DATA_0     : INTEGER := 16#110#;
    constant ADDR_DEBL_CFG_THETA_18_CTRL       : INTEGER := 16#114#;
    constant ADDR_DEBL_CFG_THETA_19_DATA_0     : INTEGER := 16#118#;
    constant ADDR_DEBL_CFG_THETA_19_CTRL       : INTEGER := 16#11c#;
    constant ADDR_DEBL_CFG_THETA_20_DATA_0     : INTEGER := 16#120#;
    constant ADDR_DEBL_CFG_THETA_20_CTRL       : INTEGER := 16#124#;
    constant ADDR_DEBL_CFG_THETA_21_DATA_0     : INTEGER := 16#128#;
    constant ADDR_DEBL_CFG_THETA_21_CTRL       : INTEGER := 16#12c#;
    constant ADDR_DEBL_CFG_THETA_22_DATA_0     : INTEGER := 16#130#;
    constant ADDR_DEBL_CFG_THETA_22_CTRL       : INTEGER := 16#134#;
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
    signal int_debl_cfg_n_layers : UNSIGNED(7 downto 0) := (others => '0');
    signal int_debl_cfg_K      : UNSIGNED(5 downto 0) := (others => '0');
    signal int_debl_cfg_lap_off_0 : UNSIGNED(11 downto 0) := (others => '0');
    signal int_debl_cfg_lap_off_1 : UNSIGNED(11 downto 0) := (others => '0');
    signal int_debl_cfg_lap_off_2 : UNSIGNED(11 downto 0) := (others => '0');
    signal int_debl_cfg_lap_off_3 : UNSIGNED(11 downto 0) := (others => '0');
    signal int_debl_cfg_lap_off_4 : UNSIGNED(11 downto 0) := (others => '0');
    signal int_debl_cfg_lap_off_5 : UNSIGNED(11 downto 0) := (others => '0');
    signal int_debl_cfg_theta_0 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_1 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_2 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_3 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_4 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_5 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_6 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_7 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_8 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_9 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_10 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_11 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_12 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_13 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_14 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_15 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_16 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_17 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_18 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_19 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_20 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_21 : UNSIGNED(17 downto 0) := (others => '0');
    signal int_debl_cfg_theta_22 : UNSIGNED(17 downto 0) := (others => '0');


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
                    when ADDR_DEBL_CFG_N_LAYERS_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_n_layers(7 downto 0), 32);
                    when ADDR_DEBL_CFG_K_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_K(5 downto 0), 32);
                    when ADDR_DEBL_CFG_LAP_OFF_0_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_lap_off_0(11 downto 0), 32);
                    when ADDR_DEBL_CFG_LAP_OFF_1_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_lap_off_1(11 downto 0), 32);
                    when ADDR_DEBL_CFG_LAP_OFF_2_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_lap_off_2(11 downto 0), 32);
                    when ADDR_DEBL_CFG_LAP_OFF_3_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_lap_off_3(11 downto 0), 32);
                    when ADDR_DEBL_CFG_LAP_OFF_4_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_lap_off_4(11 downto 0), 32);
                    when ADDR_DEBL_CFG_LAP_OFF_5_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_lap_off_5(11 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_0_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_0(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_1_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_1(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_2_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_2(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_3_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_3(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_4_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_4(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_5_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_5(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_6_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_6(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_7_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_7(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_8_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_8(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_9_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_9(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_10_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_10(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_11_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_11(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_12_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_12(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_13_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_13(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_14_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_14(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_15_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_15(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_16_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_16(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_17_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_17(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_18_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_18(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_19_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_19(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_20_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_20(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_21_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_21(17 downto 0), 32);
                    when ADDR_DEBL_CFG_THETA_22_DATA_0 =>
                        rdata_data <= RESIZE(int_debl_cfg_theta_22(17 downto 0), 32);
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
    debl_cfg_n_layers    <= STD_LOGIC_VECTOR(int_debl_cfg_n_layers);
    debl_cfg_K           <= STD_LOGIC_VECTOR(int_debl_cfg_K);
    debl_cfg_lap_off_0   <= STD_LOGIC_VECTOR(int_debl_cfg_lap_off_0);
    debl_cfg_lap_off_1   <= STD_LOGIC_VECTOR(int_debl_cfg_lap_off_1);
    debl_cfg_lap_off_2   <= STD_LOGIC_VECTOR(int_debl_cfg_lap_off_2);
    debl_cfg_lap_off_3   <= STD_LOGIC_VECTOR(int_debl_cfg_lap_off_3);
    debl_cfg_lap_off_4   <= STD_LOGIC_VECTOR(int_debl_cfg_lap_off_4);
    debl_cfg_lap_off_5   <= STD_LOGIC_VECTOR(int_debl_cfg_lap_off_5);
    debl_cfg_theta_0     <= STD_LOGIC_VECTOR(int_debl_cfg_theta_0);
    debl_cfg_theta_1     <= STD_LOGIC_VECTOR(int_debl_cfg_theta_1);
    debl_cfg_theta_2     <= STD_LOGIC_VECTOR(int_debl_cfg_theta_2);
    debl_cfg_theta_3     <= STD_LOGIC_VECTOR(int_debl_cfg_theta_3);
    debl_cfg_theta_4     <= STD_LOGIC_VECTOR(int_debl_cfg_theta_4);
    debl_cfg_theta_5     <= STD_LOGIC_VECTOR(int_debl_cfg_theta_5);
    debl_cfg_theta_6     <= STD_LOGIC_VECTOR(int_debl_cfg_theta_6);
    debl_cfg_theta_7     <= STD_LOGIC_VECTOR(int_debl_cfg_theta_7);
    debl_cfg_theta_8     <= STD_LOGIC_VECTOR(int_debl_cfg_theta_8);
    debl_cfg_theta_9     <= STD_LOGIC_VECTOR(int_debl_cfg_theta_9);
    debl_cfg_theta_10    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_10);
    debl_cfg_theta_11    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_11);
    debl_cfg_theta_12    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_12);
    debl_cfg_theta_13    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_13);
    debl_cfg_theta_14    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_14);
    debl_cfg_theta_15    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_15);
    debl_cfg_theta_16    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_16);
    debl_cfg_theta_17    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_17);
    debl_cfg_theta_18    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_18);
    debl_cfg_theta_19    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_19);
    debl_cfg_theta_20    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_20);
    debl_cfg_theta_21    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_21);
    debl_cfg_theta_22    <= STD_LOGIC_VECTOR(int_debl_cfg_theta_22);

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
                int_debl_cfg_n_layers(7 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_N_LAYERS_DATA_0) then
                    int_debl_cfg_n_layers(7 downto 0) <= (UNSIGNED(WDATA(7 downto 0)) and wmask(7 downto 0)) or ((not wmask(7 downto 0)) and int_debl_cfg_n_layers(7 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_K(5 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_K_DATA_0) then
                    int_debl_cfg_K(5 downto 0) <= (UNSIGNED(WDATA(5 downto 0)) and wmask(5 downto 0)) or ((not wmask(5 downto 0)) and int_debl_cfg_K(5 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_lap_off_0(11 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_LAP_OFF_0_DATA_0) then
                    int_debl_cfg_lap_off_0(11 downto 0) <= (UNSIGNED(WDATA(11 downto 0)) and wmask(11 downto 0)) or ((not wmask(11 downto 0)) and int_debl_cfg_lap_off_0(11 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_lap_off_1(11 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_LAP_OFF_1_DATA_0) then
                    int_debl_cfg_lap_off_1(11 downto 0) <= (UNSIGNED(WDATA(11 downto 0)) and wmask(11 downto 0)) or ((not wmask(11 downto 0)) and int_debl_cfg_lap_off_1(11 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_lap_off_2(11 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_LAP_OFF_2_DATA_0) then
                    int_debl_cfg_lap_off_2(11 downto 0) <= (UNSIGNED(WDATA(11 downto 0)) and wmask(11 downto 0)) or ((not wmask(11 downto 0)) and int_debl_cfg_lap_off_2(11 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_lap_off_3(11 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_LAP_OFF_3_DATA_0) then
                    int_debl_cfg_lap_off_3(11 downto 0) <= (UNSIGNED(WDATA(11 downto 0)) and wmask(11 downto 0)) or ((not wmask(11 downto 0)) and int_debl_cfg_lap_off_3(11 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_lap_off_4(11 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_LAP_OFF_4_DATA_0) then
                    int_debl_cfg_lap_off_4(11 downto 0) <= (UNSIGNED(WDATA(11 downto 0)) and wmask(11 downto 0)) or ((not wmask(11 downto 0)) and int_debl_cfg_lap_off_4(11 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_lap_off_5(11 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_LAP_OFF_5_DATA_0) then
                    int_debl_cfg_lap_off_5(11 downto 0) <= (UNSIGNED(WDATA(11 downto 0)) and wmask(11 downto 0)) or ((not wmask(11 downto 0)) and int_debl_cfg_lap_off_5(11 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_0(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_0_DATA_0) then
                    int_debl_cfg_theta_0(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_0(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_1(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_1_DATA_0) then
                    int_debl_cfg_theta_1(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_1(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_2(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_2_DATA_0) then
                    int_debl_cfg_theta_2(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_2(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_3(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_3_DATA_0) then
                    int_debl_cfg_theta_3(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_3(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_4(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_4_DATA_0) then
                    int_debl_cfg_theta_4(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_4(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_5(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_5_DATA_0) then
                    int_debl_cfg_theta_5(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_5(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_6(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_6_DATA_0) then
                    int_debl_cfg_theta_6(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_6(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_7(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_7_DATA_0) then
                    int_debl_cfg_theta_7(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_7(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_8(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_8_DATA_0) then
                    int_debl_cfg_theta_8(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_8(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_9(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_9_DATA_0) then
                    int_debl_cfg_theta_9(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_9(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_10(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_10_DATA_0) then
                    int_debl_cfg_theta_10(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_10(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_11(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_11_DATA_0) then
                    int_debl_cfg_theta_11(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_11(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_12(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_12_DATA_0) then
                    int_debl_cfg_theta_12(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_12(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_13(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_13_DATA_0) then
                    int_debl_cfg_theta_13(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_13(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_14(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_14_DATA_0) then
                    int_debl_cfg_theta_14(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_14(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_15(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_15_DATA_0) then
                    int_debl_cfg_theta_15(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_15(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_16(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_16_DATA_0) then
                    int_debl_cfg_theta_16(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_16(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_17(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_17_DATA_0) then
                    int_debl_cfg_theta_17(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_17(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_18(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_18_DATA_0) then
                    int_debl_cfg_theta_18(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_18(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_19(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_19_DATA_0) then
                    int_debl_cfg_theta_19(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_19(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_20(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_20_DATA_0) then
                    int_debl_cfg_theta_20(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_20(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_21(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_21_DATA_0) then
                    int_debl_cfg_theta_21(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_21(17 downto 0));
                end if;
            end if;
        end if;
    end process;

    process (ACLK)
    begin
        if (ACLK'event and ACLK = '1') then
            if (ARESET = '1') then
                int_debl_cfg_theta_22(17 downto 0) <= (others => '0');
            elsif (ACLK_EN = '1') then
                if (w_hs = '1' and waddr = ADDR_DEBL_CFG_THETA_22_DATA_0) then
                    int_debl_cfg_theta_22(17 downto 0) <= (UNSIGNED(WDATA(17 downto 0)) and wmask(17 downto 0)) or ((not wmask(17 downto 0)) and int_debl_cfg_theta_22(17 downto 0));
                end if;
            end if;
        end if;
    end process;


-- ----------------------- Memory logic ------------------

end architecture behave;
