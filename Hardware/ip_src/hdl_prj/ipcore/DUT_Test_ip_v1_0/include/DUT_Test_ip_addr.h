/*
 * File Name:         hdl_prj\ipcore\DUT_Test_ip_v1_0\include\DUT_Test_ip_addr.h
 * Description:       C Header File
 * Created:           2025-10-15 11:04:50
*/

#ifndef DUT_TEST_IP_H_
#define DUT_TEST_IP_H_

#define  IPCore_Reset_DUT_Test_ip       0x0  //write 0x1 to bit 0 to reset IP core
#define  IPCore_Enable_DUT_Test_ip      0x4  //enabled (by default) when bit 0 is 0x1
#define  IPCore_Timestamp_DUT_Test_ip   0x8  //contains unique IP timestamp (yymmddHHMM): 2510151022: 2510151048: 2510151104

#endif /* DUT_TEST_IP_H_ */
