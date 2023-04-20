/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * File : uart.h
 * Description: this file contains the macros support uart operations
 * Copyright (C):  2008 C-SKY Microsystem  Ltd.
 * Author(s):   Shuli wu
 * E_mail:  shuli_wu@c-sky.com
 * Contributors: Yun Ye
 * Date:  2008-9-25
 */

#ifndef __UART_H__
#define __UART_H__

#include "smartl.h"

#ifndef NULL
#define	NULL  0x00
#endif

#ifndef TRUE
#define TRUE  0x01
#endif
#ifndef FALSE
#define FALSE 0x00
#endif

#ifndef SUCCESS
#define SUCCESS  0
#endif
#ifndef FAILURE
#define FAILURE  -1
#endif

typedef unsigned char CK_UINT8;
typedef unsigned short CK_UINT16;
typedef unsigned int CK_UINT32;
typedef signed char CK_INT8;
typedef signed short CK_INT16;
typedef signed int CK_INT32;
typedef signed long CK_INT64;
typedef unsigned int BOOL;
#ifndef BYTE
typedef unsigned char BYTE;
#endif
#ifndef WORD
typedef unsigned short WORD;
#endif

#define  IN
#define  OUT
#define INOUT

#define UART_BUSY_TIMEOUT      1000000
#define UART_RECEIVE_TIMEOUT   1000
#define UART_TRANSMIT_TIMEOUT  1000


/* UART register bit definitions */
/* CK5108 */

#define USR_UART_BUSY           0x01
#define LSR_DATA_READY          0x01
#define LSR_THR_EMPTY           0x20
#define IER_RDA_INT_ENABLE      0x01
#define IER_THRE_INT_ENABLE     0x02
#define IIR_NO_ISQ_PEND         0x01

#define LCR_SET_DLAB            0x80       /* enable r/w DLR to set the baud rate */
#define LCR_PARITY_ENABLE	    0x08       /* parity enabled */
#define LCR_PARITY_EVEN         0x10   /* Even parity enabled */
#define LCR_PARITY_ODD          0xef   /* Odd parity enabled */
#define LCR_WORD_SIZE_5         0xfc   /* the data length is 5 bits */
#define LCR_WORD_SIZE_6         0x01   /* the data length is 6 bits */
#define LCR_WORD_SIZE_7         0x02   /* the data length is 7 bits */
#define LCR_WORD_SIZE_8         0x03   /* the data length is 8 bits */
#define LCR_STOP_BIT1           0xfb   /* 1 stop bit */
#define LCR_STOP_BIT2           0x04  /* 1.5 stop bit */

#define CK_LSR_PFE              0x80
#define CK_LSR_TEMT             0x40
#define CK_LSR_THRE             0x40
#define	CK_LSR_BI               0x10
#define	CK_LSR_FE               0x08
#define	CK_LSR_PE               0x04
#define	CK_LSR_OE               0x02
#define	CK_LSR_DR               0x01
#define CK_LSR_TRANS_EMPTY      0x20

/************************************
 * (8 data bitbs, ODD, 1 stop bits)
 ***********************************/
#define BAUDRATE   19200
/*
 * Terminal uart to use
 */
#define  CONFIG_TERMINAL_UART UART0

//////////////////////////////////////////////////////////////////////////////////////////
typedef enum{
  B4800=4800,
  B9600=9600,
  B14400=14400,
  B19200=19200,
  B56000=56000,
  B38400=38400,
  B57600=57600,
  B115200=115200
}CK_Uart_Baudrate;


typedef enum{
  UART0,
  UART1,
  UART2,
  UART3
}CK_Uart_Device;

typedef enum{
  WORD_SIZE_5,
  WORD_SIZE_6,
  WORD_SIZE_7,
  WORD_SIZE_8
}CK_Uart_WordSize;

typedef enum{
  ODD,
  EVEN,
  NONE
}CK_Uart_Parity;

typedef enum{
	  LCR_STOP_BIT_1,
		LCR_STOP_BIT_2
}CK_Uart_StopBit;


typedef enum{
    CK_Uart_CTRL_C = 0,
    CK_Uart_FrameError = 1,
    CK_Uart_ParityError = 2
}CKEnum_Uart_Error;

typedef struct CK_UART_Info_t {
	CK_UINT32 id;
	SMARTL_UART_TypeDef *addr;
	CK_UINT32 irq ;
	BOOL bopened;
	void  (* handler)(CK_INT8 error);
	CK_Uart_Baudrate baudrate;
	CK_Uart_Parity parity;
	CK_Uart_WordSize word;
	CK_Uart_StopBit stop;
	BOOL btxquery;
	BOOL brxquery;
} CKStruct_UartInfo, *PCKStruct_UartInfo;

CK_INT32 CK_Uart_DriverInit();


/////////////////////////////////////////////////////////////////
/* open the uart :
 * set the callback function --- handler(void);
 * intilize the serial port,sending and receiving buffer;
 * intilize irqhandler ;
 * register irqhandler
 * return: SUCCESS or FAILURE
 */
CK_INT32 CK_Uart_Open( CK_Uart_Device uartid,void (*handler)(CK_INT8 error));

/* This function is used to close the uart
 * clear the callback function
 * free the irq
 * return: SUCCESS or FAILURE
 */
 CK_INT32 CK_Uart_Close( CK_Uart_Device uartid);

 /*
 * This function is used to change the bautrate of uart.
 * Parameters:
 * uartid--a basepointer, could be one of UART0, UART1, UART2 or UART3.
 * baudrate--the baudrate that user typed in.
 * return: SUCCESS or FAILURE
*/

CK_INT32 CK_Uart_ChangeBaudrate(
     CK_Uart_Device uartid,  CK_Uart_Baudrate baudrate);

/*
 * This function is used to enable or disable parity, also to set ODD or EVEN
 * parity.
 * Parameters:
 *   uartid--a basepointer, could be one of UART0, UART1, UART2 or UART3.
 *   parity--ODD=8, EVEN=16, or NONE=0.
 * return: SUCCESS or FAILURE
*/

CK_INT32 CK_Uart_SetParity(
     CK_Uart_Device uartid,  CK_Uart_Parity parity);

/*
 * We can call this function to set the stop bit--1 bit, 1.5 bits, or 2 bits.
 * But that it's 1.5 bits or 2, is decided by the wordlenth. When it's 5 bits,
 * there are 1.5 stop bits, else 2.
 * Parameters:
 *   uartid--a basepointer, could be one of UART0, UART1, UART2 or UART3.
 *	 stopbit--it has two possible value: STOP_BIT_1 and STOP_BIT_2.
 * return: SUCCESS or FAILURE
*/

CK_INT32 CK_Uart_SetStopBit(
     CK_Uart_Device uartid,  CK_Uart_StopBit stopbit);

/*
 * We can use this function to reset the transmit data length,and we
 * have four choices:5, 6, 7, and 8 bits.
 * Parameters:
 * 	uartid--a basepointer, could be one of UART0, UART1, UART2 or UART3.
 * 	wordsize--the data length that user decides
 * return: SUCCESS or FAILURE
*/

CK_INT32 CK_Uart_SetWordSize(CK_Uart_Device uartid,  CK_Uart_WordSize wordsize);

/*
 * This function is used to set the transmit mode, interrupt mode or
 * query mode.
 * Parameters:
 * 	uartid--a basepointer, could be one of UART0, UART1, UART2 or UART3.
 *  bQuery--it indicates the transmit mode: TRUE - query mode, FALSE -
 *  inerrupt mode
 * return: SUCCESS or FAILURE
*/

CK_INT32 CK_Uart_SetTXMode( CK_Uart_Device uartid, BOOL  bQuery);

/*
 * This function is used to set the receive mode, interrupt mode or
 * query mode.
 * Parameters:
 * 	uartid--a basepointer, could be one of UART0, UART1, UART2 or UART3.
 *  bQuery--it indicates the receive mode: TRUE - query mode, FALSE -
 *  interrupt mode
 * return: SUCCESS or FAILURE

*/
CK_INT32 CK_Uart_SetRXMode( CK_Uart_Device uartid, BOOL bQuery);

/* This function is used to get character,in query mode or interrupt mode.
 * Parameters:
 * 	 uartid--a basepointer, could be one of UART0, UART1, UART2 or UART3.
 *   brxquery--it indicates the receive mode: TRUE - query mode, FALSE -
 *   interrupt mode
 * return: SUCCESS or FAILURE
 */
CK_INT32 CK_Uart_GetChar(CK_Uart_Device uartid,  CK_UINT8 *ch);

/* This function is used to transmit character,in query mode or interrupt mode.
 * Parameters:
 * 	 uartid--a basepointer, could be one of UART0, UART1, UART2 or UART3.
 *   brxquery--it indicates the receive mode: TRUE - query mode, FALSE -
 *   interrupt mode
 * Return: SUCCESS or FAILURE.
 */
CK_INT32 CK_Uart_PutChar(CK_Uart_Device uartid, CK_UINT8 ch);

/*
 * initialize the uart:
 * baudrate: 19200
 * date length: 8 bits
 * paity: None(disabled)
 * number of stop bits: 1 stop bit
 * query mode
 * return: SUCCESS
 */
CK_INT32 CK_Uart_Init( CK_Uart_Device uartid);

/*
 */
CK_INT32 CK_Uart_ConfigDMA(
 CK_Uart_Device uartid,
 char *buffer,
 BOOL btx,
 CK_INT32 txrxsize,
 void (*handler)()
);

/*
 */
void CK_Uart_StartDMARxTx (void);

void CK_UART_ClearRxBuffer(CK_Uart_Device uartid);

/* This function is used to get character,in query mode or interrupt mode*/
CK_INT32 CK_Uart_GetCharUnBlock(IN CK_Uart_Device uartid, OUT CK_UINT8 *ch);
#endif
