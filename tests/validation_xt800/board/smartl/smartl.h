/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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


#ifndef SMART_CARD_H
#define SMART_CARD_H

/* APB frequence definition */
#define APB_DEFAULT_FREQ       48000000	/* Hz */

/* -------------------------  Interrupt Number Definition  ------------------------ */

typedef enum IRQn {
	CORET_IRQn = 1,
	UART0_IRQn = 2,
} IRQn_Type;

/* ================================================================================ */
/* ================      Processor and Core Peripheral Section     ================ */
/* ================================================================================ */

#include "csi_core.h"			/* Processor and core peripherals */

/* ================================================================================ */
/* ================       Device Specific Peripheral Section       ================ */
/* ================================================================================ */

/* ================================================================================ */
/* ============== Universal Asyncronous Receiver / Transmitter (UART) ============= */
/* ================================================================================ */
typedef struct {
	union {
		__IM uint32_t RBR;	/* Offset: 0x000 (R/ )  Receive buffer register */
		__OM uint32_t THR;	/* Offset: 0x000 ( /W)  Transmission hold register */
		__IOM uint32_t DLL;	/* Offset: 0x000 (R/W)  Clock frequency division low section register */
	};
	union {
		__IOM uint32_t DLH;	/* Offset: 0x004 (R/W)  Clock frequency division high section register */
		__IOM uint32_t IER;	/* Offset: 0x004 (R/W)  Interrupt enable register */
	};
	__IM uint32_t IIR;		/* Offset: 0x008 (R/ )  Interrupt indicia register */
	__IOM uint32_t LCR;		/* Offset: 0x00C (R/W)  Transmission control register */
	uint32_t RESERVED0;
	__IM uint32_t LSR;		/* Offset: 0x014 (R/ )  Transmission state register */
	uint32_t RESERVED1[25];
	__IM uint32_t USR;		/* Offset: 0x07c (R/ )  UART state register */
} SMARTL_UART_TypeDef;

/* ================================================================================ */
/* ================              Peripheral memory map             ================ */
/* ================================================================================ */
#define SMARTL_UART0_BASE            (0x40015000UL)

/* ================================================================================ */
/* ================             Peripheral declaration             ================ */
/* ================================================================================ */
#define SMARTL_UART0                 ((SMARTL_UART_TypeDef *)    SMARTL_UART0_BASE)

#endif
