/*
 * Copyright (C) 2016-2019 C-SKY Limited. All rights reserved.
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
 * crt0.S -- startup codes for CK-CPU.
 *
 * Define a exception vector table
 *
 * Initial PSR register.
 * Initial VBR register, relocate exception vector table.
 * Initial sp(stack pointer) register.
 * Initial BSS section to "0"
 * Then goto "main".
 *
 * Copyright (C) 2008~2015  Hangzhou C-SKY Microsystems Co., Ltd
 */

.file "crt0.S"

/*
 * For importing variable or functions from script file "ckcpu.ld".
 */
.import __stack		/* The stack pointer from linking script file */
.import	__sbss		/* The end of bss from linking script file */
.import __ebss		/* The end of bss from linking script file */

/*
 * For importing variable or functions from other c or assemble files.
 */
.import main				/* The c codes entry */
.import coretimer_irq_handler
.import user1_int_irq_handler
.import user2_int_irq_handler
.import user3_int_irq_handler

/* For exporting varialbe or functions */
.export	ckcpu_vsr_table	/* Vector table base address. */
.export __start				/* The system entry. */

.section .exp_table
/* Vector table space. */

.align 10
ckcpu_vsr_table:

.long __start
.rept 31
.long default_exception_handler
.endr
.long default_exception_handler
.long coret_vsr_autovec
.rept 29
.long default_exception_handler
.endr

/* The ckcpu startup codes. */
.text
	/*
	 *
	 * This is the codes first entry point. This is where it all
	 * begins...
	 */

	.text
	.align	2
	.globl	__start
	.type	__start, %function
__start:
	/*
	 * Init psr value, enable exception, disable interrupt and fast interrupt.
	 * psr = 0x80000100
	 */
	lrw	a3, 0x80000300
	mtcr	a3, psr

	/*
	 * Setup initial vector base table for interrupts and exceptions
	 */
	lrw	a2, ckcpu_vsr_table
	mtcr	a2, vbr

	/* Initialize the normal stack pointer from the linker definition. */
	lrw	a1, __stack
	mov	sp, a1

	/*
	 * Zero out the bss region.
	 * NOTE: __sbss must align 4
	 */
	lrw	a3, __sbss	/* Get start of bss from linking script file */
	lrw	a2, __ebss	/* Get end of bss from linking script file */
	subu	a2, a3		/* Calculate size of bss */
	lsri	a2, 2		/* Size of whole words */
	cmpnei	a2, 0
	bf	__goto_c
	movi	a1, 0		/* Set zero value to write */
2:
	stw	a1, (a3)	/* Zero next word */
	addi	a3, 4		/* Increase bss pointer */
	subi	a2, 1		/* Decrease counter */
	cmpnei	a2, 0
	bt	2b		/* Repeat for all bss */

	/*
	 * Assember start up done, C codes start here.
	 */
__goto_c:
	bsr	main

	/* Should never get here. */
__exit:
	lrw     r1, 0x10002044
	st.w    r0, (r1, 0x0)
	lrw     r1, 0x10002000
	st.w    r0, (r1, 0x0)
	bkpt
	br	__exit

	.size	__start, . - __start
/*
 * default exception service routine, before r0 initialized
 */
	.text
	.balign 4
default_exception_handler:
	br	default_exception_handler
	rte

.weak coretimer_irq_handler
/*
 * coret interrupt vector handler
 */
	.text
	.balign 4
	.globl	coret_vsr_autovec
	.type	coret_vsr_autovec, %function
coret_vsr_autovec:
	/* save context */
	subi    sp, 28             /* Allocate space for all registers */
	stw     a0, (sp, 0)
	stw     a1, (sp, 4)
	stw     a2, (sp, 8)
	stw     a3, (sp, 12)
	stw     t0, (sp, 16)
	stw     t1, (sp, 20)
	stw     lr, (sp, 24)

	subi    sp, 8
	mfcr    a3, epsr
	stw     a3, (sp, 4)	/* save epsr registwr */
	mfcr    a2, epc
	stw     a2, (sp, 0)	/* save epc register */
	psrset	ie

	bsr	coretimer_irq_handler

	ldw     a3, (sp, 0)
	mtcr    a3, epc		/* restore the epc */
	ldw     a2, (sp, 4)
	mtcr    a2, epsr	/* restore the epsr */
	addi    sp, 8

	/* restore all the regs */
	ldw     a0, (sp, 0)
	ldw     a1, (sp, 4)
	ldw     a2, (sp, 8)
	ldw     a3, (sp, 12)
	ldw     t0, (sp, 16)
	ldw     t1, (sp, 20)
	ldw     lr, (sp, 24)
	addi    sp, 28

	rte

	.size	coret_vsr_autovec, . - coret_vsr_autovec

