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

/**************************************************************************************************

    void gemm_int8_to_int32_nhwc_2rowxn_matrix(const int32_t *output,
                                               const int8_t *kernel,
                                               const int8_t *input,
                                               const int32_t *bias,
                                               int m,          // maxtrix A row
                                               int k,          // maxtrix A col / maxtrix B row
                                               int n)          // maxtrix B col

    constrains:
        k % mlenb == 0

    Algorithm works as follows:
        (1) perform matrix-multiplication by matrix extension: [m, k] x [n, k]T = [m, n]
            ...

    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr
        a4: m [row]  MLEN/32
        a5: k [kernel_size * inc]
        a6: n [out_hw]
        a7: output addr line1

        s0: out_zp
        s1: k_loop cnt

        t0:
        t1: row * mlenb
        t2: input addr line1
        t3: mlenb
        t4: k_loop
        t5: n_loop

 *************************************************************************************************/
    .file           "gemm_int8_to_int32_nhwc_matrix.S"
    .section        .text.gemm_int8_to_int32_nhwc_2rowxn_matrix, "ax", @progbits
    .align          5
    .global         gemm_int8_to_int32_nhwc_2rowxn_matrix
    .type           gemm_int8_to_int32_nhwc_2rowxn_matrix, @function


gemm_int8_to_int32_nhwc_2rowxn_matrix:
    addi            sp, sp, -16
    sd              s0, 0(sp)
    sd              s1, 8(sp)

    slli            s0, a6, 2   // n * 4: output stride
    slli            t4, a4, 1   // row * 2
    mv              t5, a6      // t5 = n

    mul             t1, a4, a5  // row * k
    add             t2, a2, t1  // t2 = input addr line1

    mul             t1, a4, s0  // row * n * 4
    add             a7, a0, t1  // a7 = output addr line1

    slli            t3, a4, 2   // mlenb = row * 4 = 16
    mul             t1, a4, t3  // row * mlenb

    mcfgn           zero, a4
    bgt             t4, t5, matrix_2x1_start    // if row * 2 > n, jump to matrix_m2xn1

matrix_2x2_start:
    mcfgmi	        zero, 2
    mcfgk	        zero, t3        // set K = mlenb
    mldw	        m0, t3, (a3)    // load bias
    add             a3, a3, t3
    add             a3, a3, t3      // + row * 2 * sizeof(int32_t)

    mcfgm	        zero, a4
    mmov.mv.i	    m4, m0[0]
    mmov.mv.i	    m5, m0[1]
    mmov.mv.i	    m6, m0[0]
    mmov.mv.i	    m7, m0[1]

    mv              s1, a5  // k_loop

    mv              t0, a2
    mv              t6, t2  // input addr tmp

matrix_2x2_loop:
    mldb	        m0, a5, (t0)	// a00  input line0
    add             t0, t0, t3
    mldb	        m1, a5, (t6)	// a10  input line1
    add             t6, t6, t3

    msldb	        m2, t3, (a1)	// b00  kernel
    add             a1, a1, t1
    msldb	        m3, t3, (a1)	// b10
    add             a1, a1, t1

    mmaqa.b         m4, m2, m0		// c00 += a00 * b00
    mmaqa.b         m5, m3, m0		// c01 += a00 * b10
    mmaqa.b         m6, m2, m1		// c10 += a10 * b00
    mmaqa.b         m7, m3, m1		// c11 += a10 * b10

    sub             s1, s1, t3
    bnez            s1, matrix_2x2_loop

matrix_2x2_end:
    mcfgk	        zero, t3
    mstb	        m4, s0, (a0)
    add             a0, a0, t3
    mstb	        m5, s0, (a0)
    add             a0, a0, t3
    mstb	        m6, s0, (a7)
    add             a7, a7, t3
    mstb	        m7, s0, (a7)
    add             a7, a7, t3

    sub             t5, t5, t4      // n -= row * 2
    ble             t4, t5, matrix_2x2_start

matrix_2x1_start:
    bgt             a4, t5, matrix_2xtail_start    // if row > n, jump to matrix_m2xn_tail
    mcfgmi	        zero, 1
    mcfgk	        zero, t3      // set K = mlenb
    mldw	        m0, t3, (a3)
    add             a3, a3, t3

    mcfgm	        zero, a4
    mmov.mv.i	    m4, m0[0]
    mmov.mv.i	    m6, m0[0]

    mv              s1, a5  // k_loop
    mv              t0, a2
    mv              t6, t2  // input addr tmp

matrix_2x1_loop:
    mldb	        m0, a5, (t0)	// a00  input line0
    add             t0, t0, t3
    mldb	        m1, a5, (t6)	// a10  input line1
    add             t6, t6, t3
    msldb	        m2, t3, (a1)	// b00  kernel
    add             a1, a1, t1

    mmaqa.b         m4, m2, m0		// c00 += a00 * b00
    mmaqa.b         m6, m2, m1		// c10 += a10 * b00

    sub             s1, s1, t3
    bnez            s1, matrix_2x1_loop

matrix_2x1_end:
    mcfgk	        zero, t3
    mstb	        m4, s0, (a0)
    add             a0, a0, t3
    mstb	        m6, s0, (a7)
    add             a7, a7, t3
    sub             t5, t5, a4

matrix_2xtail_start:
    blez            t5, matrix_m2_end    // n <= 0, jump end

    slli            t4, t5, 2   // n_tail * 4
    mcfgmi	        zero, 1
    mcfgk	        zero, t4
    mldw	        m0, t3, (a3)
    // add             a3, a3, t4

    mcfgm	        zero, a4
    mmov.mv.i	    m4, m0[0]
    mmov.mv.i	    m6, m0[0]
    mcfgk	        zero, t3

    mv              s1, a5  // k_loop
    mv              t0, a2
    mv              t6, t2  // input addr tmp

matrix_2xtail_loop:
    mldb	        m0, a5, (t0)	// a00  input line0
    add             t0, t0, t3
    mldb	        m1, a5, (t6)	// a10  input line1
    add             t6, t6, t3
    msldb	        m2, t3, (a1)	// b00  kernel
    add             a1, a1, t1

    mmaqa.b         m4, m2, m0		// c00 += a00 * b00
    mmaqa.b         m6, m2, m1		// c10 += a10 * b00

    sub             s1, s1, t3
    bnez            s1, matrix_2xtail_loop

matrix_2xtail_end:
    mcfgk	        zero, t4
    mstb	        m4, s0, (a0)
    // add             a0, a0, t4
    mstb	        m6, s0, (a7)
    // add             a7, a7, t4

matrix_m2_end:
    ld              s0, 0(sp)
    ld              s1, 8(sp)
    addi            sp, sp, 16
    ret

/**************************************************************************************************

    void gemm_int8_to_int32_nhwc_rowxn_matrix(const int32_t *output,
                                              const int8_t *kernel,
                                              const int8_t *input,
                                              const int32_t *bias,
                                              int m,          // maxtrix A row
                                              int k,          // maxtrix A col / maxtrix B row
                                              int n)          // maxtrix B col

    constrains:
        k % mlenb == 0

    Algorithm works as follows:
        (1) perform matrix-multiplication by matrix extension: [m_tail, k] x [n, k]T = [m_tail, n]
            ...

    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr
        a4: m [row]  MLEN/32
        a5: k [kernel_size * inc]
        a6: n [out_hw]
        a7: output addr line1

        s0: out_zp
        s1: k_loop cnt

 *************************************************************************************************/

     .section        .text.gemm_int8_to_int32_nhwc_rowxn_matrix, "ax", @progbits
    .align          5
    .global         gemm_int8_to_int32_nhwc_rowxn_matrix
    .type           gemm_int8_to_int32_nhwc_rowxn_matrix, @function

gemm_int8_to_int32_nhwc_rowxn_matrix:
    addi            sp, sp, -16
    sd              s0, 0(sp)
    sd              s1, 8(sp)

    slli            s0, a6, 2   // n * 4
    slli            t4, a4, 1   // row * 2
    mv              t5, a6      // t5 = n

    slli            t3, a4, 2   // mlenb = row * 4 = 16
    mul             t1, a4, t3  // row * mlenb

    mcfgn           zero, a4
    bgt             t4, t5, matrix_1x1_start    // if row * 2 > n, jump to matrix_m1xn1

matrix_1x2_start:
    mcfgmi	        zero, 2
    mcfgk	        zero, t3		// set K = mlenb
    mldw	        m0, t3, (a3)    // load bias
    add             a3, a3, t3
    add             a3, a3, t3      // + row * 2 * sizeof(int32_t)

    mcfgm	        zero, a4
    mmov.mv.i	    m4, m0[0]
    mmov.mv.i	    m5, m0[1]

    mv              s1, a5  // k_loop
    mv              t0, a2  // input addr tmp

matrix_1x2_loop:
    mldb	        m0, a5, (t0)	// a00  input line0
    add             t0, t0, t3

    msldb	        m2, t3, (a1)	// b00  kernel
    add             a1, a1, t1
    msldb	        m3, t3, (a1)	// b10
    add             a1, a1, t1

    mmaqa.b         m4, m2, m0		// c00 += a00 * b00
    mmaqa.b         m5, m3, m0		// c01 += a00 * b10

    sub             s1, s1, t3
    bnez            s1, matrix_1x2_loop

matrix_1x2_end:
    mcfgk	        zero, t3
    mstb	        m4, s0, (a0)
    add             a0, a0, t3
    mstb	        m5, s0, (a0)
    add             a0, a0, t3

    sub             t5, t5, t4
    ble             t4, t5, matrix_1x2_start

matrix_1x1_start:
    bgt             a4, t5, matrix_1xtail_start
    mcfgmi	        zero, 1
    mcfgk	        zero, t3        // set K = mlenb
    mldw	        m0, t3, (a3)    // load bias
    add             a3, a3, t3      // + row * sizeof(int32_t)

    mcfgm	        zero, a4
    mmov.mv.i	    m4, m0[0]

    mv              s1, a5  // k_loop
    mv              t0, a2  // input addr tmp

matrix_1x1_loop:
    mldb	        m0, a5, (t0)	// a00  input line0
    add             t0, t0, t3
    msldb	        m2, t3, (a1)	// b00  kernel
    add             a1, a1, t1

    mmaqa.b         m4, m2, m0		// c00 += a00 * b00

    sub             s1, s1, t3
    bnez            s1, matrix_1x1_loop

matrix_1x1_end:
    mstb	        m4, s0, (a0)
    add             a0, a0, t3
    sub             t5, t5, a4

matrix_1xtail_start:
    blez            t5, matrix_m1_end    // n <= 0, jump end

    slli            t4, t5, 2   // n_tail * 4
    mcfgmi	        zero, 1
    mcfgk	        zero, t4
    mldw	        m0, t3, (a3)    // load bias
    // add             a3, a3, t4      // + n_tail * sizeof(int32_t)

    mcfgm	        zero, a4
    mmov.mv.i	    m4, m0[0]
    mcfgk	        zero, t3

    mv              s1, a5  // k_loop
    mv              t0, a2  // input addr tmp

matrix_1xtail_loop:
    mldb	        m0, a5, (t0)	// a00  input line0
    add             t0, t0, t3
    msldb	        m2, t3, (a1)	// b00  kernel
    add             a1, a1, t1

    mmaqa.b         m4, m2, m0		// c00 += a00 * b00

    sub             s1, s1, t3
    bnez            s1, matrix_1xtail_loop

matrix_1xtail_end:
    mcfgk	        zero, t4
    mstb	        m4, s0, (a0)
    // add             a0, a0, t4

matrix_m1_end:
    ld              s0, 0(sp)
    ld              s1, 8(sp)
    addi            sp, sp, 16
    ret

/**************************************************************************************************

    void gemm_int8_to_int32_nhwc_row_tailxn_matrix(const int32_t *output,
                                                   const int8_t *kernel,
                                                   const int8_t *input,
                                                   const int32_t *bias,
                                                   int m,          // maxtrix A row
                                                   int k,          // maxtrix A col / maxtrix B row
                                                   int n)          // maxtrix B col

    Algorithm works as follows:
        (1) perform matrix-multiplication by matrix extension: [packn, k] x [k, n] = [packn, n]
            ...

    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr
        a4: m [row]  MLEN/32
        a5: k [kernel_size * inc]
        a6: n [out_hw]
        a7: output addr line1

        s0: out_zp
        s1: k_loop cnt

        t2: mult addr
        t6: shift addr

 *************************************************************************************************/
     .section        .text.gemm_int8_to_int32_nhwc_row_tailxn_matrix, "ax", @progbits
    .align          5
    .global         gemm_int8_to_int32_nhwc_row_tailxn_matrix
    .type           gemm_int8_to_int32_nhwc_row_tailxn_matrix, @function

gemm_int8_to_int32_nhwc_row_tailxn_matrix:
    addi            sp, sp, -16
    sd              s0, 0(sp)
    sd              s1, 8(sp)
    ld              t2, 16(sp)
    ld              t6, 24(sp)

    slli            s0, a6, 2   // n * 4

    csrr            a7, xrlenb
    srai            a7, a7, 2   // row
    slli            t4, a7, 1   // row * 2
    mv              t5, a6      // t5 = n

    slli            t3, a7, 2   // mlenb = row * 4 = 16
    mul             t1, a7, t3  // row * mlenb

    mcfgn           zero, a7
    bgt             t4, t5, matrix_tailx1_start    // if row * 2 > n, jump to matrix_m1xn1

matrix_tailx2_start:
    mcfgmi	        zero, 2
    mcfgk	        zero, t3		// set K = mlenb
    mldw	        m0, t3, (a3)    // load bias
    add             a3, a3, t3
    add             a3, a3, t3      // + row * 2 * sizeof(int32_t)

    mcfgm	        zero, a4
    mmov.mv.i	    m4, m0[0]
    mmov.mv.i	    m5, m0[1]

    mv              s1, a5  // k_loop
    mv              t0, a2  // input addr tmp

matrix_tailx2_loop:
    mcfgm           zero, a7
    msldb	        m2, t3, (a1)	// b00  kernel
    add             a1, a1, t1
    msldb	        m3, t3, (a1)	// b10
    add             a1, a1, t1

    mcfgm	        zero, a4
    mldb	        m0, a5, (t0)	// a00  input line0
    add             t0, t0, t3

    mmaqa.b         m4, m2, m0		// c00 += a00 * b00
    mmaqa.b         m5, m3, m0		// c01 += a00 * b10

    sub             s1, s1, t3
    bnez            s1, matrix_tailx2_loop

matrix_tailx2_end:
    mcfgm	        zero, a4
    mcfgk	        zero, t3
    mstb	        m4, s0, (a0)
    add             a0, a0, t3
    mstb	        m5, s0, (a0)
    add             a0, a0, t3

    sub             t5, t5, t4
    ble             t4, t5, matrix_tailx2_start

matrix_tailx1_start:
    bgt             a7, t5, matrix_tailxtail_start
    mcfgmi	        zero, 1
    mcfgk	        zero, t3		// set K = mlenb
    mldw	        m0, t3, (a3)    // load bias
    add             a3, a3, t3      // + row * sizeof(int32_t)

    mcfgm	        zero, a4
    mmov.mv.i	    m4, m0[0]

    mv              s1, a5  // k_loop
    mv              t0, a2  // input addr tmp

matrix_tailx1_loop:
    mcfgm	        zero, a7
    msldb	        m2, t3, (a1)	// b00  kernel
    add             a1, a1, t1
    mcfgm	        zero, a4
    mldb	        m0, a5, (t0)	// a00  input line0
    add             t0, t0, t3

    mmaqa.b         m4, m2, m0		// c00 += a00 * b00

    sub             s1, s1, t3
    bnez            s1, matrix_tailx1_loop

matrix_tailx1_end:
    mstb	        m4, s0, (a0)
    add             a0, a0, t3
    sub             t5, t5, a7

matrix_tailxtail_start:
    blez            t5, matrix_m_tail_end

    slli            t4, t5, 2   // n_tail * 4
    mcfgmi	        zero, 1
    mcfgk	        zero, t4		// set K = mlenb
    mldw	        m0, t3, (a3)    // load bias
    // add             a3, a3, t4

    mcfgm	        zero, a4
    mmov.mv.i	    m4, m0[0]
    mcfgk	        zero, t3

    mv              s1, a5  // k_loop
    mv              t0, a2  // input addr tmp

matrix_tailxtail_loop:
    mcfgm	        zero, a7
    msldb	        m2, t3, (a1)	// b00  kernel
    add             a1, a1, t1
    mcfgm	        zero, a4
    mldb	        m0, a5, (t0)	// a00  input line0
    add             t0, t0, t3

    mmaqa.b         m4, m2, m0		// c00 += a00 * b00

    sub             s1, s1, t3
    bnez            s1, matrix_tailxtail_loop

matrix_tailxtail_end:
    mcfgk	        zero, t4
    mstb	        m4, s0, (a0)
    // add             a0, a0, t4

matrix_m_tail_end:
    ld              s0, 0(sp)
    ld              s1, 8(sp)
    addi            sp, sp, 16
    ret
    .end
