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

/* SHL version 2.1.x */

/**************************************************************************************************

    void gemm_fp16_nhwc_matrix_2rowxn(__fp16 *output,
                                     const __fp16 *kernel,
                                     const __fp16 *input,
                                     const __fp16 *bias,
                                     int row,        // mregrows
                                     int k,          // matrix A col / matrix B row
                                     int n)          // matrix B col

    Algorithm works as follows:
        (1) perform matrix-multiplication [m, k] x [k, n] = [m, n]
            ...
    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr
        a4: row
        a5: k
        a6: n

        a0, a7: output addr (c00 c10)
        s1, s2: intput addr (a00 a10)

        t0 = col [mlenb/2]
        t1 = col * sizeof(__fp16)
        t2 = input stride [k * sizeof(__fp16)]
        t3 = output stride [n * sizeof(__fp16)]
        t4 = tmp
        t5 = row * col * sizeof(__fp16)

        m0-m1: intput
        m2-m3: kernel
        m4-m5: output

 *************************************************************************************************/
    .section        .text.gemm_fp16_nhwc_matrix_2rowxn, "ax", @progbits
    .align          5
    .global         gemm_fp16_nhwc_matrix_2rowxn
    .type           gemm_fp16_nhwc_matrix_2rowxn, @function

gemm_fp16_nhwc_matrix_2rowxn:
    addi            sp, sp, -24
    sd              s0, 0(sp)
    sd              s1, 8(sp)
    sd              s2, 16(sp)

    slli            t0, a4, 1       // row * sizeof(__fp16) = col
    slli            t1, t0, 1       // col * sizeof(__fp16)
    slli            t2, a5, 1       // input stride = k * sizeof(__fp16)
    slli            t3, a6, 1       // output stride = n * sizeof(__fp16)

    mul             t4, t0, a6      // row * n * sizeof(__fp16)
    add             a7, a0, t4      // out1
    mul             t4, t0, a5      // row * k * sizeof(__fp16)
    add             s0, a2, t4      // in1
    mul             t5, a4, t1      // row * col * sizeof(__fp16)

    mcfgn           zero, t0        // n = col
    bgt             t0, a6, gemm_fp16_2rowxn_tail   // n < col, jump n_tail
    mcfgk           zero, t1        // m = col * sizeof(__fp16) for load = col for matmul

gemm_fp16_2rowxn_n:
    mv              s1, a2          // in0
    mv              s2, s0          // in1

    mcfgmi          zero, 1
    mldh            m0, t3, (a3)    // bias [1][n]
    add             a3, a3, t1

    mcfgm           zero, a4        // m = row
    mmov.mv.i       m4, m0[0]
    mmov.mv.i       m5, m0[0]

    mv              t4, a5          // K

gemm_fp16_2rowxn_k:
    mldh            m0, t2, (s1)    // a00
    add             s1, s1, t1      // in0 += col * sizeof(__fp16)
    mldh            m1, t2, (s2)    // a10
    add             s2, s2, t1      // in1 += col * sizeof(__fp16)

    msldh           m2, t1, (a1)    // b00
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)
    msldh           m3, t1, (a1)    // b00 (half)
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)
    fmmacc.h        m4, m2, m0      // c00 += a00 * b00
    fmmacc.h        m5, m2, m1      // c10 += a10 * b00

    sub             t4, t4, t0      // K -= col
    bnez            t4, gemm_fp16_2rowxn_k

gemm_fp16_2rowxn_n_end:
    msth            m4, t3, (a0)
    add             a0, a0, t1      // out0 += col * sizeof(__fp16)
    msth            m5, t3, (a7)
    add             a7, a7, t1      // out1 += col * sizeof(__fp16)

    sub             a6, a6, t0      // n -= row * 2
    ble             t0, a6, gemm_fp16_2rowxn_n

gemm_fp16_2rowxn_tail:
    blez            a6, gemm_fp16_2rowxn_end    // n <= 0, jump end
    mv              s1, a2          // in0
    mv              s2, s0          // in1

    slli            a6, a6, 1   // n_tail * sizeof(fp16)
    mcfgmi          zero, 1
    mcfgk           zero, a6
    mldh            m0, t3, (a3)    // bias [1][n]
    // add             a3, a3, a6

    mcfgm           zero, a4        // m = row
    mmov.mv.i       m4, m0[0]
    mmov.mv.i       m5, m0[0]
    mcfgk           zero, t1        // k = col * sizeof(fp16)
    mv              t4, a5          // K

gemm_fp16_2rowxn_tail_k:
    mldh            m0, t2, (s1)    // a00
    add             s1, s1, t1      // in0 += col * sizeof(__fp16)
    mldh            m1, t2, (s2)    // a10
    add             s2, s2, t1      // in1 += col * sizeof(__fp16)

    msldh           m2, t1, (a1)    // b00
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)
    msldh           m3, t1, (a1)    // b00 (half)
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)
    fmmacc.h        m4, m2, m0      // c00 += a00 * b00
    fmmacc.h        m5, m2, m1      // c10 += a10 * b00

    sub             t4, t4, t0      // K -= col
    bnez            t4, gemm_fp16_2rowxn_tail_k

gemm_fp16_2rowxn_tail_end:
    mcfgk           zero, a6
    msth            m4, t3, (a0)
    // add             a0, a0, a6      // out0 += n_tail * sizeof(__fp16)
    msth            m5, t3, (a7)
    // add             a7, a7, a6      // out1 += n_tail * sizeof(__fp16)

gemm_fp16_2rowxn_end:
    ld              s0, 0(sp)
    ld              s1, 8(sp)
    ld              s2, 16(sp)
    addi            sp, sp, 24

    ret

/**************************************************************************************************

    void gemm_fp16_nhwc_matrix_rowxn(__fp16 *output,
                                     const __fp16 *kernel,
                                     const __fp16 *input,
                                     const __fp16 *bias,
                                     int row,        // mregrows
                                     int k,          // matrix A col / matrix B row
                                     int n)          // matrix B col

    Algorithm works as follows:
        (1) perform matrix-multiplication [m, k] x [k, n] = [m, n]
            ...
    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr
        a4: row
        a5: k
        a6: n
        a7: intput addr (a00)

        t0 = col [mlenb/2]
        t1 = col * sizeof(__fp16)
        t2 = input stride [k * sizeof(__fp16)]
        t3 = output stride [n * sizeof(__fp16)]
        t4 = tmp
        t5 = row * col * sizeof(__fp16)

        m0-m1: intput
        m2-m3: kernel
        m4-m5: output

 *************************************************************************************************/
    .section        .text.gemm_fp16_nhwc_matrix_rowxn, "ax", @progbits
    .align          5
    .global         gemm_fp16_nhwc_matrix_rowxn
    .type           gemm_fp16_nhwc_matrix_rowxn, @function

gemm_fp16_nhwc_matrix_rowxn:

    slli            t0, a4, 1       // row * sizeof(__fp16) = col
    slli            t1, t0, 1       // col * sizeof(__fp16)
    slli            t2, a5, 1       // input stride = k * sizeof(__fp16)
    slli            t3, a6, 1       // output stride = n * sizeof(__fp16)

    mul             t5, a4, t1      // row * col * sizeof(__fp16)

    mcfgn           zero, t0
    bgt             t0, a6, gemm_fp16_rowxn_tail   // n < col, jump n_tail
    mcfgk           zero, t1        // m = col * sizeof(__fp16) for load = col for matmul

gemm_fp16_rowxn_n:
    mcfgk           zero, t1
    mv              a7, a2          // in0

    mcfgmi          zero, 1
    mldh            m0, t3, (a3)    // bias [1][n]
    add             a3, a3, t0

    mcfgm           zero, a4
    mmov.mv.i       m4, m0[0]

    mv              t4, a5          // K

gemm_fp16_rowxn_k:
    mldh            m0, t2, (a7)    // a00
    add             a7, a7, t1      // in0 += col * sizeof(__fp16)

    msldh           m2, t1, (a1)    // b00
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)
    msldh           m3, t1, (a1)    // b00 (half)
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)
    fmmacc.h        m4, m2, m0      // c00 += a00 * b00

    sub             t4, t4, t0      // K -= col
    bnez            t4, gemm_fp16_rowxn_k

gemm_fp16_rowxn_n_end:
    msth            m4, t3, (a0)
    add             a0, a0, t1      // out0 += col * sizeof(__fp16)

    sub             a6, a6, t0      // N -= row * 2
    ble             t0, a6, gemm_fp16_rowxn_n

gemm_fp16_rowxn_tail:
    blez            a6, gemm_fp16_rowxn_end    // n <= 0, jump end
    mv              a7, a2          // in0

    slli            a6, a6, 1   // n_tail * sizeof(fp16)
    mcfgmi          zero, 1
    mcfgk           zero, a6
    mldh            m0, t3, (a3)    // bias [1][n]
    // add             a3, a3, t0

    mcfgm           zero, a4
    mmov.mv.i       m4, m0[0]
    mcfgk           zero, t1        // k = col * sizeof(fp16)
    mv              t4, a5          // K

gemm_fp16_rowxn_tail_k:
    mldh            m0, t2, (a7)    // a00
    add             a7, a7, t1      // in0 += col * sizeof(__fp16)

    msldh           m2, t1, (a1)    // b00
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)
    msldh           m3, t1, (a1)    // b00 (half)
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)
    fmmacc.h        m4, m2, m0      // c00 += a00 * b00

    sub             t4, t4, t0      // K -= col
    bnez            t4, gemm_fp16_rowxn_tail_k

gemm_fp16_rowxn_tail_end:
    mcfgk           zero, a6
    msth            m4, t3, (a0)

gemm_fp16_rowxn_end:
    ret

/**************************************************************************************************

    void gemm_fp16_nhwc_matrix_row_tailxn(__fp16 *output,
                                          const __fp16 *kernel,
                                          const __fp16 *input,
                                          const __fp16 *bias,
                                          int row,        // matrix A row
                                          int k,          // matrix A col / matrix B row
                                          int n)          // matrix B col

    Algorithm works as follows:
        (1) perform matrix-multiplication [m, k] x [k, n] = [m, n]
            ...
    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr
        a4: row
        a5: k
        a6: n
        a7: intput addr (a00)

        t0 = col [mlenb/2]
        t1 = col * sizeof(__fp16)
        t2 = input stride [k * sizeof(__fp16)]
        t3 = output stride [n * sizeof(__fp16)]
        t4 = tmp
        t5 = row * col * sizeof(__fp16)

        s0 = row

        m0-m1: intput
        m2-m3: kernel
        m4-m5: output

 *************************************************************************************************/
    .section        .text.gemm_fp16_nhwc_matrix_row_tailxn, "ax", @progbits
    .align          5
    .global         gemm_fp16_nhwc_matrix_row_tailxn
    .type           gemm_fp16_nhwc_matrix_row_tailxn, @function

gemm_fp16_nhwc_matrix_row_tailxn:
    addi            sp, sp, -8
    sd              s0, 0(sp)

    csrr            s0, xmlenb
    srai            s0, s0, 2       // row
    slli            t0, s0, 1       // col
    slli            t1, t0, 1       // col * sizeof(__fp16)
    slli            t2, a5, 1       // input stride = k * sizeof(__fp16)
    slli            t3, a6, 1       // output stride = n * sizeof(__fp16)

    mul             t5, s0, t1      // row * col * sizeof(__fp16)

    mcfgn           zero, t0
    bgt             t0, a6, gemm_fp16_row_tailxn_tail   // n < col, jump n_tail
    mcfgk           zero, t1        // m = col * sizeof(__fp16) for load = col for matmul

gemm_fp16_row_tailxn_n:
    mcfgk           zero, t1
    mv              a7, a2          // in0

    mcfgmi          zero, 1
    mldh            m0, t3, (a3)    // bias [1][n]
    add             a3, a3, t0

    mcfgm           zero, a4
    mmov.mv.i       m4, m0[0]

    mv              t4, a5          // K

gemm_fp16_row_tailxn_k:
    mcfgm           zero, s0
    msldh           m2, t1, (a1)    // b00
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)
    msldh           m3, t1, (a1)    // b00 (half)
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)

    mcfgm           zero, a4
    mldh            m0, t2, (a7)    // a00
    add             a7, a7, t1      // in0 += col * sizeof(__fp16)
    fmmacc.h        m4, m2, m0      // c00 += a00 * b00

    sub             t4, t4, t0      // K -= col
    bnez            t4, gemm_fp16_row_tailxn_k

gemm_fp16_row_tailxn_n_end:
    msth            m4, t3, (a0)
    add             a0, a0, t1      // out0 += col * sizeof(__fp16)

    sub             a6, a6, t0      // N -= row * 2
    ble             t0, a6, gemm_fp16_row_tailxn_n

gemm_fp16_row_tailxn_tail:
    blez            a6, gemm_fp16_row_tailxn_end    // n <= 0, jump end
    mv              a7, a2          // in0

    slli            a6, a6, 1   // n_tail * sizeof(fp16)
    mcfgmi          zero, 1
    mcfgk           zero, a6
    mldh            m0, t3, (a3)    // bias [1][n]
    // add             a3, a3, t0

    mcfgm           zero, a4
    mmov.mv.i       m4, m0[0]
    mcfgk           zero, t1        // k = col * sizeof(fp16)
    mv              t4, a5          // K

gemm_fp16_row_tailxn_tail_k:
    mcfgm           zero, s0
    msldh           m2, t1, (a1)    // b00
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)
    msldh           m3, t1, (a1)    // b00 (half)
    add             a1, a1, t5      // k0 += row * col * sizeof(__fp16)

    mcfgm           zero, a4
    mldh            m0, t2, (a7)    // a00
    add             a7, a7, t1      // in0 += col * sizeof(__fp16)
    fmmacc.h        m4, m2, m0      // c00 += a00 * b00

    sub             t4, t4, t0      // K -= col
    bnez            t4, gemm_fp16_row_tailxn_tail_k

gemm_fp16_row_tailxn_tail_end:
    mcfgk           zero, a6
    msth            m4, t3, (a0)

gemm_fp16_row_tailxn_end:
    ld              s0, 0(sp)
    addi            sp, sp, 8

    ret
    .end
