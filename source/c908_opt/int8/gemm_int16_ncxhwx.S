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

    void gemm_int16_ncxhwx_12xpackn(const int32_t *output,
                                    const int16_t *kernel,
                                    const int16_t *input,
                                    int k,          // maxtrix A col / maxtrix B row
                                    int n)          // maxtrix B col

    Algorithm works as follows:
        (1) perform matrix-multiplication [packn, k] x [k, n] = [packn, n]
        (2) for int8 winograd
            ...

    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: k [in_ch]
        a4: n [tile]

        a5 = hold kernel data addr
        t0 = packn * 2: kernel_addr stride
        t5 = packn * 4: output_addr stride
        t6 = k2 loop cnt
        a6 = n12
        a7 = n_tail

        t1-t4: hold input data
        s1-s4: hold input data

        v2/v4:   hold kernel data
        v8-v31:  acc v-reg

 *************************************************************************************************/
    .file           "gemm_int16_ncxhwx.S"
    .section        .text.gemm_int16_ncxhwx_12xpackn, "ax", @progbits
    .align          5
    .global         gemm_int16_ncxhwx_12xpackn
    .type           gemm_int16_ncxhwx_12xpackn, @function

gemm_int16_ncxhwx_12xpackn:
    addi            sp, sp, -32
    sd              s1, 0(sp)
    sd              s2, 8(sp)
    sd              s3, 16(sp)
    sd              s4, 24(sp)

    li              t0, 12
    divw            a6, a4, t0  // a6 = n12
    remw            a7, a4, t0  // a7 = n % 12 (n_tail)

    csrr            t0, vlenb   // t0 = vlen/8 = packn/2 * 4 = 16
    slli            t5, t0, 1   // packn * 4 = 32

    beqz            a6, packnx8_start  // if n12==0, jump to packnx8

packnx12_start:
    vsetvli         zero, t0, e16, m1
    vmv.v.x         v8, zero
    vmv.v.x         v9, zero
    vmv.v.x         v10, zero
    vmv.v.x         v11, zero
    vmv.v.x         v12, zero
    vmv.v.x         v13, zero
    vmv.v.x         v14, zero
    vmv.v.x         v15, zero
    vmv.v.x         v16, zero
    vmv.v.x         v17, zero
    vmv.v.x         v18, zero
    vmv.v.x         v19, zero
    vmv.v.x         v20, zero
    vmv.v.x         v21, zero
    vmv.v.x         v22, zero
    vmv.v.x         v23, zero
    vmv.v.x         v24, zero
    vmv.v.x         v25, zero
    vmv.v.x         v26, zero
    vmv.v.x         v27, zero
    vmv.v.x         v28, zero
    vmv.v.x         v29, zero
    vmv.v.x         v30, zero
    vmv.v.x         v31, zero

    mv              a5, a1  // kernel origin addr
    // pre-load kernel matrix
    vle16.v         v2, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2
    // pre-load input matrix
    lwd             t1, t3, 0(a2)
    srli            t2, t1, 16
    srli            t4, t3, 16

    srai            t6, a3, 1   // k2

packnx12_k2:
    vle16.v         v4, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2

    vwmacc.vx       v8, t1, v2
    vwmacc.vx       v12, t3, v2
    lwd             s1, s3, 8(a2)
    vwmacc.vx       v10, t2, v2
    srli            s2, s1, 16
    vwmacc.vx       v14, t4, v2
    srli            s4, s3, 16
    vwmacc.vx       v16, s1, v2
    vwmacc.vx       v20, s3, v2
    lwd             t1, t3, 16(a2)
    addi            a2, a2, 24
    vwmacc.vx       v18, s2, v2
    srli            t2, t1, 16
    vwmacc.vx       v22, s4, v2
    srli            t4, t3, 16
    vwmacc.vx       v24, t1, v2
    vwmacc.vx       v28, t3, v2
    lwd             s1, s3, 0(a2)
    vwmacc.vx       v26, t2, v2
    srli            s2, s1, 16
    vwmacc.vx       v30, t4, v2
    srli            s4, s3, 16

    vle16.v         v2, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2

    vwmacc.vx       v8, s1, v4
    vwmacc.vx       v12, s3, v4
    lwd             t1, t3, 8(a2)
    vwmacc.vx       v10, s2, v4
    srli            t2, t1, 16
    vwmacc.vx       v14, s4, v4
    srli            t4, t3, 16
    vwmacc.vx       v16, t1, v4
    vwmacc.vx       v20, t3, v4
    lwd             s1, s3, 16(a2)
    addi            a2, a2, 24
    vwmacc.vx       v18, t2, v4
    srli            s2, s1, 16
    vwmacc.vx       v22, t4, v4
    srli            s4, s3, 16
    vwmacc.vx       v24, s1, v4
    vwmacc.vx       v28, s3, v4
    lwd             t1, t3, 0(a2)
    vwmacc.vx       v26, s2, v4
    srli            t2, t1, 16
    vwmacc.vx       v30, s4, v4
    srli            t4, t3, 16

    addi            t6, t6, -1
    bnez            t6, packnx12_k2

packnx12_end:
    vsetvli         zero, zero, e32, m2
    vse32.v         v8, (a0)
    add             a0, a0, t5
    vse32.v         v10, (a0)
    add             a0, a0, t5
    vse32.v         v12, (a0)
    add             a0, a0, t5
    vse32.v         v14, (a0)
    add             a0, a0, t5
    vse32.v         v16, (a0)
    add             a0, a0, t5
    vse32.v         v18, (a0)
    add             a0, a0, t5
    vse32.v         v20, (a0)
    add             a0, a0, t5
    vse32.v         v22, (a0)
    add             a0, a0, t5
    vse32.v         v24, (a0)
    add             a0, a0, t5
    vse32.v         v26, (a0)
    add             a0, a0, t5
    vse32.v         v28, (a0)
    add             a0, a0, t5
    vse32.v         v30, (a0)
    add             a0, a0, t5

    addi            a6, a6, -1
    bnez            a6, packnx12_start

packnx8_start:
    andi            a6, a7, 8       // s1 = bool_n8
    beqz            a6, packnx4_start  // if n8==0, jump to packnx4

    vsetvli         zero, t0, e16, m1
    vmv.v.x         v8, zero
    vmv.v.x         v9, zero
    vmv.v.x         v10, zero
    vmv.v.x         v11, zero
    vmv.v.x         v12, zero
    vmv.v.x         v13, zero
    vmv.v.x         v14, zero
    vmv.v.x         v15, zero
    vmv.v.x         v16, zero
    vmv.v.x         v17, zero
    vmv.v.x         v18, zero
    vmv.v.x         v19, zero
    vmv.v.x         v20, zero
    vmv.v.x         v21, zero
    vmv.v.x         v22, zero
    vmv.v.x         v23, zero

    mv              a5, a1  // kernel origin addr
    // pre-load kernel matrix
    vle16.v         v2, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2
    // pre-load input matrix
    lwd             t1, t3, 0(a2)
    srli            t2, t1, 16
    srli            t4, t3, 16

    srai            t6, a3, 1   // k2

packnx8_k2:
    vle16.v         v4, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2

    vwmacc.vx       v8, t1, v2
    vwmacc.vx       v12, t3, v2
    lwd             s1, s3, 8(a2)
    vwmacc.vx       v10, t2, v2
    srli            s2, s1, 16
    vwmacc.vx       v14, t4, v2
    srli            s4, s3, 16
    vwmacc.vx       v16, s1, v2
    vwmacc.vx       v20, s3, v2
    lwd             t1, t3, 16(a2)
    vwmacc.vx       v18, s2, v2
    srli            t2, t1, 16
    vwmacc.vx       v22, s4, v2
    srli            t4, t3, 16

    vle16.v         v2, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2

    vwmacc.vx       v8, t1, v4
    vwmacc.vx       v12, t3, v4
    lwd             s1, s3, 24(a2)
    addi            a2, a2, 32
    vwmacc.vx       v10, t2, v4
    srli            s2, s1, 16
    vwmacc.vx       v14, t4, v4
    srli            s4, s3, 16
    vwmacc.vx       v16, s1, v4
    vwmacc.vx       v20, s3, v4
    lwd             t1, t3, 0(a2)
    vwmacc.vx       v18, s2, v4
    srli            t2, t1, 16
    vwmacc.vx       v22, s4, v4
    srli            t4, t3, 16

    addi            t6, t6, -1
    bnez            t6, packnx8_k2

packnx8_end:
    vsetvli         zero, zero, e32, m2
    vse32.v         v8, (a0)
    add             a0, a0, t5
    vse32.v         v10, (a0)
    add             a0, a0, t5
    vse32.v         v12, (a0)
    add             a0, a0, t5
    vse32.v         v14, (a0)
    add             a0, a0, t5
    vse32.v         v16, (a0)
    add             a0, a0, t5
    vse32.v         v18, (a0)
    add             a0, a0, t5
    vse32.v         v20, (a0)
    add             a0, a0, t5
    vse32.v         v22, (a0)
    add             a0, a0, t5

packnx4_start:
    andi            a6, a7, 4       // s1 = bool_n4
    beqz            a6, packnx2_start  // if n4==0, jump to packnx2

    vsetvli         zero, t0, e16, m1
    vmv.v.x         v8, zero
    vmv.v.x         v9, zero
    vmv.v.x         v10, zero
    vmv.v.x         v11, zero
    vmv.v.x         v12, zero
    vmv.v.x         v13, zero
    vmv.v.x         v14, zero
    vmv.v.x         v15, zero

    mv              a5, a1  // kernel origin addr
    // pre-load kernel matrix
    vle16.v         v2, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2
    // pre-load input matrix
    lwd             t1, t3, 0(a2)
    srli            t2, t1, 16
    srli            t4, t3, 16

    srai            t6, a3, 1   // k2

packnx4_k2:
    vle16.v         v4, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2

    vwmacc.vx       v8, t1, v2
    lwd             s1, s3, 8(a2)
    vwmacc.vx       v12, t3, v2
    srli            s2, s1, 16
    vwmacc.vx       v10, t2, v2
    srli            s4, s3, 16
    vwmacc.vx       v14, t4, v2
    addi            a2, a2, 16

    vle16.v         v2, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2

    vwmacc.vx       v8, s1, v4
    lwd             t1, t3, 0(a2)
    vwmacc.vx       v12, s3, v4
    srli            t2, t1, 16
    vwmacc.vx       v10, s2, v4
    srli            t4, t3, 16
    vwmacc.vx       v14, s4, v4

    addi            t6, t6, -1
    bnez            t6, packnx4_k2

packnx4_end:
    vsetvli         zero, zero, e32, m2
    vse32.v         v8, (a0)
    add             a0, a0, t5
    vse32.v         v10, (a0)
    add             a0, a0, t5
    vse32.v         v12, (a0)
    add             a0, a0, t5
    vse32.v         v14, (a0)
    add             a0, a0, t5

packnx2_start:
    andi            a6, a7, 2       // s1 = bool_n2
    beqz            a6, packnx1_start  // if n2==0, jump to packnx1

    vsetvli         zero, t0, e16, m1
    vmv.v.x         v8, zero
    vmv.v.x         v9, zero
    vmv.v.x         v10, zero
    vmv.v.x         v11, zero

    mv              a5, a1  // kernel origin addr
    // pre-load kernel matrix
    vle16.v         v2, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2
    // pre-load input matrix
    lh              t1, 0(a2)
    lh              t2, 2(a2)

    srai            t6, a3, 1   // k2

packnx2_k2:
    vle16.v         v4, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2

    vwmacc.vx       v8, t1, v2
    lh              s1, 4(a2)
    vwmacc.vx       v10, t2, v2
    lh              s2, 6(a2)
    addi            a2, a2, 8

    vle16.v         v2, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2

    vwmacc.vx       v8, s1, v4
    lh              t1, 0(a2)
    vwmacc.vx       v10, s2, v4
    lh              t2, 2(a2)

    addi            t6, t6, -1
    bnez            t6, packnx2_k2

packnx2_end:
    vsetvli         zero, zero, e32, m2
    vse32.v         v8, (a0)
    add             a0, a0, t5
    vse32.v         v10, (a0)
    add             a0, a0, t5

packnx1_start:
    andi            a6, a7, 1       // s1 = bool_n1
    beqz            a6, packn_end  // if n1==0, jump to packn_end

    vsetvli         zero, t0, e16, m1
    vmv.v.x         v8, zero
    vmv.v.x         v9, zero

    mv              a5, a1  // kernel origin addr
    // pre-load kernel matrix
    vle16.v         v2, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2
    // pre-load input matrix
    lh              t1, 0(a2)

    srai            t6, a3, 1   // k2

packnx1_k2:
    vle16.v         v4, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2

    vwmacc.vx       v8, t1, v2
    lh              s1, 2(a2)
    addi            a2, a2, 4

    vle16.v         v2, (a5)
    add             a5, a5, t0  // kernel_ptr += packn * 2

    vwmacc.vx       v8, s1, v4
    lh              t1, 0(a2)

    addi            t6, t6, -1
    bnez            t6, packnx1_k2

packnx1_end:
    vsetvli         zero, zero, e32, m2
    vse32.v         v8, (a0)
    add             a0, a0, t5

packn_end:
    ld              s1, 0(sp)
    ld              s2, 8(sp)
    ld              s3, 16(sp)
    ld              s4, 24(sp)
    addi            sp, sp, 32

    ret
    .end
