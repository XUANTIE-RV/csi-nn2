/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

    void gemm_int4_dot_ncxhwx_12xpackn(const int8_t *output,
                                       const int8_t *kernel,
                                       const int8_t *input,
                                       const int32_t *bias,
                                       int k,          // maxtrix A col / maxtrix B row
                                       int n,          // maxtrix B col
                                       int32_t out_zp,
                                       int32_t *mult,
                                       int32_t *shift)

    Algorithm works as follows:
        (1) perform matrix-multiplication [packn, k] x [k, n] = [packn, n]
            ...

    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr
        a4: k [kernel_size]
        a5: n [out_hw]
        a6: out_zp
        a7: mult addr
        s0: shift addr

        t0 = packn/2 * 4  maintenance kernel_addr
        s7 = tmp variable
        s8 = k8(k2)  input_channel dim loop count
        s9 = kernel data addr
        s10 = n12
        s11 = n_tail

        t1-t6: hold input data
        s1-s6: hold input data

        v2-v3:   acc initial = bias
        v4-v7:   hold kernel data
        v8-v19:  fisrt packn line acc
        v20-v31: second packn line acc

 *************************************************************************************************/
    .file           "gemm_int4_dot_ncxhwx.S"
    .section        .text.gemm_int4_dat_ncxhwx_12xpackn, "ax", @progbits
    .align          5
    .global         gemm_int4_dat_ncxhwx_12xpackn
    .type           gemm_int4_dat_ncxhwx_12xpackn, @function

.macro GEMM_INT4_DOT_NCXHWX_REQUANTIZE v_dst
    vsetvli         zero, s7, e32, m2
    vmulh.vv        \v_dst, \v_dst, v4  // * mult
    vssra.vv        \v_dst, \v_dst, v6  // shift
    vadd.vx         \v_dst, \v_dst, a6  // + out_zp
    vsetvli         zero, s7, e16, m1
    vnclip.wi	    v0, \v_dst, 0
    vsetvli         zero, s7, e8, mf2
    vnclip.wi	    v1, v0, 0
    vsetvli         zero, s8, e8, mf4
    vpnclip.wx      \v_dst, v1, zero

.endm

gemm_int4_dat_ncxhwx_12xpackn:
    addi            sp, sp, -96
    sd              s0, 0(sp)
    sd              s1, 8(sp)
    sd              s2, 16(sp)
    sd              s3, 24(sp)
    sd              s4, 32(sp)
    sd              s5, 40(sp)
    sd              s6, 48(sp)
    sd              s7, 56(sp)
    sd              s8, 64(sp)
    sd              s9, 72(sp)
    sd              s10, 80(sp)
    sd              s11, 88(sp)

    ld              s0, 96(sp)

    csrr            t0, vlenb   // t0 = vlen/8 = packn/2 * 4 = 16
    slli            t0, t0, 1   // t0 = packn * 4 = 32
    srai            s7, t0, 2   // t1 = packn = 8
    vsetvli         zero, s7, e32, m2

    li              s7, 12
    divw            s10, a5, s7  // s10 = n12
    remw            s11, a5, s7  // s11 = n % 12 (n_tail)

    vle32.v         v2, (a3)    // bias

    beqz            s10, packnx8_start  // if n12==0, jump to packnx8

packnx12_start:
    vmv.v.v         v8, v2
    vmv.v.v         v10, v2
    vmv.v.v         v12, v2
    vmv.v.v         v14, v2
    vmv.v.v         v16, v2
    vmv.v.v         v18, v2
    vmv.v.v         v20, v2
    vmv.v.v         v22, v2
    vmv.v.v         v24, v2
    vmv.v.v         v26, v2
    vmv.v.v         v28, v2
    vmv.v.v         v30, v2

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn

    // pre-load input_data
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)
    lwd             t5, t6, 16(a2)

    srai            s8, a4, 3   // k8(k2)
    addi            s8, s8, -1  // k8(k2)_end
    beqz            s8, packnx12_k2_end

packnx12_k2:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    vmaqa.vx        v10, t2, v4
    vmaqa.vx        v12, t3, v4
    lwd             s1, s2, 24(a2)
    addi            a2, a2, 32
    vmaqa.vx        v14, t4, v4
    vmaqa.vx        v16, t5, v4
    lwd             s3, s4, 0(a2)
    lwd             s5, s6, 8(a2)
    vmaqa.vx        v18, t6, v4
    vmaqa.vx        v20, s1, v4
    vmaqa.vx        v22, s2, v4
    lwd             t1, t2, 16(a2)
    lwd             t3, t4, 24(a2)
    addi            a2, a2, 32
    vmaqa.vx        v24, s3, v4
    vmaqa.vx        v26, s4, v4
    lwd             t5, t6, 0(a2)
    vmaqa.vx        v28, s5, v4
    vmaqa.vx        v30, s6, v4

    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v6
    vmaqa.vx        v10, t2, v6
    lwd             s1, s2, 8(a2)
    lwd             s3, s4, 16(a2)
    vmaqa.vx        v12, t3, v6
    vmaqa.vx        v14, t4, v6
    lwd             s5, s6, 24(a2)
    addi            a2, a2, 32
    vmaqa.vx        v16, t5, v6
    vmaqa.vx        v18, t6, v6
    lwd             t1, t2, 0(a2)
    vmaqa.vx        v20, s1, v6
    vmaqa.vx        v22, s2, v6
    lwd             t3, t4, 8(a2)
    vmaqa.vx        v24, s3, v6
    vmaqa.vx        v26, s4, v6
    lwd             t5, t6, 16(a2)
    vmaqa.vx        v28, s5, v6
    vmaqa.vx        v30, s6, v6

    addi            s8, s8, -1
    bnez            s8, packnx12_k2

packnx12_k2_end:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    vmaqa.vx        v10, t2, v4
    vmaqa.vx        v12, t3, v4
    lwd             s1, s2, 24(a2)
    addi            a2, a2, 32
    vmaqa.vx        v14, t4, v4
    vmaqa.vx        v16, t5, v4
    lwd             s3, s4, 0(a2)
    lwd             s5, s6, 8(a2)
    vmaqa.vx        v18, t6, v4
    vmaqa.vx        v20, s1, v4
    vmaqa.vx        v22, s2, v4
    lwd             t1, t2, 16(a2)
    lwd             t3, t4, 24(a2)
    addi            a2, a2, 32
    vmaqa.vx        v24, s3, v4
    vmaqa.vx        v26, s4, v4
    lwd             t5, t6, 0(a2)
    vmaqa.vx        v28, s5, v4
    vmaqa.vx        v30, s6, v4

    vmaqa.vx        v8, t1, v6
    vmaqa.vx        v10, t2, v6
    lwd             s1, s2, 8(a2)
    lwd             s3, s4, 16(a2)
    vmaqa.vx        v12, t3, v6
    vmaqa.vx        v14, t4, v6
    lwd             s5, s6, 24(a2)
    addi            a2, a2, 32
    vmaqa.vx        v16, t5, v6
    vmaqa.vx        v18, t6, v6
    vmaqa.vx        v20, s1, v6
    vmaqa.vx        v22, s2, v6
    vmaqa.vx        v24, s3, v6
    vmaqa.vx        v26, s4, v6
    vmaqa.vx        v28, s5, v6
    vmaqa.vx        v30, s6, v6

packnx12_post:
    srai            s7, t0, 2
    vsetvli         zero, s7, e32, m2   // set vl = 8
    vle32.v         v4, (a7)    // mult
    srai            s8, s7, 1
    vle32.v         v6, (s0)    // shift
    vxor.vi         v6, v6, -1

    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v8
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v10
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v12
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v14
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v16
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v18
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v20
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v22
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v24
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v26
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v28
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v30

packnx12_end:
    vse8.v          v8, (a0)
    add             a0, a0, s8
    vse8.v          v10, (a0)
    add             a0, a0, s8
    vse8.v          v12, (a0)
    add             a0, a0, s8
    vse8.v          v14, (a0)
    add             a0, a0, s8
    vse8.v          v16, (a0)
    add             a0, a0, s8
    vse8.v          v18, (a0)
    add             a0, a0, s8
    vse8.v          v20, (a0)
    add             a0, a0, s8
    vse8.v          v22, (a0)
    add             a0, a0, s8
    vse8.v          v24, (a0)
    add             a0, a0, s8
    vse8.v          v26, (a0)
    add             a0, a0, s8
    vse8.v          v28, (a0)
    add             a0, a0, s8
    vse8.v          v30, (a0)
    add             a0, a0, s8

    vsetvli         zero, s7, e32, m2
    addi            s10, s10, -1
    bnez            s10, packnx12_start

packnx8_start:
    andi            s10, s11, 8       // s1 = bool_n8
    beqz            s10, packnx4_start  // if n8==0, jump to packnx4

    vmv.v.v         v8, v2
    vmv.v.v         v10, v2
    vmv.v.v         v12, v2
    vmv.v.v         v14, v2
    vmv.v.v         v16, v2
    vmv.v.v         v18, v2
    vmv.v.v         v20, v2
    vmv.v.v         v22, v2

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn
    // pre-load input_data
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)

    srai            s8, a4, 3   // k2
    addi            s8, s8, -1  // k8(k2)_end
    beqz            s8, packnx8_k2_end

packnx8_k2:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    vmaqa.vx        v10, t2, v4
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v4
    vmaqa.vx        v14, t4, v4
    vmaqa.vx        v16, s1, v4
    addi            a2, a2, 32
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)
    vmaqa.vx        v18, s2, v4
    vmaqa.vx        v20, s3, v4
    vmaqa.vx        v22, s4, v4

    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v6
    vmaqa.vx        v10, t2, v6
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v6
    vmaqa.vx        v14, t4, v6
    vmaqa.vx        v16, s1, v6
    addi            a2, a2, 32
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)
    vmaqa.vx        v18, s2, v6
    vmaqa.vx        v20, s3, v6
    vmaqa.vx        v22, s4, v6

    addi            s8, s8, -1
    bnez            s8, packnx8_k2

packnx8_k2_end:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    vmaqa.vx        v10, t2, v4
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v4
    vmaqa.vx        v14, t4, v4
    vmaqa.vx        v16, s1, v4
    addi            a2, a2, 32
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)
    vmaqa.vx        v18, s2, v4
    vmaqa.vx        v20, s3, v4
    vmaqa.vx        v22, s4, v4

    vmaqa.vx        v8, t1, v6
    vmaqa.vx        v10, t2, v6
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v6
    vmaqa.vx        v14, t4, v6
    vmaqa.vx        v16, s1, v6
    addi            a2, a2, 32
    vmaqa.vx        v18, s2, v6
    vmaqa.vx        v20, s3, v6
    vmaqa.vx        v22, s4, v6


packnx8_post:
    srai            s7, t0, 2
    vsetvli         zero, s7, e32, m2   // set vl = 8
    vle32.v         v4, (a7)    // mult
    srai            s8, s7, 1
    vle32.v         v6, (s0)    // shift
    vxor.vi         v6, v6, -1

    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v8
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v10
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v12
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v14
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v16
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v18
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v20
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v22

packnx8_end:
    vse8.v          v8, (a0)
    add             a0, a0, s8
    vse8.v          v10, (a0)
    add             a0, a0, s8
    vse8.v          v12, (a0)
    add             a0, a0, s8
    vse8.v          v14, (a0)
    add             a0, a0, s8
    vse8.v          v16, (a0)
    add             a0, a0, s8
    vse8.v          v18, (a0)
    add             a0, a0, s8
    vse8.v          v20, (a0)
    add             a0, a0, s8
    vse8.v          v22, (a0)
    add             a0, a0, s8

packnx4_start:
    andi            s10, s11, 4       // s1 = bool_n4
    beqz            s10, packnx2_start  // if n4==0, jump to packnx2

    vmv.v.v         v8, v2
    vmv.v.v         v10, v2
    vmv.v.v         v12, v2
    vmv.v.v         v14, v2

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn
    // pre-load input_data
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)

    srai            s8, a4, 3   // k2
    addi            s8, s8, -1  // k8(k2)_end
    beqz            s8, packnx4_k2_end

packnx4_k2:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    vmaqa.vx        v10, t2, v4
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v4
    vmaqa.vx        v14, t4, v4
    addi            a2, a2, 32

    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, s1, v6
    vmaqa.vx        v10, s2, v6
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)
    vmaqa.vx        v12, s3, v6
    vmaqa.vx        v14, s4, v6

    addi            s8, s8, -1
    bnez            s8, packnx4_k2

packnx4_k2_end:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    vmaqa.vx        v10, t2, v4
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v4
    vmaqa.vx        v14, t4, v4
    addi            a2, a2, 32

    vmaqa.vx        v8, s1, v6
    vmaqa.vx        v10, s2, v6
    vmaqa.vx        v12, s3, v6
    vmaqa.vx        v14, s4, v6

packnx4_post:
    srai            s7, t0, 2
    vsetvli         zero, s7, e32, m2   // set vl = 8
    vle32.v         v4, (a7)    // mult
    srai            s8, s7, 1
    vle32.v         v6, (s0)    // shift
    vxor.vi         v6, v6, -1

    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v8
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v10
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v12
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v14

packnx4_end:
    vse8.v          v8, (a0)
    add             a0, a0, s8
    vse8.v          v10, (a0)
    add             a0, a0, s8
    vse8.v          v12, (a0)
    add             a0, a0, s8
    vse8.v          v14, (a0)
    add             a0, a0, s8

packnx2_start:
    andi            s10, s11, 2       // s1 = bool_n2
    beqz            s10, packnx1_start  // if n2==0, jump to packnx1

    vsetvli         zero, s7, e32, m2
    vmv.v.v         v8, v2
    vmv.v.v         v10, v2

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn
    // pre-load input_data
    lwd             t1, t2, 0(a2)

    srai            s8, a4, 3   // k2
    addi            s8, s8, -1  // k8(k2)_end
    beqz            s8, packnx2_k2_end

packnx2_k2:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    lwd             s1, s2, 8(a2)
    vmaqa.vx        v10, t2, v4
    addi            a2, a2, 16

    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, s1, v6
    lwd             t1, t2, 0(a2)
    vmaqa.vx        v10, s2, v6

    addi            s8, s8, -1
    bnez            s8, packnx2_k2

packnx2_k2_end:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    lwd             s1, s2, 8(a2)
    vmaqa.vx        v10, t2, v4
    addi            a2, a2, 16

    vmaqa.vx        v8, s1, v6
    vmaqa.vx        v10, s2, v6

packnx2_post:
    srai            s7, t0, 2
    vsetvli         zero, s7, e32, m2   // set vl = 8
    vle32.v         v4, (a7)    // mult
    srai            s8, s7, 1
    vle32.v         v6, (s0)    // shift
    vxor.vi         v6, v6, -1

    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v8
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v10

packnx2_end:
    vse8.v          v8, (a0)
    add             a0, a0, s8
    vse8.v          v10, (a0)
    add             a0, a0, s8

packnx1_start:
    andi            s10, s11, 1       // s1 = bool_n1
    beqz            s10, packn_end  // if n1==0, jump to packn_end

    vsetvli         zero, s7, e32, m2
    vmv.v.v         v8, v2

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn
    // pre-load input_data
    lw              t1, 0(a2)

    srai            s8, a4, 3   // k2
    addi            s8, s8, -1  // k8(k2)_end
    beqz            s8, packnx1_k2_end

packnx1_k2:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    lw              s1, 4(a2)
    addi            a2, a2, 8

    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, s1, v6
    lw              t1, 0(a2)

    addi            s8, s8, -1
    bnez            s8, packnx1_k2

packnx1_k2_end:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    lw              s1, 4(a2)
    addi            a2, a2, 8

    vmaqa.vx        v8, s1, v6

packnx1_post:
    srai            s7, t0, 2
    vsetvli         zero, s7, e32, m2   // set vl = 8
    vle32.v         v4, (a7)    // mult
    srai            s8, s7, 1
    vle32.v         v6, (s0)    // shift
    vxor.vi         v6, v6, -1

    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v8

packnx1_end:
    vse8.v          v8, (a0)
    add             a0, a0, s8

packn_end:
    ld              s0, 0(sp)
    ld              s1, 8(sp)
    ld              s2, 16(sp)
    ld              s3, 24(sp)
    ld              s4, 32(sp)
    ld              s5, 40(sp)
    ld              s6, 48(sp)
    ld              s7, 56(sp)
    ld              s8, 64(sp)
    ld              s9, 72(sp)
    ld              s10, 80(sp)
    ld              s11, 88(sp)
    addi            sp, sp, 96

    ret


/**************************************************************************************************

    void gemm_int4_dot_ncxhwx_8xpackn(const int8_t *output,
                                      const int8_t *kernel,
                                      const int8_t *input,
                                      const int32_t *bias,
                                      int k,          // maxtrix A col / maxtrix B row
                                      int n,          // maxtrix B col
                                      int32_t out_zp,
                                      int32_t *mult,
                                      int32_t *shift)

    Algorithm works as follows:
        (1) perform matrix-multiplication [packn, k] x [k, n] = [packn, n]
            ...

    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr
        a4: k [kernel_size]
        a5: n [out_hw]
        a6: out_zp
        a7: mult addr
        s0: shift addr

        t0 = packn/2 * 4  maintenance kernel_addr
        s7 = tmp variable
        s8 = k8(k2)  input_channel dim loop count
        s9 = kernel data addr
        s10 = n8 / n4 / n2 / n1

        t1-t4: hold input data
        s1-s4: hold input data

        v2-v3:   acc initial = bias
        v4-v7:   hold kernel data
        v8-v19:  fisrt packn line acc
        v20-v31: second packn line acc

 *************************************************************************************************/
    .section        .text.gemm_int4_dot_ncxhwx_8xpackn, "ax", @progbits
    .align          5
    .global         gemm_int4_dot_ncxhwx_8xpackn
    .type           gemm_int4_dot_ncxhwx_8xpackn, @function

gemm_int4_dot_ncxhwx_8xpackn:
    addi            sp, sp, -72
    sd              s0, 0(sp)
    sd              s1, 8(sp)
    sd              s2, 16(sp)
    sd              s3, 24(sp)
    sd              s4, 32(sp)
    sd              s7, 40(sp)
    sd              s8, 48(sp)
    sd              s9, 56(sp)
    sd              s10, 64(sp)

    ld              s0, 72(sp)

    csrr            t0, vlenb   // t0 = vlen/8 = packn/2 * 4 = 16
    slli            t0, t0, 1   // t0 = packn * 4 = 32
    srai            s7, t0, 2   // t1 = packn = 8
    vsetvli         zero, s7, e32, m2

    srai            s10, a5, 3  // s10 = n8

    vle32.v         v2, (a3)    // bias

    beqz            s10, packnx4_start_1  // if n8==0, jump to packnx4

packnx8_start_1:
    vsetvli         zero, s7, e32, m2

    vmv.v.v         v8, v2
    vmv.v.v         v10, v2
    vmv.v.v         v12, v2
    vmv.v.v         v14, v2
    vmv.v.v         v16, v2
    vmv.v.v         v18, v2
    vmv.v.v         v20, v2
    vmv.v.v         v22, v2

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn
    // pre-load input_data
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)

    srai            s8, a4, 3   // k2
    addi            s8, s8, -1  // k2_end
    beqz            s8, packnx8_k2_end_1

packnx8_k2_1:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    vmaqa.vx        v10, t2, v4
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v4
    vmaqa.vx        v14, t4, v4
    vmaqa.vx        v16, s1, v4
    addi            a2, a2, 32
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)
    vmaqa.vx        v18, s2, v4
    vmaqa.vx        v20, s3, v4
    vmaqa.vx        v22, s4, v4

    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v6
    vmaqa.vx        v10, t2, v6
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v6
    vmaqa.vx        v14, t4, v6
    vmaqa.vx        v16, s1, v6
    addi            a2, a2, 32
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)
    vmaqa.vx        v18, s2, v6
    vmaqa.vx        v20, s3, v6
    vmaqa.vx        v22, s4, v6

    addi            s8, s8, -1
    bnez            s8, packnx8_k2_1

packnx8_k2_end_1:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    vmaqa.vx        v10, t2, v4
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v4
    vmaqa.vx        v14, t4, v4
    vmaqa.vx        v16, s1, v4
    addi            a2, a2, 32
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)
    vmaqa.vx        v18, s2, v4
    vmaqa.vx        v20, s3, v4
    vmaqa.vx        v22, s4, v4

    vmaqa.vx        v8, t1, v6
    vmaqa.vx        v10, t2, v6
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v6
    vmaqa.vx        v14, t4, v6
    vmaqa.vx        v16, s1, v6
    addi            a2, a2, 32
    vmaqa.vx        v18, s2, v6
    vmaqa.vx        v20, s3, v6
    vmaqa.vx        v22, s4, v6

packnx8_post_1:
    vsetvli         zero, s7, e32, m2   // set vl = 8
    vle32.v         v4, (a7)    // mult
    srai            s8, s7, 1
    vle32.v         v6, (s0)    // shift
    vxor.vi         v6, v6, -1

    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v8
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v10
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v12
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v14
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v16
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v18
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v20
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v22

packnx8_end_1:
    vse8.v          v8, (a0)
    add             a0, a0, s8
    vse8.v          v10, (a0)
    add             a0, a0, s8
    vse8.v          v12, (a0)
    add             a0, a0, s8
    vse8.v          v14, (a0)
    add             a0, a0, s8
    vse8.v          v16, (a0)
    add             a0, a0, s8
    vse8.v          v18, (a0)
    add             a0, a0, s8
    vse8.v          v20, (a0)
    add             a0, a0, s8
    vse8.v          v22, (a0)
    add             a0, a0, s8

    addi            s10, s10, -1
    bnez            s10, packnx8_start_1

packnx4_start_1:
    andi            s10, a5, 4       // s1 = bool_n4
    beqz            s10, packnx2_start_1  // if n4==0, jump to packnx2

    vsetvli         zero, s7, e32, m2

    vmv.v.v         v8, v2
    vmv.v.v         v10, v2
    vmv.v.v         v12, v2
    vmv.v.v         v14, v2

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn
    // pre-load input_data
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)

    srai            s8, a4, 3   // k2
    addi            s8, s8, -1  // k2_end
    beqz            s8, packnx4_k2_end_1

packnx4_k2_1:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    vmaqa.vx        v10, t2, v4
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v4
    vmaqa.vx        v14, t4, v4
    addi            a2, a2, 32

    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, s1, v6
    vmaqa.vx        v10, s2, v6
    lwd             t1, t2, 0(a2)
    lwd             t3, t4, 8(a2)
    vmaqa.vx        v12, s3, v6
    vmaqa.vx        v14, s4, v6

    addi            s8, s8, -1
    bnez            s8, packnx4_k2_1

packnx4_k2_end_1:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    vmaqa.vx        v10, t2, v4
    lwd             s1, s2, 16(a2)
    lwd             s3, s4, 24(a2)
    vmaqa.vx        v12, t3, v4
    vmaqa.vx        v14, t4, v4
    addi            a2, a2, 32

    vmaqa.vx        v8, s1, v6
    vmaqa.vx        v10, s2, v6
    vmaqa.vx        v12, s3, v6
    vmaqa.vx        v14, s4, v6

packnx4_post_1:
    vsetvli         zero, s7, e32, m2   // set vl = 8
    vle32.v         v4, (a7)    // mult
    srai            s8, s7, 1
    vle32.v         v6, (s0)    // shift
    vxor.vi         v6, v6, -1

    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v8
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v10
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v12
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v14

packnx4_end_1:
    vse8.v          v8, (a0)
    add             a0, a0, s8
    vse8.v          v10, (a0)
    add             a0, a0, s8
    vse8.v          v12, (a0)
    add             a0, a0, s8
    vse8.v          v14, (a0)
    add             a0, a0, s8

packnx2_start_1:
    andi            s10, a5, 2       // s1 = bool_n2
    beqz            s10, packnx1_start_1  // if n2==0, jump to packnx1

    vsetvli         zero, s7, e32, m2
    vmv.v.v         v8, v2
    vmv.v.v         v10, v2

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn
    // pre-load input_data
    lwd             t1, t2, 0(a2)

    srai            s8, a4, 3   // k2
    addi            s8, s8, -1  // k2_end
    beqz            s8, packnx2_k2_end_1

packnx2_k2_1:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    lwd             s1, s2, 8(a2)
    vmaqa.vx        v10, t2, v4
    addi            a2, a2, 16

    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, s1, v6
    lwd             t1, t2, 0(a2)
    vmaqa.vx        v10, s2, v6

    addi            s8, s8, -1
    bnez            s8, packnx2_k2_1

packnx2_k2_end_1:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    lwd             s1, s2, 8(a2)
    vmaqa.vx        v10, t2, v4
    addi            a2, a2, 16

    vmaqa.vx        v8, s1, v6
    vmaqa.vx        v10, s2, v6

packnx2_post_1:
    vsetvli         zero, s7, e32, m2   // set vl = 8
    vle32.v         v4, (a7)    // mult
    srai            s8, s7, 1
    vle32.v         v6, (s0)    // shift
    vxor.vi         v6, v6, -1

    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v8
    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v10

packnx2_end_1:
    vse8.v          v8, (a0)
    add             a0, a0, s8
    vse8.v          v10, (a0)
    add             a0, a0, s8

packnx1_start_1:
    andi            s10, a5, 1       // s1 = bool_n1
    beqz            s10, packn_end_1  // if n1==0, jump to packn_end

    vsetvli         zero, s7, e32, m2
    vmv.v.v         v8, v2

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn
    // pre-load input_data
    lw              t1, 0(a2)

    srai            s8, a4, 3   // k2
    addi            s8, s8, -1  // k2_end
    beqz            s8, packnx1_k2_end_1

packnx1_k2_1:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    lw              s1, 4(a2)
    addi            a2, a2, 8

    vle32.v         v4, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, s1, v6
    lw              t1, 0(a2)

    addi            s8, s8, -1
    bnez            s8, packnx1_k2_1

packnx1_k2_end_1:
    vle32.v         v6, (s9)
    add             s9, s9, t0  // +packn

    vmaqa.vx        v8, t1, v4
    lw              s1, 4(a2)
    addi            a2, a2, 8

    vmaqa.vx        v8, s1, v6

packnx1_post_1:
    vsetvli         zero, s7, e32, m2   // set vl = 8
    vle32.v         v4, (a7)    // mult
    srai            s8, s7, 1
    vle32.v         v6, (s0)    // shift
    vxor.vi         v6, v6, -1

    GEMM_INT4_DOT_NCXHWX_REQUANTIZE v8

packnx1_end_1:
    vse8.v          v8, (a0)
    add             a0, a0, s8

packn_end_1:
    ld              s0, 0(sp)
    ld              s1, 8(sp)
    ld              s2, 16(sp)
    ld              s3, 24(sp)
    ld              s4, 32(sp)
    ld              s7, 40(sp)
    ld              s8, 48(sp)
    ld              s9, 56(sp)
    ld              s10, 64(sp)
    addi            sp, sp, 72

    ret
    .end
