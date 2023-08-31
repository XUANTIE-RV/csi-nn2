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

    void gemm_int8_ncxhwx_4xpack2n(const int8_t *output,
                                   const int8_t *kernel,
                                   const int8_t *input,
                                   const int32_t *bias,
                                   int m,          // maxtrix A row
                                   int k,          // maxtrix A col / maxtrix B row
                                   int n,          // maxtrix B col
                                   int32_t out_zp,
                                   int32_t *mult,
                                   int32_t *shift)

    Algorithm works as follows:
        (1) perform matrix-multiplication [packn, k] x [k, n] = [packn, n]
        (2) without dot instr: vwmul + vwmacc

    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr
        a4: m [pack2n]
        a5: k [kernel_size]
        a6: n [out_hw]
        a7: out_zp

        a3 = next packn line output

        t5: mult addr
        t6: shift addr

        t0: const 1
        s0: tmp k2 / k1
        s9: kernel data addr
        s10: n4 / n2 / n1

        t1-t4: hold input data
        s1-s4: hold input data

 *************************************************************************************************/
    .file           "gemm_int8_ncxhwx.S"
    .section        .text.gemm_int8_ncxhwx_4xpack2n, "ax", @progbits
    .align          5
    .global         gemm_int8_ncxhwx_4xpack2n
    .type           gemm_int8_ncxhwx_4xpack2n, @function

.macro GEMM_INT8_NCXHWX_REQUANTIZE v_dst
    vsetvli         zero, a4, e32, m4
    vmulh.vv        \v_dst, \v_dst, v4  // * mult
    vssra.vv        \v_dst, \v_dst, v8  // shift
    vadd.vx         \v_dst, \v_dst, a7  // + out_zp
    vsetvli         zero, a4, e16, m2
    vnclip.wi	    v12, \v_dst, 0
    vsetvli         zero, a4, e8, m1
    vnclip.wi	    \v_dst, v12, 0
.endm

gemm_int8_ncxhwx_4xpack2n:
    addi            sp, sp, -56
    sd              s0, 0(sp)
    sd              s1, 8(sp)
    sd              s2, 16(sp)
    sd              s3, 24(sp)
    sd              s4, 32(sp)
    sd              s9, 40(sp)
    sd              s10, 48(sp)

    ld              t5, 56(sp)
    ld              t6, 64(sp)

    vsetvli         zero, a4, e32, m4   // vl = pack2n = 16

    srai            s10, a6, 2 // s10 = n4

    vle32.v         v0, (a3)

    srai            t0, a4, 1   // packn
    mul             t0, t0, a6  // packn * n
    add             a3, a0, t0  // a3[out1_addr] = out0_addr + packn * n

    li              t0, 1
    beqz            s10, pack2nx2_start // if n4==0, jump to pack2nx2

pack2nx4_start:
    vsetvli         zero, a4, e32, m4
    vmv.v.v         v16, v0
    vmv.v.v         v20, v0
    vmv.v.v         v24, v0
    vmv.v.v         v28, v0

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +pack2n
    // pre-load input_data
    lb              t1, 0(a2)
    lb              t2, 1(a2)
    lb              t3, 2(a2)
    lb              t4, 3(a2)

    srai            s0, a5, 1   // k2
    beqz            s0, pack2nx4_k1
    addi            s0, s0, -1  // k2_end
    beqz            s0, pack2nx4_k2_end

pack2nx4_k2:
    vsetvli         zero, a4, e8, m1
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 4(a2)
    vwmul.vx        v8, v4, t2
    vwmul.vx        v10, v4, t3
    lb              s2, 5(a2)
    vwmul.vx        v12, v4, t4

    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6
    lb              s3, 6(a2)
    vwmacc.vx       v20, t0, v8
    vwmacc.vx       v24, t0, v10
    lb              s4, 7(a2)
    vwmacc.vx       v28, t0, v12
    addi            a2, a2, 8   // input_data += 8

    vsetvli         zero, a4, e8, m1
    vle8.v          v4, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v5, s1
    lb              t1, 0(a2)
    vwmul.vx        v8, v5, s2
    vwmul.vx        v10, v5, s3
    lb              t2, 1(a2)
    vwmul.vx        v12, v5, s4

    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6
    lb              t3, 2(a2)
    vwmacc.vx       v20, t0, v8
    vwmacc.vx       v24, t0, v10
    lb              t4, 3(a2)
    vwmacc.vx       v28, t0, v12

    addi            s0, s0, -1
    bnez            s0, pack2nx4_k2

pack2nx4_k2_end:
    vsetvli         zero, a4, e8, m1
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 4(a2)
    vwmul.vx        v8, v4, t2
    vwmul.vx        v10, v4, t3
    lb              s2, 5(a2)
    vwmul.vx        v12, v4, t4

    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6
    lb              s3, 6(a2)
    vwmacc.vx       v20, t0, v8
    vwmacc.vx       v24, t0, v10
    lb              s4, 7(a2)
    vwmacc.vx       v28, t0, v12
    addi            a2, a2, 8   // input_data += 8

    vsetvli         zero, a4, e8, m1

    vwmul.vx        v6, v5, s1
    vwmul.vx        v8, v5, s2
    vwmul.vx        v10, v5, s3
    vwmul.vx        v12, v5, s4

    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8
    vwmacc.vx       v24, t0, v10
    vwmacc.vx       v28, t0, v12

    andi            s0, a5, 1   // k1
    beqz            s0, pack2nx4_post

    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +pack2n
    // pre-load input_data
    lb              t1, 0(a2)
    lb              t2, 1(a2)
    lb              t3, 2(a2)
    lb              t4, 3(a2)

pack2nx4_k1:
    vsetvli         zero, a4, e8, m1
    vwmul.vx        v6, v4, t1
    vwmul.vx        v8, v4, t2
    vwmul.vx        v10, v4, t3
    vwmul.vx        v12, v4, t4
    addi            a2, a2, 4   // input_data += 4

    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8
    vwmacc.vx       v24, t0, v10
    vwmacc.vx       v28, t0, v12

pack2nx4_post:
    vsetvli         zero, a4, e32, m4
    vle32.v         v4, (t5)    // mult
    vle32.v         v8, (t6)    // shift
    vxor.vi         v8, v8, -1

    GEMM_INT8_NCXHWX_REQUANTIZE v16
    GEMM_INT8_NCXHWX_REQUANTIZE v20
    GEMM_INT8_NCXHWX_REQUANTIZE v24
    GEMM_INT8_NCXHWX_REQUANTIZE v28

pack2nx4_end:
    srai            s0, a4, 1   // s0 = packn
    vsetvli         zero, s0, e8, m1

    vse8.v          v16, (a0)
    add             a0, a0, s0
    vse8.v          v20, (a0)
    add             a0, a0, s0
    vse8.v          v24, (a0)
    add             a0, a0, s0
    vse8.v          v28, (a0)
    add             a0, a0, s0

    vslidedown.vx   v16, v16, s0
    vslidedown.vx   v20, v20, s0
    vslidedown.vx   v24, v24, s0
    vslidedown.vx   v28, v28, s0

    vse8.v          v16, (a3)
    add             a3, a3, s0
    vse8.v          v20, (a3)
    add             a3, a3, s0
    vse8.v          v24, (a3)
    add             a3, a3, s0
    vse8.v          v28, (a3)
    add             a3, a3, s0

    addi            s10, s10, -1
    bnez            s10, pack2nx4_start

pack2nx2_start:
    andi            s10, a6, 2      // s10 = bool_n2
    beqz            s10, pack2nx1_start

    vsetvli         zero, a4, e32, m4
    vmv.v.v         v16, v0
    vmv.v.v         v20, v0

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +pack2n
    // pre-load input_data
    lb              t1, 0(a2)
    lb              t2, 1(a2)

    srai            s0, a5, 1   // k2
    beqz            s0, pack2nx2_k1
    addi            s0, s0, -1  // k2_end
    beqz            s0, pack2nx2_k2_end

pack2nx2_k2:
    vsetvli         zero, a4, e8, m1
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 2(a2)
    vwmul.vx        v8, v4, t2
    lb              s2, 3(a2)
    vsetvli         zero, a4, e16, m2
    addi            a2, a2, 4
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8

    vsetvli         zero, a4, e8, m1
    vle8.v          v4, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v5, s1
    lb              t1, 0(a2)
    vwmul.vx        v8, v5, s2
    lb              t2, 1(a2)
    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8

    addi            s0, s0, -1
    bnez            s0, pack2nx2_k2

pack2nx2_k2_end:
    vsetvli         zero, a4, e8, m1
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 2(a2)
    vwmul.vx        v8, v4, t2
    lb              s2, 3(a2)
    vsetvli         zero, a4, e16, m2
    addi            a2, a2, 4
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8

    vsetvli         zero, a4, e8, m1

    vwmul.vx        v6, v5, s1
    vwmul.vx        v8, v5, s2
    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8

    andi            s0, a5, 1   // k1
    beqz            s0, pack2nx2_post

    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +pack2n
    // pre-load input_data
    lb              t1, 0(a2)
    lb              t2, 1(a2)

pack2nx2_k1:
    vsetvli         zero, a4, e8, m1
    vwmul.vx        v6, v4, t1
    vwmul.vx        v8, v4, t2
    addi            a2, a2, 2   // input_data += 2
    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8

pack2nx2_post:
    vsetvli         zero, a4, e32, m4
    vle32.v         v4, (t5)    // mult
    vle32.v         v8, (t6)    // shift
    vxor.vi         v8, v8, -1

    GEMM_INT8_NCXHWX_REQUANTIZE v16
    GEMM_INT8_NCXHWX_REQUANTIZE v20

pack2nx2_end:
    srai            s0, a4, 1   // s0 = packn
    vsetvli         zero, s0, e8, m1

    vse8.v          v16, (a0)
    add             a0, a0, s0
    vse8.v          v20, (a0)
    add             a0, a0, s0

    vslidedown.vx   v16, v16, s0
    vslidedown.vx   v20, v20, s0

    vse8.v          v16, (a3)
    add             a3, a3, s0
    vse8.v          v20, (a3)
    add             a3, a3, s0

pack2nx1_start:
    andi            s10, a6, 1  // s10 = bool_n1
    beqz            s10, pack2n_end

    vsetvli         zero, a4, e32, m4
    vmv.v.v         v16, v0

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +pack2n
    // pre-load input_data
    lb              t1, 0(a2)

    srai            s0, a5, 1   // k2
    beqz            s0, pack2nx1_k1
    addi            s0, s0, -1  // k2_end
    beqz            s0, pack2nx1_k2_end

pack2nx1_k2:
    vsetvli         zero, a4, e8, m1
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 1(a2)
    vsetvli         zero, a4, e16, m2
    addi            a2, a2, 2
    vwmacc.vx       v16, t0, v6

    vsetvli         zero, a4, e8, m1
    vle8.v          v4, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v5, s1
    lb              t1, 0(a2)
    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6

    addi            s0, s0, -1
    bnez            s0, pack2nx1_k2

pack2nx1_k2_end:
    vsetvli         zero, a4, e8, m1
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 1(a2)
    vsetvli         zero, a4, e16, m2
    addi            a2, a2, 2
    vwmacc.vx       v16, t0, v6

    vsetvli         zero, a4, e8, m1

    vwmul.vx        v6, v5, s1
    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6

    andi            s0, a5, 1   // k1
    beqz            s0, pack2nx1_post

    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +pack2n
    // pre-load input_data
    lb              t1, 0(a2)

pack2nx1_k1:
    vsetvli         zero, a4, e8, m1
    vwmul.vx        v6, v4, t1
    addi            a2, a2, 1   // input_data += 1
    vsetvli         zero, a4, e16, m2
    vwmacc.vx       v16, t0, v6

pack2nx1_post:
    vsetvli         zero, a4, e32, m4
    vle32.v         v4, (t5)    // mult
    vle32.v         v8, (t6)    // shift
    vxor.vi         v8, v8, -1

    GEMM_INT8_NCXHWX_REQUANTIZE v16

pack2nx1_end:
    srai            s0, a4, 1   // s0 = packn
    vsetvli         zero, s0, e8, m1

    vse8.v          v16, (a0)
    add             a0, a0, s0

    vslidedown.vx   v16, v16, s0

    vse8.v          v16, (a3)
    add             a3, a3, s0

pack2n_end:
    ld              s0, 0(sp)
    ld              s1, 8(sp)
    ld              s2, 16(sp)
    ld              s3, 24(sp)
    ld              s4, 32(sp)
    ld              s9, 40(sp)
    ld              s10, 48(sp)
    addi            sp, sp, 56

    ret

 /**************************************************************************************************

    void gemm_int8_ncxhwx_4xpackn(const int8_t *output,
                                  const int8_t *kernel,
                                  const int8_t *input,
                                  const int32_t *bias,
                                  int m,          // maxtrix A row
                                  int k,          // maxtrix A col / maxtrix B row
                                  int n,          // maxtrix B col
                                  int32_t out_zp,
                                  int32_t *mult,
                                  int32_t *shift)

    Algorithm works as follows:
        (1) perform matrix-multiplication [packn, k] x [k, n] = [packn, n]
        (2) without dot instr: vwmul + vwmacc

    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr
        a4: m [packn or tail_packn]
        a5: k [kernel_size]
        a6: n [out_hw]
        a7: out_zp

        t5: mult addr
        t6: shift addr

        t0: const 1
        s0: tmp k2 / k1
        s9: kernel data addr
        s10: n4 / n2 / n1

        t1-t4: hold input data
        s1-s4: hold input data

 *************************************************************************************************/
    .section        .text.gemm_int8_ncxhwx_4xpackn, "ax", @progbits
    .align          5
    .global         gemm_int8_ncxhwx_4xpackn
    .type           gemm_int8_ncxhwx_4xpackn, @function

.macro GEMM_INT8_NCXHWX_REQUANTIZE_M2 v_dst
    vsetvli         zero, a4, e32, m2
    vmulh.vv        \v_dst, \v_dst, v4  // * mult
    vssra.vv        \v_dst, \v_dst, v8  // shift
    vadd.vx         \v_dst, \v_dst, a7  // + out_zp
    vsetvli         zero, a4, e16, m1
    vnclip.wi	    v12, \v_dst, 0
    vsetvli         zero, a4, e8, mf2
    vnclip.wi	    \v_dst, v12, 0
.endm


gemm_int8_ncxhwx_4xpackn:
    addi            sp, sp, -56
    sd              s0, 0(sp)
    sd              s1, 8(sp)
    sd              s2, 16(sp)
    sd              s3, 24(sp)
    sd              s4, 32(sp)
    sd              s9, 40(sp)
    sd              s10, 48(sp)

    ld              t5, 56(sp)
    ld              t6, 64(sp)

    vsetvli         zero, a4, e32, m2   // vl = packn = 8

    srai            s10, a6, 2 // s10 = n4

    vle32.v         v0, (a3)

    li              t0, 1
    beqz            s10, packnx2_start // if n4==0, jump to packnx2

packnx4_start:
    vsetvli         zero, a4, e32, m2
    vmv.v.v         v16, v0
    vmv.v.v         v20, v0
    vmv.v.v         v24, v0
    vmv.v.v         v28, v0

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +packn
    // pre-load input_data
    lb              t1, 0(a2)
    lb              t2, 1(a2)
    lb              t3, 2(a2)
    lb              t4, 3(a2)

    srai            s0, a5, 1   // k2
    beqz            s0, packnx4_k1
    addi            s0, s0, -1  // k2_end
    beqz            s0, packnx4_k2_end

packnx4_k2:
    vsetvli         zero, a4, e8, mf2
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 4(a2)
    vwmul.vx        v8, v4, t2
    vwmul.vx        v10, v4, t3
    lb              s2, 5(a2)
    vwmul.vx        v12, v4, t4

    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6
    lb              s3, 6(a2)
    vwmacc.vx       v20, t0, v8
    vwmacc.vx       v24, t0, v10
    lb              s4, 7(a2)
    vwmacc.vx       v28, t0, v12
    addi            a2, a2, 8   // input_data += 8

    vsetvli         zero, a4, e8, mf2
    vle8.v          v4, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v5, s1
    lb              t1, 0(a2)
    vwmul.vx        v8, v5, s2
    vwmul.vx        v10, v5, s3
    lb              t2, 1(a2)
    vwmul.vx        v12, v5, s4

    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6
    lb              t3, 2(a2)
    vwmacc.vx       v20, t0, v8
    vwmacc.vx       v24, t0, v10
    lb              t4, 3(a2)
    vwmacc.vx       v28, t0, v12

    addi            s0, s0, -1
    bnez            s0, packnx4_k2

packnx4_k2_end:
    vsetvli         zero, a4, e8, mf2
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 4(a2)
    vwmul.vx        v8, v4, t2
    vwmul.vx        v10, v4, t3
    lb              s2, 5(a2)
    vwmul.vx        v12, v4, t4

    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6
    lb              s3, 6(a2)
    vwmacc.vx       v20, t0, v8
    vwmacc.vx       v24, t0, v10
    lb              s4, 7(a2)
    vwmacc.vx       v28, t0, v12
    addi            a2, a2, 8   // input_data += 8

    vsetvli         zero, a4, e8, mf2

    vwmul.vx        v6, v5, s1
    vwmul.vx        v8, v5, s2
    vwmul.vx        v10, v5, s3
    vwmul.vx        v12, v5, s4

    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8
    vwmacc.vx       v24, t0, v10
    vwmacc.vx       v28, t0, v12

    andi            s0, a5, 1   // k1
    beqz            s0, packnx4_post

    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +packn
    // pre-load input_data
    lb              t1, 0(a2)
    lb              t2, 1(a2)
    lb              t3, 2(a2)
    lb              t4, 3(a2)

packnx4_k1:
    vsetvli         zero, a4, e8, mf2
    vwmul.vx        v6, v4, t1
    vwmul.vx        v8, v4, t2
    vwmul.vx        v10, v4, t3
    vwmul.vx        v12, v4, t4
    addi            a2, a2, 4   // input_data += 4

    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8
    vwmacc.vx       v24, t0, v10
    vwmacc.vx       v28, t0, v12

packnx4_post:
    vsetvli         zero, a4, e32, m2
    vle32.v         v4, (t5)    // mult
    vle32.v         v8, (t6)    // shift
    vxor.vi         v8, v8, -1

    GEMM_INT8_NCXHWX_REQUANTIZE_M2 v16
    GEMM_INT8_NCXHWX_REQUANTIZE_M2 v20
    GEMM_INT8_NCXHWX_REQUANTIZE_M2 v24
    GEMM_INT8_NCXHWX_REQUANTIZE_M2 v28

packnx4_end:
    vse8.v          v16, (a0)
    add             a0, a0, a4
    vse8.v          v20, (a0)
    add             a0, a0, a4
    vse8.v          v24, (a0)
    add             a0, a0, a4
    vse8.v          v28, (a0)
    add             a0, a0, a4

    addi            s10, s10, -1
    bnez            s10, packnx4_start

packnx2_start:
    andi            s10, a6, 2      // s10 = bool_n2
    beqz            s10, packnx1_start

    vsetvli         zero, a4, e32, m2
    vmv.v.v         v16, v0
    vmv.v.v         v20, v0

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +packn
    // pre-load input_data
    lb              t1, 0(a2)
    lb              t2, 1(a2)

    srai            s0, a5, 1   // k2
    beqz            s0, packnx2_k1
    addi            s0, s0, -1  // k2_end
    beqz            s0, packnx2_k2_end

packnx2_k2:
    vsetvli         zero, a4, e8, mf2
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 2(a2)
    vwmul.vx        v8, v4, t2
    lb              s2, 3(a2)
    vsetvli         zero, a4, e16, m1
    addi            a2, a2, 4
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8

    vsetvli         zero, a4, e8, mf2
    vle8.v          v4, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v5, s1
    lb              t1, 0(a2)
    vwmul.vx        v8, v5, s2
    lb              t2, 1(a2)
    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8

    addi            s0, s0, -1
    bnez            s0, packnx2_k2

packnx2_k2_end:
    vsetvli         zero, a4, e8, mf2
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 2(a2)
    vwmul.vx        v8, v4, t2
    lb              s2, 3(a2)
    vsetvli         zero, a4, e16, m1
    addi            a2, a2, 4
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8

    vsetvli         zero, a4, e8, mf2

    vwmul.vx        v6, v5, s1
    vwmul.vx        v8, v5, s2
    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8

    andi            s0, a5, 1   // k1
    beqz            s0, packnx2_post

    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +packn
    // pre-load input_data
    lb              t1, 0(a2)
    lb              t2, 1(a2)

packnx2_k1:
    vsetvli         zero, a4, e8, mf2
    vwmul.vx        v6, v4, t1
    vwmul.vx        v8, v4, t2
    addi            a2, a2, 2   // input_data += 2
    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6
    vwmacc.vx       v20, t0, v8

packnx2_post:
    vsetvli         zero, a4, e32, m2
    vle32.v         v4, (t5)    // mult
    vle32.v         v8, (t6)    // shift
    vxor.vi         v8, v8, -1

    GEMM_INT8_NCXHWX_REQUANTIZE_M2 v16
    GEMM_INT8_NCXHWX_REQUANTIZE_M2 v20

packnx2_end:
    vse8.v          v16, (a0)
    add             a0, a0, a4
    vse8.v          v20, (a0)
    add             a0, a0, a4

packnx1_start:
    andi            s10, a6, 1  // s10 = bool_n1
    beqz            s10, packn_end

    vsetvli         zero, a4, e32, m2
    vmv.v.v         v16, v0

    mv              s9, a1  // kernel origin addr
    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +packn
    // pre-load input_data
    lb              t1, 0(a2)

    srai            s0, a5, 1   // k2
    beqz            s0, packnx1_k1
    addi            s0, s0, -1  // k2_end
    beqz            s0, packnx1_k2_end

packnx1_k2:
    vsetvli         zero, a4, e8, mf2
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 1(a2)
    vsetvli         zero, a4, e16, m1
    addi            a2, a2, 2
    vwmacc.vx       v16, t0, v6

    vsetvli         zero, a4, e8, mf2
    vle8.v          v4, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v5, s1
    lb              t1, 0(a2)
    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6

    addi            s0, s0, -1
    bnez            s0, packnx1_k2

packnx1_k2_end:
    vsetvli         zero, a4, e8, mf2
    vle8.v          v5, (s9)
    add             s9, s9, a4  // kernel_data += packn

    vwmul.vx        v6, v4, t1
    lb              s1, 1(a2)
    vsetvli         zero, a4, e16, m1
    addi            a2, a2, 2
    vwmacc.vx       v16, t0, v6

    vsetvli         zero, a4, e8, mf2

    vwmul.vx        v6, v5, s1
    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6

    andi            s0, a5, 1   // k1
    beqz            s0, packnx1_post

    // pre-load kernel_data
    vle8.v          v4, (s9)
    add             s9, s9, a4  // +packn
    // pre-load input_data
    lb              t1, 0(a2)

packnx1_k1:
    vsetvli         zero, a4, e8, mf2
    vwmul.vx        v6, v4, t1
    addi            a2, a2, 1   // input_data += 1
    vsetvli         zero, a4, e16, m1
    vwmacc.vx       v16, t0, v6

packnx1_post:
    vsetvli         zero, a4, e32, m2
    vle32.v         v4, (t5)    // mult
    vle32.v         v8, (t6)    // shift
    vxor.vi         v8, v8, -1

    GEMM_INT8_NCXHWX_REQUANTIZE_M2 v16

packnx1_end:
    vse8.v          v16, (a0)
    add             a0, a0, a4

packn_end:
    ld              s0, 0(sp)
    ld              s1, 8(sp)
    ld              s2, 16(sp)
    ld              s3, 24(sp)
    ld              s4, 32(sp)
    ld              s9, 40(sp)
    ld              s10, 48(sp)
    addi            sp, sp, 56

    ret
    .end