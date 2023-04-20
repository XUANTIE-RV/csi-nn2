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

/* SHL version 2.1.x */

/**************************************************************************************************

    void gemm_fp32_ncxhwx_12xpack2n(const float *output,
                                    const float *kernel,
                                    const float *input,
                                    const float *bias,
                                    int m,          // maxtrix A row
                                    int k,          // maxtrix A col / maxtrix B row
                                    int n,          // maxtrix B col
                                    bool fuse_relu)

    Algorithm works as follows:
        (1) perform matrix-multiplication [pack2n, k] x [k, n] = [pack2n, n]
            ...

    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr [NULL without bais]
        a4: m [packn]
        a5: k [kernel_size]
        a6: n [out_hw]
        a7: fuse_bias

        t0 = packn * 4  maintenance kernel_addr
        t1 = tmp variable
        t2 = k2  input_channel dim loop count
        t3 = kernel data addr
        t4 = n12
        t5 = n_tail
        t6 = next packn line output

        ft0-ft5: hold input data
        fa0-fa5: hold input data

        v1-v2:   acc initial (bias or zero)
        v3-v6:   hold kernel data
        v8-v19:  fisrt packn line acc
        v20-v31: second packn line acc

 *************************************************************************************************/
    .file           "gemm_fp32_ncxhwx.S"
    .section        .text.gemm_fp32_ncxhwx_12xpack2n, "ax", @progbits
    .align          5
    .global         gemm_fp32_ncxhwx_12xpack2n
    .type           gemm_fp32_ncxhwx_12xpack2n, @function

gemm_fp32_ncxhwx_12xpack2n:
    slli            t0, a4, 2   // t0 = packn * 4
    vsetvli         zero, a4, e32, m1

    mul             t1, t0, a6  // packn * n
    add             t6, a0, t1  // t6[out1_addr] = out0_addr + packn * n

    li              t1, 12
    divw            t4, a6, t1  // t4 = n12
    remw            t5, a6, t1  // t5 = n % 12 (n_tail)

    vmv.v.x         v1, zero    // clear acc
    vmv.v.x         v2, zero
    // pack2n * n [init]
    beqz            a3, non_bias1
    vle32.v         v1, (a3)
    add             a3, a3, t0  // +packn
    vle32.v         v2, (a3)

non_bias1:

    beqz            t4, pack2nx8_start  // if n12==0, jump to pack2nx8

pack2nx12_start:
    vmv.v.v         v8, v1
    vmv.v.v         v9, v1
    vmv.v.v         v10, v1
    vmv.v.v         v11, v1
    vmv.v.v         v12, v1
    vmv.v.v         v13, v1
    vmv.v.v         v14, v1
    vmv.v.v         v15, v1
    vmv.v.v         v16, v1
    vmv.v.v         v17, v1
    vmv.v.v         v18, v1
    vmv.v.v         v19, v1

    vmv.v.v         v20, v2
    vmv.v.v         v21, v2
    vmv.v.v         v22, v2
    vmv.v.v         v23, v2
    vmv.v.v         v24, v2
    vmv.v.v         v25, v2
    vmv.v.v         v26, v2
    vmv.v.v         v27, v2
    vmv.v.v         v28, v2
    vmv.v.v         v29, v2
    vmv.v.v         v30, v2
    vmv.v.v         v31, v2

    mv              t3, a1  // kernel origin addr

    // pre-load kernel_data
    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v4, (t3)
    add             t3, t3, t0  // +packn

    // pre-load input_data
    flw             ft0, 0(a2)
    flw             ft1, 4(a2)
    flw             ft2, 8(a2)
    flw             ft3, 12(a2)
    flw             ft4, 16(a2)
    flw             ft5, 20(a2)

    srai            t2, a5, 1   // k2
    beqz            t2, pack2nx12_k1

pack2nx12_k2:
    vle32.v         v5, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v6, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v20, ft0, v4
    flw             fa0, 24(a2)
    vfmacc.vf       v9, ft1, v3
    vfmacc.vf       v21, ft1, v4
    flw             fa1, 28(a2)
    vfmacc.vf       v10, ft2, v3
    vfmacc.vf       v22, ft2, v4
    flw             fa2, 32(a2)
    vfmacc.vf       v11, ft3, v3
    vfmacc.vf       v23, ft3, v4
    flw             fa3, 36(a2)
    vfmacc.vf       v12, ft4, v3
    vfmacc.vf       v24, ft4, v4
    flw             fa4, 40(a2)
    vfmacc.vf       v13, ft5, v3
    vfmacc.vf       v25, ft5, v4
    flw             fa5, 44(a2)
    vfmacc.vf       v14, fa0, v3
    vfmacc.vf       v26, fa0, v4
    flw             ft0, 48(a2)
    vfmacc.vf       v15, fa1, v3
    vfmacc.vf       v27, fa1, v4
    flw             ft1, 52(a2)
    vfmacc.vf       v16, fa2, v3
    vfmacc.vf       v28, fa2, v4
    flw             ft2, 56(a2)
    vfmacc.vf       v17, fa3, v3
    vfmacc.vf       v29, fa3, v4
    flw             ft3, 60(a2)
    vfmacc.vf       v18, fa4, v3
    vfmacc.vf       v30, fa4, v4
    flw             ft4, 64(a2)
    vfmacc.vf       v19, fa5, v3
    vfmacc.vf       v31, fa5, v4
    flw             ft5, 68(a2)

    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v4, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v5
    vfmacc.vf       v20, ft0, v6
    flw             fa0, 72(a2)
    vfmacc.vf       v9, ft1, v5
    vfmacc.vf       v21, ft1, v6
    flw             fa1, 76(a2)
    vfmacc.vf       v10, ft2, v5
    vfmacc.vf       v22, ft2, v6
    flw             fa2, 80(a2)
    vfmacc.vf       v11, ft3, v5
    vfmacc.vf       v23, ft3, v6
    flw             fa3, 84(a2)
    vfmacc.vf       v12, ft4, v5
    vfmacc.vf       v24, ft4, v6
    flw             fa4, 88(a2)
    vfmacc.vf       v13, ft5, v5
    vfmacc.vf       v25, ft5, v6
    flw             fa5, 92(a2)
    addi            a2, a2, 96
    vfmacc.vf       v14, fa0, v5
    vfmacc.vf       v26, fa0, v6
    flw             ft0, 0(a2)
    vfmacc.vf       v15, fa1, v5
    vfmacc.vf       v27, fa1, v6
    flw             ft1, 4(a2)
    vfmacc.vf       v16, fa2, v5
    vfmacc.vf       v28, fa2, v6
    flw             ft2, 8(a2)
    vfmacc.vf       v17, fa3, v5
    vfmacc.vf       v29, fa3, v6
    flw             ft3, 12(a2)
    vfmacc.vf       v18, fa4, v5
    vfmacc.vf       v30, fa4, v6
    flw             ft4, 16(a2)
    vfmacc.vf       v19, fa5, v5
    vfmacc.vf       v31, fa5, v6
    flw             ft5, 20(a2)

    addi            t2, t2, -1
    bnez            t2, pack2nx12_k2

pack2nx12_k1:
    andi            t2, a5, 1   // k1
    beqz            t2, pack2nx12_relu

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v20, ft0, v4
    flw             fa0, 24(a2)
    vfmacc.vf       v9, ft1, v3
    vfmacc.vf       v21, ft1, v4
    flw             fa1, 28(a2)
    vfmacc.vf       v10, ft2, v3
    vfmacc.vf       v22, ft2, v4
    flw             fa2, 32(a2)
    vfmacc.vf       v11, ft3, v3
    vfmacc.vf       v23, ft3, v4
    flw             fa3, 36(a2)
    vfmacc.vf       v12, ft4, v3
    vfmacc.vf       v24, ft4, v4
    flw             fa4, 40(a2)
    vfmacc.vf       v13, ft5, v3
    vfmacc.vf       v25, ft5, v4
    flw             fa5, 44(a2)
    addi            a2, a2, 48
    vfmacc.vf       v14, fa0, v3
    vfmacc.vf       v26, fa0, v4
    vfmacc.vf       v15, fa1, v3
    vfmacc.vf       v27, fa1, v4
    vfmacc.vf       v16, fa2, v3
    vfmacc.vf       v28, fa2, v4
    vfmacc.vf       v17, fa3, v3
    vfmacc.vf       v29, fa3, v4
    vfmacc.vf       v18, fa4, v3
    vfmacc.vf       v30, fa4, v4
    vfmacc.vf       v19, fa5, v3
    vfmacc.vf       v31, fa5, v4

pack2nx12_relu:
    beqz            a7, pack2nx12_end
    vmv.v.x         v0, zero
    vfmax.vv        v8, v8, v0
    vfmax.vv        v9, v9, v0
    vfmax.vv        v10, v10, v0
    vfmax.vv        v11, v11, v0
    vfmax.vv        v12, v12, v0
    vfmax.vv        v13, v13, v0
    vfmax.vv        v14, v14, v0
    vfmax.vv        v15, v15, v0
    vfmax.vv        v16, v16, v0
    vfmax.vv        v17, v17, v0
    vfmax.vv        v18, v18, v0
    vfmax.vv        v19, v19, v0
    vfmax.vv        v20, v20, v0
    vfmax.vv        v21, v21, v0
    vfmax.vv        v22, v22, v0
    vfmax.vv        v23, v23, v0
    vfmax.vv        v24, v24, v0
    vfmax.vv        v25, v25, v0
    vfmax.vv        v26, v26, v0
    vfmax.vv        v27, v27, v0
    vfmax.vv        v28, v28, v0
    vfmax.vv        v29, v29, v0
    vfmax.vv        v30, v30, v0
    vfmax.vv        v31, v31, v0

pack2nx12_end:
    vse32.v         v8, (a0)
    add             a0, a0, t0
    vse32.v         v9, (a0)
    add             a0, a0, t0
    vse32.v         v10, (a0)
    add             a0, a0, t0
    vse32.v         v11, (a0)
    add             a0, a0, t0
    vse32.v         v12, (a0)
    add             a0, a0, t0
    vse32.v         v13, (a0)
    add             a0, a0, t0
    vse32.v         v14, (a0)
    add             a0, a0, t0
    vse32.v         v15, (a0)
    add             a0, a0, t0
    vse32.v         v16, (a0)
    add             a0, a0, t0
    vse32.v         v17, (a0)
    add             a0, a0, t0
    vse32.v         v18, (a0)
    add             a0, a0, t0
    vse32.v         v19, (a0)
    add             a0, a0, t0

    vse32.v         v20, (t6)
    add             t6, t6, t0
    vse32.v         v21, (t6)
    add             t6, t6, t0
    vse32.v         v22, (t6)
    add             t6, t6, t0
    vse32.v         v23, (t6)
    add             t6, t6, t0
    vse32.v         v24, (t6)
    add             t6, t6, t0
    vse32.v         v25, (t6)
    add             t6, t6, t0
    vse32.v         v26, (t6)
    add             t6, t6, t0
    vse32.v         v27, (t6)
    add             t6, t6, t0
    vse32.v         v28, (t6)
    add             t6, t6, t0
    vse32.v         v29, (t6)
    add             t6, t6, t0
    vse32.v         v30, (t6)
    add             t6, t6, t0
    vse32.v         v31, (t6)
    add             t6, t6, t0

    addi            t4, t4, -1
    bnez            t4, pack2nx12_start

pack2nx8_start:
    andi            t4, t5, 8       // s1 = bool_n8
    beqz            t4, pack2nx4_start  // if n8==0, jump to pack2nx4

    vmv.v.v         v8, v1
    vmv.v.v         v9, v1
    vmv.v.v         v10, v1
    vmv.v.v         v11, v1
    vmv.v.v         v12, v1
    vmv.v.v         v13, v1
    vmv.v.v         v14, v1
    vmv.v.v         v15, v1

    vmv.v.v         v20, v2
    vmv.v.v         v21, v2
    vmv.v.v         v22, v2
    vmv.v.v         v23, v2
    vmv.v.v         v24, v2
    vmv.v.v         v25, v2
    vmv.v.v         v26, v2
    vmv.v.v         v27, v2

    mv              t3, a1  // kernel origin addr

    // pre-load kernel_data
    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v4, (t3)
    add             t3, t3, t0  // +packn
    // pre-load input_data
    flw             ft0, 0(a2)
    flw             ft1, 4(a2)
    flw             ft2, 8(a2)
    flw             ft3, 12(a2)

    srai            t2, a5, 1   // k2
    beqz            t2, pack2nx8_k1

pack2nx8_k2:
    vle32.v         v5, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v6, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v20, ft0, v4
    flw             fa0, 16(a2)
    vfmacc.vf       v9, ft1, v3
    vfmacc.vf       v21, ft1, v4
    flw             fa1, 20(a2)
    vfmacc.vf       v10, ft2, v3
    vfmacc.vf       v22, ft2, v4
    flw             fa2, 24(a2)
    vfmacc.vf       v11, ft3, v3
    vfmacc.vf       v23, ft3, v4
    flw             fa3, 28(a2)
    vfmacc.vf       v12, fa0, v3
    vfmacc.vf       v24, fa0, v4
    flw             ft0, 32(a2)
    vfmacc.vf       v13, fa1, v3
    vfmacc.vf       v25, fa1, v4
    flw             ft1, 36(a2)
    vfmacc.vf       v14, fa2, v3
    vfmacc.vf       v26, fa2, v4
    flw             ft2, 40(a2)
    vfmacc.vf       v15, fa3, v3
    vfmacc.vf       v27, fa3, v4
    flw             ft3, 44(a2)

    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v4, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v5
    vfmacc.vf       v20, ft0, v6
    flw             fa0, 48(a2)
    vfmacc.vf       v9, ft1, v5
    vfmacc.vf       v21, ft1, v6
    flw             fa1, 52(a2)
    vfmacc.vf       v10, ft2, v5
    vfmacc.vf       v22, ft2, v6
    flw             fa2, 56(a2)
    vfmacc.vf       v11, ft3, v5
    vfmacc.vf       v23, ft3, v6
    flw             fa3, 60(a2)
    addi            a2, a2, 64
    vfmacc.vf       v12, fa0, v5
    vfmacc.vf       v24, fa0, v6
    flw             ft0, 0(a2)
    vfmacc.vf       v13, fa1, v5
    vfmacc.vf       v25, fa1, v6
    flw             ft1, 4(a2)
    vfmacc.vf       v14, fa2, v5
    vfmacc.vf       v26, fa2, v6
    flw             ft2, 8(a2)
    vfmacc.vf       v15, fa3, v5
    vfmacc.vf       v27, fa3, v6
    flw             ft3, 12(a2)

    addi            t2, t2, -1
    bnez            t2, pack2nx8_k2

pack2nx8_k1:
    andi            t2, a5, 1   // k1
    beqz            t2, pack2nx8_relu

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v20, ft0, v4
    flw             fa0, 16(a2)
    vfmacc.vf       v9, ft1, v3
    vfmacc.vf       v21, ft1, v4
    flw             fa1, 20(a2)
    vfmacc.vf       v10, ft2, v3
    vfmacc.vf       v22, ft2, v4
    flw             fa2, 24(a2)
    vfmacc.vf       v11, ft3, v3
    vfmacc.vf       v23, ft3, v4
    flw             fa3, 28(a2)
    addi            a2, a2, 32
    vfmacc.vf       v12, fa0, v3
    vfmacc.vf       v24, fa0, v4
    vfmacc.vf       v13, fa1, v3
    vfmacc.vf       v25, fa1, v4
    vfmacc.vf       v14, fa2, v3
    vfmacc.vf       v26, fa2, v4
    vfmacc.vf       v15, fa3, v3
    vfmacc.vf       v27, fa3, v4

pack2nx8_relu:
    beqz            a7, pack2nx8_end
    vfmax.vv        v8, v8, v0
    vfmax.vv        v9, v9, v0
    vfmax.vv        v10, v10, v0
    vfmax.vv        v11, v11, v0
    vfmax.vv        v12, v12, v0
    vfmax.vv        v13, v13, v0
    vfmax.vv        v14, v14, v0
    vfmax.vv        v15, v15, v0
    vfmax.vv        v20, v20, v0
    vfmax.vv        v21, v21, v0
    vfmax.vv        v22, v22, v0
    vfmax.vv        v23, v23, v0
    vfmax.vv        v24, v24, v0
    vfmax.vv        v25, v25, v0
    vfmax.vv        v26, v26, v0
    vfmax.vv        v27, v27, v0

pack2nx8_end:
    vse32.v         v8, (a0)
    add             a0, a0, t0
    vse32.v         v9, (a0)
    add             a0, a0, t0
    vse32.v         v10, (a0)
    add             a0, a0, t0
    vse32.v         v11, (a0)
    add             a0, a0, t0
    vse32.v         v12, (a0)
    add             a0, a0, t0
    vse32.v         v13, (a0)
    add             a0, a0, t0
    vse32.v         v14, (a0)
    add             a0, a0, t0
    vse32.v         v15, (a0)
    add             a0, a0, t0

    vse32.v         v20, (t6)
    add             t6, t6, t0
    vse32.v         v21, (t6)
    add             t6, t6, t0
    vse32.v         v22, (t6)
    add             t6, t6, t0
    vse32.v         v23, (t6)
    add             t6, t6, t0
    vse32.v         v24, (t6)
    add             t6, t6, t0
    vse32.v         v25, (t6)
    add             t6, t6, t0
    vse32.v         v26, (t6)
    add             t6, t6, t0
    vse32.v         v27, (t6)
    add             t6, t6, t0

pack2nx4_start:
    andi            t4, t5, 4       // s1 = bool_n4
    beqz            t4, pack2nx2_start  // if n4==0, jump to pack2nx2

    vmv.v.v         v8, v1
    vmv.v.v         v9, v1
    vmv.v.v         v10, v1
    vmv.v.v         v11, v1

    vmv.v.v         v20, v2
    vmv.v.v         v21, v2
    vmv.v.v         v22, v2
    vmv.v.v         v23, v2

    mv              t3, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v4, (t3)
    add             t3, t3, t0  // +packn
    // pre-load input_data
    flw             ft0, 0(a2)
    flw             ft1, 4(a2)
    flw             ft2, 8(a2)
    flw             ft3, 12(a2)

    srai            t2, a5, 1   // k2
    beqz            t2, pack2nx4_k1

pack2nx4_k2:
    vle32.v         v5, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v6, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v20, ft0, v4
    flw             fa0, 16(a2)
    vfmacc.vf       v9, ft1, v3
    vfmacc.vf       v21, ft1, v4
    flw             fa1, 20(a2)
    vfmacc.vf       v10, ft2, v3
    vfmacc.vf       v22, ft2, v4
    flw             fa2, 24(a2)
    vfmacc.vf       v11, ft3, v3
    vfmacc.vf       v23, ft3, v4
    flw             fa3, 28(a2)
    addi            a2, a2, 32

    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v4, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, fa0, v5
    vfmacc.vf       v20, fa0, v6
    flw             ft0, 0(a2)
    vfmacc.vf       v9, fa1, v5
    vfmacc.vf       v21, fa1, v6
    flw             ft1, 4(a2)
    vfmacc.vf       v10, fa2, v5
    vfmacc.vf       v22, fa2, v6
    flw             ft2, 8(a2)
    vfmacc.vf       v11, fa3, v5
    vfmacc.vf       v23, fa3, v6
    flw             ft3, 12(a2)

    addi            t2, t2, -1
    bnez            t2, pack2nx4_k2

pack2nx4_k1:
    andi            t2, a5, 1   // k1
    beqz            t2, pack2nx4_relu

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v20, ft0, v4
    vfmacc.vf       v9, ft1, v3
    vfmacc.vf       v21, ft1, v4
    vfmacc.vf       v10, ft2, v3
    vfmacc.vf       v22, ft2, v4
    vfmacc.vf       v11, ft3, v3
    vfmacc.vf       v23, ft3, v4
    addi            a2, a2, 16

pack2nx4_relu:
    beqz            a7, pack2nx4_end
    vfmax.vv        v8, v8, v0
    vfmax.vv        v9, v9, v0
    vfmax.vv        v10, v10, v0
    vfmax.vv        v11, v11, v0
    vfmax.vv        v20, v20, v0
    vfmax.vv        v21, v21, v0
    vfmax.vv        v22, v22, v0
    vfmax.vv        v23, v23, v0

pack2nx4_end:
    vse32.v         v8, (a0)
    add             a0, a0, t0
    vse32.v         v9, (a0)
    add             a0, a0, t0
    vse32.v         v10, (a0)
    add             a0, a0, t0
    vse32.v         v11, (a0)
    add             a0, a0, t0

    vse32.v         v20, (t6)
    add             t6, t6, t0
    vse32.v         v21, (t6)
    add             t6, t6, t0
    vse32.v         v22, (t6)
    add             t6, t6, t0
    vse32.v         v23, (t6)
    add             t6, t6, t0

pack2nx2_start:
    andi            t4, t5, 2       // s1 = bool_n2
    beqz            t4, pack2nx1_start  // if n2==0, jump to pack2nx1

    vmv.v.v         v8, v1
    vmv.v.v         v9, v1

    vmv.v.v         v20, v2
    vmv.v.v         v21, v2

    mv              t3, a1  // kernel origin addr

    // pre-load kernel_data
    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v4, (t3)
    add             t3, t3, t0  // +packn
    // pre-load input_data
    flw             ft0, 0(a2)
    flw             ft1, 4(a2)

    srai            t2, a5, 1   // k2
    beqz            t2, pack2nx2_k1

pack2nx2_k2:
    vle32.v         v5, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v6, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v20, ft0, v4
    flw             fa0, 8(a2)
    vfmacc.vf       v9, ft1, v3
    vfmacc.vf       v21, ft1, v4
    flw             fa1, 12(a2)
    addi            a2, a2, 16

    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v4, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, fa0, v5
    vfmacc.vf       v20, fa0, v6
    flw             ft0, 0(a2)
    vfmacc.vf       v9, fa1, v5
    vfmacc.vf       v21, fa1, v6
    flw             ft1, 4(a2)

    addi            t2, t2, -1
    bnez            t2, pack2nx2_k2

pack2nx2_k1:
    andi            t2, a5, 1   // k1
    beqz            t2, pack2nx2_relu

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v20, ft0, v4
    vfmacc.vf       v9, ft1, v3
    vfmacc.vf       v21, ft1, v4
    addi            a2, a2, 8

pack2nx2_relu:
    beqz            a7, pack2nx2_end
    vfmax.vv        v8, v8, v0
    vfmax.vv        v9, v9, v0
    vfmax.vv        v20, v20, v0
    vfmax.vv        v21, v21, v0

pack2nx2_end:
    vse32.v         v8, (a0)
    add             a0, a0, t0
    vse32.v         v9, (a0)
    add             a0, a0, t0

    vse32.v         v20, (t6)
    add             t6, t6, t0
    vse32.v         v21, (t6)
    add             t6, t6, t0

pack2nx1_start:
    andi            t4, t5, 1       // s1 = bool_n1
    beqz            t4, pack2n_end  // if n1==0, jump to end

    vmv.v.v         v8, v1
    vmv.v.v         v20, v2

    mv              t3, a1      // kernel origin addr
    // pre-load kernel_data
    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v4, (t3)
    add             t3, t3, t0  // +packn
    // pre-load input_data
    flw             ft0, 0(a2)

    srai            t2, a5, 1   // k2
    beqz            t2, pack2nx1_k1

pack2nx1_k2:
    vle32.v         v5, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v6, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v20, ft0, v4
    flw             fa0, 4(a2)
    addi            a2, a2, 8

    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vle32.v         v4, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, fa0, v5
    vfmacc.vf       v20, fa0, v6
    flw             ft0, 0(a2)

    addi            t2, t2, -1
    bnez            t2, pack2nx1_k2

pack2nx1_k1:
    andi            t2, a5, 1   // k1
    beqz            t2, pack2nx1_relu

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v20, ft0, v4
    addi            a2, a2, 4

pack2nx1_relu:
    beqz            a7, pack2nx1_end
    vfmax.vv        v8, v8, v0
    vfmax.vv        v20, v20, v0

pack2nx1_end:
    vse32.v         v8, (a0)
    vse32.v         v20, (t6)

pack2n_end:
    ret

/**************************************************************************************************

    void gemm_fp32_ncxhwx_12xpackn(const float *output,
                                   const float *kernel,
                                   const float *input,
                                   const float *bias,
                                   int m,          // maxtrix A row
                                   int k,          // maxtrix A col / maxtrix B row
                                   int n,          // maxtrix B col
                                   bool fuse_relu)

    Algorithm works as follows:
        (1) perform matrix-multiplication [m, k] x [k, n] = [m, n]
                                           m = packn or tail_packn
            ...

    register definition:
        a0: output addr
        a1: kernel addr
        a2: input addr
        a3: bias addr [NULL without bais]
        a4: m [packn or tail_packn]
        a5: k [kernel_size]
        a6: n [out_hw]
        a7: fuse_bias

        t0 = packn * 4  maintenance kernel_addr
        t1 = tmp variable
        t2 = k2  input_channel dim loop count
        t3 = kernel data addr
        t4 = n12
        t5 = n_tail
        t6 = unused

        ft0-ft5: hold input data
        fa0-fa5: hold input data

        v1:     acc initial (bias or zero)
        v3/v5:  hold kernel data
        v8-v19: packn line acc

 *************************************************************************************************/
    .section        .text.gemm_fp32_ncxhwx_12xpackn, "ax", @progbits
    .align          5
    .global         gemm_fp32_ncxhwx_12xpackn
    .type           gemm_fp32_ncxhwx_12xpackn, @function

gemm_fp32_ncxhwx_12xpackn:
    slli            t0, a4, 2   // t0 = packn * 4
    vsetvli         zero, a4, e32, m1

    li              t1, 12
    divw            t4, a6, t1  // t4 = n12
    remw            t5, a6, t1  // t5 = n % 12 (n_tail)

    vmv.v.x         v1, zero    // clear acc

    beqz            a3, non_bias2
    vle32.v         v1, (a3)

non_bias2:
    beqz            t4, packnx8_start  // if n12==0, jump to pack2nx8

packnx12_start:
    vmv.v.v         v8, v1
    vmv.v.v         v9, v1
    vmv.v.v         v10, v1
    vmv.v.v         v11, v1
    vmv.v.v         v12, v1
    vmv.v.v         v13, v1
    vmv.v.v         v14, v1
    vmv.v.v         v15, v1
    vmv.v.v         v16, v1
    vmv.v.v         v17, v1
    vmv.v.v         v18, v1
    vmv.v.v         v19, v1

    mv              t3, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    // pre-load input_data
    flw             ft0, 0(a2)
    flw             ft1, 4(a2)
    flw             ft2, 8(a2)
    flw             ft3, 12(a2)
    flw             ft4, 16(a2)
    flw             ft5, 20(a2)

    srai            t2, a5, 1   // k2
    beqz            t2, packnx12_k1

packnx12_k2:
    vle32.v         v5, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v3
    flw             fa0, 24(a2)
    vfmacc.vf       v9, ft1, v3
    flw             fa1, 28(a2)
    vfmacc.vf       v10, ft2, v3
    flw             fa2, 32(a2)
    vfmacc.vf       v11, ft3, v3
    flw             fa3, 36(a2)
    vfmacc.vf       v12, ft4, v3
    flw             fa4, 40(a2)
    vfmacc.vf       v13, ft5, v3
    flw             fa5, 44(a2)
    vfmacc.vf       v14, fa0, v3
    flw             ft0, 48(a2)
    vfmacc.vf       v15, fa1, v3
    flw             ft1, 52(a2)
    vfmacc.vf       v16, fa2, v3
    flw             ft2, 56(a2)
    vfmacc.vf       v17, fa3, v3
    flw             ft3, 60(a2)
    vfmacc.vf       v18, fa4, v3
    flw             ft4, 64(a2)
    vfmacc.vf       v19, fa5, v3
    flw             ft5, 68(a2)

    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v5
    flw             fa0, 72(a2)
    vfmacc.vf       v9, ft1, v5
    flw             fa1, 76(a2)
    vfmacc.vf       v10, ft2, v5
    flw             fa2, 80(a2)
    vfmacc.vf       v11, ft3, v5
    flw             fa3, 84(a2)
    vfmacc.vf       v12, ft4, v5
    flw             fa4, 88(a2)
    vfmacc.vf       v13, ft5, v5
    flw             fa5, 92(a2)
    addi            a2, a2, 96
    vfmacc.vf       v14, fa0, v5
    flw             ft0, 0(a2)
    vfmacc.vf       v15, fa1, v5
    flw             ft1, 4(a2)
    vfmacc.vf       v16, fa2, v5
    flw             ft2, 8(a2)
    vfmacc.vf       v17, fa3, v5
    flw             ft3, 12(a2)
    vfmacc.vf       v18, fa4, v5
    flw             ft4, 16(a2)
    vfmacc.vf       v19, fa5, v5
    flw             ft5, 20(a2)

    addi            t2, t2, -1
    bnez            t2, packnx12_k2

packnx12_k1:
    andi            t2, a5, 1   // k1
    beqz            t2, packnx12_relu

    vfmacc.vf       v8, ft0, v3
    flw             fa0, 24(a2)
    vfmacc.vf       v9, ft1, v3
    flw             fa1, 28(a2)
    vfmacc.vf       v10, ft2, v3
    flw             fa2, 32(a2)
    vfmacc.vf       v11, ft3, v3
    flw             fa3, 36(a2)
    vfmacc.vf       v12, ft4, v3
    flw             fa4, 40(a2)
    vfmacc.vf       v13, ft5, v3
    flw             fa5, 44(a2)
    addi            a2, a2, 48
    vfmacc.vf       v14, fa0, v3
    vfmacc.vf       v15, fa1, v3
    vfmacc.vf       v16, fa2, v3
    vfmacc.vf       v17, fa3, v3
    vfmacc.vf       v18, fa4, v3
    vfmacc.vf       v19, fa5, v3

packnx12_relu:
    beqz            a7, packnx12_end
    vmv.v.x         v0, zero
    vfmax.vv        v8, v8, v0
    vfmax.vv        v9, v9, v0
    vfmax.vv        v10, v10, v0
    vfmax.vv        v11, v11, v0
    vfmax.vv        v12, v12, v0
    vfmax.vv        v13, v13, v0
    vfmax.vv        v14, v14, v0
    vfmax.vv        v15, v15, v0
    vfmax.vv        v16, v16, v0
    vfmax.vv        v17, v17, v0
    vfmax.vv        v18, v18, v0
    vfmax.vv        v19, v19, v0

packnx12_end:
    vse32.v         v8, (a0)
    add             a0, a0, t0
    vse32.v         v9, (a0)
    add             a0, a0, t0
    vse32.v         v10, (a0)
    add             a0, a0, t0
    vse32.v         v11, (a0)
    add             a0, a0, t0
    vse32.v         v12, (a0)
    add             a0, a0, t0
    vse32.v         v13, (a0)
    add             a0, a0, t0
    vse32.v         v14, (a0)
    add             a0, a0, t0
    vse32.v         v15, (a0)
    add             a0, a0, t0
    vse32.v         v16, (a0)
    add             a0, a0, t0
    vse32.v         v17, (a0)
    add             a0, a0, t0
    vse32.v         v18, (a0)
    add             a0, a0, t0
    vse32.v         v19, (a0)
    add             a0, a0, t0

    addi            t4, t4, -1
    bnez            t4, packnx12_start

packnx8_start:
    andi            t4, t5, 8       // s1 = bool_n8
    beqz            t4, packnx4_start  // if n8==0, jump to packnx4

    vmv.v.v         v8, v1
    vmv.v.v         v9, v1
    vmv.v.v         v10, v1
    vmv.v.v         v11, v1
    vmv.v.v         v12, v1
    vmv.v.v         v13, v1
    vmv.v.v         v14, v1
    vmv.v.v         v15, v1

    mv              t3, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    // pre-load input_data
    flw             ft0, 0(a2)
    flw             ft1, 4(a2)
    flw             ft2, 8(a2)
    flw             ft3, 12(a2)

    srai            t2, a5, 1   // k2
    beqz            t2, packnx8_k1

packnx8_k2:
    vle32.v         v5, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v3
    flw             fa0, 16(a2)
    vfmacc.vf       v9, ft1, v3
    flw             fa1, 20(a2)
    vfmacc.vf       v10, ft2, v3
    flw             fa2, 24(a2)
    vfmacc.vf       v11, ft3, v3
    flw             fa3, 28(a2)
    vfmacc.vf       v12, fa0, v3
    flw             ft0, 32(a2)
    vfmacc.vf       v13, fa1, v3
    flw             ft1, 36(a2)
    vfmacc.vf       v14, fa2, v3
    flw             ft2, 40(a2)
    vfmacc.vf       v15, fa3, v3
    flw             ft3, 44(a2)

    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v5
    flw             fa0, 48(a2)
    vfmacc.vf       v9, ft1, v5
    flw             fa1, 52(a2)
    vfmacc.vf       v10, ft2, v5
    flw             fa2, 56(a2)
    vfmacc.vf       v11, ft3, v5
    flw             fa3, 60(a2)
    addi            a2, a2, 64
    vfmacc.vf       v12, fa0, v5
    flw             ft0, 0(a2)
    vfmacc.vf       v13, fa1, v5
    flw             ft1, 4(a2)
    vfmacc.vf       v14, fa2, v5
    flw             ft2, 8(a2)
    vfmacc.vf       v15, fa3, v5
    flw             ft3, 12(a2)

    addi            t2, t2, -1
    bnez            t2, packnx8_k2

packnx8_k1:
    andi            t2, a5, 1   // k2
    beqz            t2, packnx8_relu

    vfmacc.vf       v8, ft0, v3
    flw             fa0, 16(a2)
    vfmacc.vf       v9, ft1, v3
    flw             fa1, 20(a2)
    vfmacc.vf       v10, ft2, v3
    flw             fa2, 24(a2)
    vfmacc.vf       v11, ft3, v3
    flw             fa3, 28(a2)
    addi            a2, a2, 32
    vfmacc.vf       v12, fa0, v3
    vfmacc.vf       v13, fa1, v3
    vfmacc.vf       v14, fa2, v3
    vfmacc.vf       v15, fa3, v3

packnx8_relu:
    beqz            a7, packnx8_end
    vfmax.vv        v8, v8, v0
    vfmax.vv        v9, v9, v0
    vfmax.vv        v10, v10, v0
    vfmax.vv        v11, v11, v0
    vfmax.vv        v12, v12, v0
    vfmax.vv        v13, v13, v0
    vfmax.vv        v14, v14, v0
    vfmax.vv        v15, v15, v0

packnx8_end:
    vse32.v         v8, (a0)
    add             a0, a0, t0
    vse32.v         v9, (a0)
    add             a0, a0, t0
    vse32.v         v10, (a0)
    add             a0, a0, t0
    vse32.v         v11, (a0)
    add             a0, a0, t0
    vse32.v         v12, (a0)
    add             a0, a0, t0
    vse32.v         v13, (a0)
    add             a0, a0, t0
    vse32.v         v14, (a0)
    add             a0, a0, t0
    vse32.v         v15, (a0)
    add             a0, a0, t0

packnx4_start:
    andi            t4, t5, 4       // s1 = bool_n4
    beqz            t4, packnx2_start  // if n4==0, jump to packnx2

    vmv.v.v         v8, v1
    vmv.v.v         v9, v1
    vmv.v.v         v10, v1
    vmv.v.v         v11, v1

    mv              t3, a1  // kernel origin addr

    // pre-load kernel_data
    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    // pre-load input_data
    flw             ft0, 0(a2)
    flw             ft1, 4(a2)
    flw             ft2, 8(a2)
    flw             ft3, 12(a2)

    srai            t2, a5, 1   // k2
    beqz            t2, packnx4_k1

packnx4_k2:
    vle32.v         v5, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v3
    flw             fa0, 16(a2)
    vfmacc.vf       v9, ft1, v3
    flw             fa1, 20(a2)
    vfmacc.vf       v10, ft2, v3
    flw             fa2, 24(a2)
    vfmacc.vf       v11, ft3, v3
    flw             fa3, 28(a2)
    addi            a2, a2, 32

    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, fa0, v5
    flw             ft0, 0(a2)
    vfmacc.vf       v9, fa1, v5
    flw             ft1, 4(a2)
    vfmacc.vf       v10, fa2, v5
    flw             ft2, 8(a2)
    vfmacc.vf       v11, fa3, v5
    flw             ft3, 12(a2)

    addi            t2, t2, -1
    bnez            t2, packnx4_k2

packnx4_k1:
    andi            t2, a5, 1   // k1
    beqz            t2, packnx4_relu

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v9, ft1, v3
    vfmacc.vf       v10, ft2, v3
    vfmacc.vf       v11, ft3, v3
    addi            a2, a2, 16

packnx4_relu:
    beqz            a7, packnx4_end
    vfmax.vv        v8, v8, v0
    vfmax.vv        v9, v9, v0
    vfmax.vv        v10, v10, v0
    vfmax.vv        v11, v11, v0

packnx4_end:
    vse32.v         v8, (a0)
    add             a0, a0, t0
    vse32.v         v9, (a0)
    add             a0, a0, t0
    vse32.v         v10, (a0)
    add             a0, a0, t0
    vse32.v         v11, (a0)
    add             a0, a0, t0

packnx2_start:
    andi            t4, t5, 2       // s1 = bool_n2
    beqz            t4, packnx1_start  // if n2==0, jump to pack1nx1

    vmv.v.v         v8, v1
    vmv.v.v         v9, v1

    mv              t3, a1  // kernel origin addr
    // pre-load kernel_data
    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    // pre-load input_data
    flw             ft0, 0(a2)
    flw             ft1, 4(a2)

    srai            t2, a5, 1   // k2
    beqz            t2, packnx2_k1

packnx2_k2:
    vle32.v         v5, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, ft0, v3
    flw             fa0, 8(a2)
    vfmacc.vf       v9, ft1, v3
    flw             fa1, 12(a2)
    addi            a2, a2, 16

    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn

    vfmacc.vf       v8, fa0, v5
    flw             ft0, 0(a2)
    vfmacc.vf       v9, fa1, v5
    flw             ft1, 4(a2)

    addi            t2, t2, -1
    bnez            t2, packnx2_k2

packnx2_k1:
    andi            t2, a5, 1   // k1
    beqz            t2, packnx2_relu

    vfmacc.vf       v8, ft0, v3
    vfmacc.vf       v9, ft1, v3
    addi            a2, a2, 8

packnx2_relu:
    beqz            a7, packnx2_end
    vfmax.vv        v8, v8, v0
    vfmax.vv        v9, v9, v0

packnx2_end:
    vse32.v         v8, (a0)
    add             a0, a0, t0
    vse32.v         v9, (a0)
    add             a0, a0, t0

packnx1_start:
    andi            t4, t5, 1       // s1 = bool_n1
    beqz            t4, packn_end   // if n1==0, jump to end

    vmv.v.v         v8, v1
    mv              t3, a1  // kernel origin addr

    // pre-load kernel_data
    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    // pre-load input_data
    flw             ft0, 0(a2)

    srai            t2, a5, 1   // k2
    beqz            t2, packnx1_k1

packnx1_k2:
    vle32.v         v5, (t3)
    add             t3, t3, t0  // +packn
    vfmacc.vf       v8, ft0, v3
    flw             fa0, 4(a2)
    addi            a2, a2, 8

    vle32.v         v3, (t3)
    add             t3, t3, t0  // +packn
    vfmacc.vf       v8, fa0, v5
    flw             ft0, 0(a2)

    addi            t2, t2, -1
    bnez            t2, packnx1_k2

packnx1_k1:
    andi            t2, a5, 1   // k2
    beqz            t2, packnx1_relu

    vfmacc.vf       v8, ft0, v3
    addi            a2, a2, 4

packnx1_relu:
    beqz            a7, packnx1_end
    vfmax.vv        v8, v8, v0

packnx1_end:
    vse32.v         v8, (a0)

packn_end:
    ret
    .end
