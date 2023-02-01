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

    void shl_c906_u8_to_f32(const uint8_t *input,
                            float *output,
                            int32_t offset,
                            float *scale,
                            uint32_t length)

    Algorithm works as follows:
        (1)

    register definition:
        a0: input addr
        a1: output addr
        a2: offset
        a3: scale point
        a4: element length

    note: vector extension 0.7.1 [support flexible vlen]

 *************************************************************************************************/
#ifndef SHL_C906_8_TO_F32
#define SHL_C906_8_TO_F32 shl_c906_u8_to_f32
#endif

    .file           "shl_c906_u8_to_f32.S"
    .section        .text.SHL_C906_8_TO_F32, "ax", @progbits
    .align          5
    .global         SHL_C906_8_TO_F32
    .type           SHL_C906_8_TO_F32, @function

SHL_C906_8_TO_F32:
    csrr            t0, vlenb   // t0 = vlen/8
    slli            t1, t0, 1
    flw             fa0, (a3)
    slli            t2, t0, 2
    vsetvli         zero, zero, e32, m4
    vfmv.v.f        v28, fa0

.L2:
    bgt             t1, a4, .L1
    vsetvli         zero, zero, e8, m1
    vle.v           v0, (a0)
    add             a0, a0, t0
    vle.v           v1, (a0)
    add             a0, a0, t0

    sub             a4, a4, t1
    bgt             t1, a4, .L2End

.L2Loop:
#ifndef SHL_C906_8_TO_F32_SIGNED
    vwaddu.vx       v2, v0, zero
    vwaddu.vx       v4, v1, zero
#else
    vwadd.vx        v2, v0, zero
    vwadd.vx        v4, v1, zero
#endif
    vle.v           v0, (a0)
    add             a0, a0, t0
    vle.v           v1, (a0)
    add             a0, a0, t0

    vsetvli         zero, zero, e16, m2
    vwsub.vx        v8, v2, a2
    vwsub.vx        v12, v4, a2
    vsetvli         zero, zero, e32, m4
    vfcvt.f.x.v     v16, v8
    vfcvt.f.x.v     v20, v12
    vfmul.vv        v8, v16, v28
    vfmul.vv        v12, v20, v28
    vse.v           v8, (a1)
    add             a1, a1, t2
    vse.v           v12, (a1)
    add             a1, a1, t2

    vsetvli         zero, zero, e8, m1
    sub             a4, a4, t1
    bgt             a4, t1, .L2Loop // xxx: >=

.L2End:
#ifndef SHL_C906_8_TO_F32_SIGNED
    vwaddu.vx       v2, v0, zero
    vwaddu.vx       v4, v1, zero
#else
    vwadd.vx        v2, v0, zero
    vwadd.vx        v4, v1, zero
#endif
    vsetvli         zero, zero, e16, m2
    vwsub.vx        v8, v2, a2
    vwsub.vx        v12, v4, a2
    vsetvli         zero, zero, e32, m4
    vfcvt.f.x.v     v16, v8
    vfcvt.f.x.v     v20, v12

    vfmul.vv        v8, v16, v28
    vfmul.vv        v12, v20, v28

    vse.v           v8, (a1)
    add             a1, a1, t2
    vse.v           v12, (a1)
    add             a1, a1, t2

.L1:
    beqz            a4, .End

.L1Loop:
    vsetvli         t0, a4, e8, m1
    slli            t1, t0, 2
    vle.v           v0, (a0)
    add             a0, a0, t0
#ifndef SHL_C906_8_TO_F32_SIGNED
    vwaddu.vx       v2, v0, zero
#else
    vwadd.vx        v2, v0, zero
#endif
    vsetvli         t0, a4, e16, m2
    vwsub.vx        v4, v2, a2
    vsetvli         t0, a4, e32, m4
    vfcvt.f.x.v     v8, v4
    vfmul.vv        v4, v8, v28
    vse.v           v4, (a1)
    add             a1, a1, t1

    sub             a4, a4, t0
    bgtz            a4, .L1Loop

.End:
    ret
    .end
