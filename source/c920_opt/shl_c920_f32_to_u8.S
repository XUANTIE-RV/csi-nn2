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

    void shl_c920_f32_to_u8(const float *input,
                            uint8_t *output,
                            int32_t offset,
                            float *scale,
                            uint32_t length)

    Algorithm works as follows:
        (1) f32 *= 1/scale, convert to i32
        (2) i32 += offset
        (3) for UNSIGNED, i32 = max(i32, 0)
        (4) i32 narrow to u16/i16
        (5) u16/i16 narrow to u8/i8

    register definition:
        a0:         input addr
        a1:         output addr
        a2:         zero_point
        a3:         scale
        a4:         element length
        fa0:        1/scale
        ft0:        1.0f
        v0-v7:      f32
        v8-v15:     f32,u32/i32
        v16-v19:    u16/i16
        v20-v21:    u8/i8
        v28-v31:    1/scale

    note: vector extension 0.7.1 [support flexible vlen]

 *************************************************************************************************/
#ifndef SHL_C920_F32_TO_8
#define SHL_C920_F32_TO_8 shl_c920_f32_to_u8
#endif

    .file           "shl_c920_f32_to_u8.S"
    .section        .text.SHL_C920_F32_TO_8, "ax", @progbits
    .align          5
    .global         SHL_C920_F32_TO_8
    .type           SHL_C920_F32_TO_8, @function


SHL_C920_F32_TO_8:
    csrr            t0, vlenb   // t0 = vlen/8
    slli            t1, t0, 1
    slli            t2, t0, 2
    flw             fa0, (a3)
    li              t3, 1
    fcvt.s.w        ft0, t3
    fdiv.s          fa0, ft0, fa0
    vsetvli         zero, zero, e32, m4
    vfmv.v.f        v28, fa0

.L2:
    bgt             t1, a4, .L1
    vsetvli         zero, zero, e32, m4
    vle.v           v0, (a0)
    add             a0, a0, t2
    vle.v           v4, (a0)
    add             a0, a0, t2

    sub             a4, a4, t1
    bgt             t1, a4, .L2End

.L2Loop:
    vfmul.vv        v8, v0, v28
    vfmul.vv        v12, v4, v28
    vfcvt.x.f.v     v8, v8
    vfcvt.x.f.v     v12, v12
    vle.v           v0, (a0)
    add             a0, a0, t2
    vle.v           v4, (a0)
    add             a0, a0, t2
    vadd.vx         v8, v8, a2
    vadd.vx         v12, v12, a2
#ifndef SHL_C920_F32_TO_8_SIGNED
    vmax.vx         v8, v8, zero
    vmax.vx         v12, v12, zero
    vsetvli         zero, zero, e16, m2
    vnclipu.vi      v16, v8, 0
    vnclipu.vi      v18, v12, 0
    vsetvli         zero, zero, e8, m1
    vnclipu.vi      v20, v16, 0
    vnclipu.vi      v21, v18, 0
#else
    vsetvli         zero, zero, e16, m2
    vnclip.vi       v16, v8, 0
    vnclip.vi       v18, v12, 0
    vsetvli         zero, zero, e8, m1
    vnclip.vi       v20, v16, 0
    vnclip.vi       v21, v18, 0
#endif
    vse.v           v20, (a1)
    add             a1, a1, t0
    vse.v           v21, (a1)
    add             a1, a1, t0

    vsetvli         zero, zero, e32, m4
    sub             a4, a4, t1
    bgt             a4, t1, .L2Loop

.L2End:
    vfmul.vv        v8, v0, v28
    vfmul.vv        v12, v4, v28
    vfcvt.x.f.v     v8, v8
    vfcvt.x.f.v     v12, v12
    vadd.vx         v8, v8, a2
    vadd.vx         v12, v12, a2
#ifndef SHL_C920_F32_TO_8_SIGNED
    vmax.vx         v8, v8, zero
    vmax.vx         v12, v12, zero
    vsetvli         zero, zero, e16, m2
    vnclipu.vi      v16, v8, 0
    vnclipu.vi      v18, v12, 0
    vsetvli         zero, zero, e8, m1
    vnclipu.vi      v20, v16, 0
    vnclipu.vi      v21, v18, 0
#else
    vsetvli         zero, zero, e16, m2
    vnclip.vi       v16, v8, 0
    vnclip.vi       v18, v12, 0
    vsetvli         zero, zero, e8, m1
    vnclip.vi       v20, v16, 0
    vnclip.vi       v21, v18, 0
#endif
    vse.v           v20, (a1)
    add             a1, a1, t0
    vse.v           v21, (a1)
    add             a1, a1, t0

.L1:
    beqz            a4, .End

.L1Loop:
    vsetvli         t0, a4, e32, m4
    slli            t2, t0, 2
    vle.v           v0, (a0)
    add             a0, a0, t2
    vfmul.vv        v8, v0, v28
    vfcvt.x.f.v     v8, v8
    vadd.vx         v8, v8, a2
#ifndef SHL_C920_F32_TO_8_SIGNED
    vmax.vx         v8, v8, zero
    vsetvli         t0, a4, e16, m2
    vnclipu.vi      v16, v8, 0
    vsetvli         t0, a4, e8, m1
    vnclipu.vi      v20, v16, 0
#else
    vsetvli         t0, a4, e16, m2
    vnclip.vi       v16, v8, 0
    vsetvli         t0, a4, e8, m1
    vnclip.vi       v20, v16, 0
#endif
    vse.v           v20, (a1)
    add             a1, a1, t0

    sub             a4, a4, t0
    bgtz            a4, .L1Loop

.End:
    ret
    .end
