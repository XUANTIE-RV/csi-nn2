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

/* CSI-NN2 version 2.0.x */


/******************************************************************************
 * @file     shl_i805_rel8_8.S
 * @brief    uint8 asym relu6 layer function.
 * @version  V1.0
 * @date     9. Jul 2021
 ******************************************************************************/

/*
    void shl_i805_relu6_opt_u8(uint8_t *data,
                               int32_t size,
                               int32_t input_zeropoint,
                               int32_t out_multiplier,
                               int32_t out_shift)

    Algorithm works as follows:
        (1) same as relu
        (2) 现在relu6完全和relu实现一样，主要是上限255和缩位到u8对上了，要FIX一下

    register definition:
        a0:         data
        a1:         size
        a2:         input_zeropoint
        a3:         out_multiplier
        sp+4/l0:    shift left
        vr5:        shift left
        vr6:        multipiler
        vr7:        input_zeropoint
        vr0:        input_data_tmp / compare result
        vr1:        output_data_tmp
        t0:         temp loop count
 */

    .file           "shl_i805_relu6_8.S"
    .section        .text.shl_i805_relu6_opt_u8,"ax",@progbits
    .align          2
    .global         shl_i805_relu6_opt_u8
    .type           shl_i805_relu6_opt_u8, @function

shl_i805_relu6_opt_u8:
    push            l0

    ld.w            l0, (sp, 0x04)      // out_shift
    vdupg.32        vr7, a2             // vr7[0..3] = input_zero_point
    vdupg.32        vr6, a3             // vr6[0..3] = multiplier
    vdupg.32        vr5, l0             // shift left > 0
    lsri            t0, a1, 2           // size >> 2

    bez             t0, .TAIL_4

.LOOP_4:
    // deal with 64 elements for this loop
    vld.8.4         vr0, (a0, 0)
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr0, vr0            // widen to 32 bit
    vmax.u32        vr0, vr0, vr7       // max_ = max(q , zp)
    vsub.u32        vr0, vr0, vr7       // q_ = max_ - zp

    vshl.s32.S      vr0, vr0, vr5       // shift left

    vrmulh.s32.rs   vr1, vr0, vr6       // doubling multiply high half

    vmov.u32.sl     vr1, vr1, vr1
    vmov.u16.sl     vr1, vr1, vr1       // narrow to 8 bit

    vstu.8.4        vr1, (a0)
    bnezad          t0, .LOOP_4

.TAIL_4:
    andi            t0, a1, 4           // size % 4
    bez             t0, .END

    vldx.8          vr0, (a0), t0
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr0, vr0            // 16bit -> 32bit
    vmax.u32        vr0, vr0, vr7       // max_ = max(q , zp)
    vsub.u32        vr0, vr0, vr7       // q_ = max_ - zp

    vshl.s32.S      vr0, vr0, vr5       // shift left

    vrmulh.s32.rs   vr1, vr0, vr6       // doubling multiply high half

    vmov.u32.sl     vr1, vr1, vr1
    vmov.u16.sl     vr1, vr1, vr1       // narrow to 8 bit

    vstx.8          vr1, (a0), t0

.END:
    pop             l0
    rts
    .size           shl_i805_relu6_opt_u8, .-shl_i805_relu6_opt_u8
