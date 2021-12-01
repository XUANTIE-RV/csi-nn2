/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.10.x */


/******************************************************************************
 * @file     csi_i805_maxpool2d_8.S
 * @brief    uint8 maxpool function.
 * @version  V1.0
 * @date     9. Jul 2021
 ******************************************************************************/

/*
    void csi_i805_maxpool2d_opt_u8(uint8_t *input_data,
                                 uint8_t *output_data,
                                 int32_t input_h,
                                 int32_t input_w,
                                 int32_t input_ch,
                                 int32_t kernel_h,
                                 int32_t kernel_w,
                                 int32_t pad_h,
                                 int32_t pad_w,
                                 int32_t stride_h,
                                 int32_t stride_w,
                                 int32_t output_h,
                                 int32_t output_w)

    Algorithm works as follows:
        (1) calculate the pooling by sliding it window by window, window_size: [kernel_h * kernel_w]
        (2) if first index in window, copy in_ch elements from start_addr to dst,
            else compare in_ch elements between start_addr and dst_addr
        (3) finish one window, dst += in_ch

    register definition:
        a0: input_data
        a1: output_data
        a2: input_h
        a3: input_w
        t0: i_out_h
        t1: i_out_w
        t2: i_ker_h
        t3: i_ker_w
        t4: base_idx_h
        t5: base_idx_w
        l3: ker_h_start
        l4: ker_h_end
        l5: ker_w_start
        l6: ker_w_end
        l7: count if 0, store input to output, else compare
        l8: output_data addr temp

 */

    .file           "csi_i805_maxpool2d_8.S"
    .section        .text.csi_i805_maxpool2d_opt_u8,"ax",@progbits
    .align          2
    .global         csi_i805_maxpool2d_opt_u8
    .type           csi_i805_maxpool2d_opt_u8, @function


csi_i805_maxpool2d_opt_u8:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8

    ld.w            l0, (sp, 0x24)      // input_ch
    ld.w            l1, (sp, 0x28)      // kernel_h
    ld.w            l2, (sp, 0x2c)      // kernel_w

    movi            t0, 0               // i_out_h
    ld.w            t8, (sp, 0x30)      // pad_h
    subu            t4, t0, t8          // base_idx_h = -pad_h

.OUT_H:
    ld.w            t8, (sp, 0x40)      // t8 = out_h
    cmplt           t0, t8              // i_out_h < out_h
    bf              .END

    movi            t1, 0               // i_out_w reset
    ld.w            t9, (sp, 0x34)      // pad_w
    subu            t5, t1, t9          // base_idx_w = -pad_w

.OUT_W:
    ld.w            t9, (sp, 0x44)      // t9 = out_w
    cmplt           t1, t9              // i_out_w < out_w
    bf              .OUT_H_ADD

/*
    ker_h_start = max(0, -base_idx_h)
    ker_w_start = max(0, -base_idx_w)
    ker_h_end = min(ker_h, input_h-base_idx_h)
    ker_w_end = min(ker_w, input_w-base_idx_w)
*/
    movi            l3, 0
    movi            l5, 0

    movi            t6, 0
    subu            t7, t6, t4          // -base_idx_h
    subu            t8, t6, t5          // -base_idx_w

    cmplt           t6, t7
    movt            l3, t7
    cmplt           t6, t8
    movt            l5, t8

    mov             l4, l1
    mov             l6, l2

    addu            t9, a2, t7
    cmplt           l1, t9
    movt            l4, t9

    addu            t9, a3, t8
    cmplt           l2, t9
    movt            l6, t9

    mov             l8, a1              // update dst addr
    movi            l7, 0               // count = 0
    mov             t2, l3              // i_ker_h = ker_h_start

.KERNEL_H:
    cmplt           t2, l4              // i_ker_h < ker_h_end
    bf              .OUT_W_ADD

    mov             t3, l5              // i_ker_w = ker_w_start

.KERNEL_W:
    cmplt           t3, l6              // i_ker_w < ker_w_end
    bf              .KERNEL_H_ADD

    addu            t6, t2, t4          // i_ker_h + base_idx_h
    addu            t7, t3, t5          // i_ker_w + base_idx_w

    mult            t6, t6, a3          // (i_ker_h + base_idx_h) * input_w
    addu            t6, t6, t7          // (i_ker_h + base_idx_h) * input_w + i_ker_w + base_idx_w
    mult            t6, t6, l0          // in_ch * ((i_ker_h + base_idx_h) * input_w + i_ker_w + base_idx_w)
    addu            t7, a0, t6          // start_addr

    cmplti          l7, 1               // count < 1?  i.e. count==0
    bf              .COMPARE

.COPY:
    // if branch, first index in window[kernel_h * kernel_w], copy in_ch elements from start_addr to output_data
    addi            l7, l7, 1           // count++
    lsri            t6, l0, 4           // in_ch >> 4
    bez             t6, .COPY_TAIL

.COPY_16:
    vldmu.8         vr0-vr0, (t7)
    vstmu.8         vr0-vr0, (l8)
    bnezad          t6, .COPY_16

.COPY_TAIL:
    andi            t6, l0, 15          // in_ch & 15u
    bez             t6, .KERNEL_W_ADD

    vldx.8          vr0, (t7), t6
    vstx.8          vr0, (l8), t6
    // addu            l8, l8, t6          // l8 update +in_ch
    mov             l8, a1              //
    br              .KERNEL_W_ADD

.COMPARE:
    // else branch
    lsri            t6, l0, 4           // in_ch >> 4
    bez             t6, .COMPARE_TAIL

.COMPARE_16:
    vldm.8          vr0-vr0, (l8)       // load from dst
    vldmu.8         vr1-vr1, (t7)       // load from start_addr
    vmax.u8         vr2, vr0, vr1       //
    vstmu.8         vr2-vr2, (l8)
    bnezad          t6, .COMPARE_16

.COMPARE_TAIL:
    andi            t6, l0, 15          // in_ch & 15u
    bez             t6, .KERNEL_W_ADD

    vldx.8          vr0, (l8), t6
    vldx.8          vr1, (t7), t6
    vmax.u8         vr2, vr0, vr1
    vstx.8          vr2, (l8), t6
    mov             l8, a1

.KERNEL_W_ADD:
    addi            t3, t3, 1           // i_ker_w ++
    br              .KERNEL_W

.KERNEL_H_ADD:
    addi            t2, t2, 1           // i_ker_h ++
    br              .KERNEL_H


.OUT_W_ADD:
    addu            a1, a1, l0          // output_data + in_ch
    ld.w            t9, (sp, 0x3c)      // stride_w
    addu            t5, t5, t9          // base_idx_w += stride_w
    addi            t1, t1, 1           // i_out_w++
    br              .OUT_W

.OUT_H_ADD:
    ld.w            t8, (sp, 0x38)      // stride_h
    addu            t4, t4, t8          // base_idx_h += stride_h
    addi            t0, t0, 1           // i_out_h++
    br              .OUT_H


.END:
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8
    rts
    .size           csi_i805_maxpool2d_opt_u8, .-csi_i805_maxpool2d_opt_u8
