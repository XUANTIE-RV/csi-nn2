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

/* CSI-NN2 version 1.11.x */

#include "csi_c906.h"

// constrains: The destination address and source address copy do not overlap
// notice: riscv gnu compiler tool-chain c-library memcpy may not use vector inst
// now gcc version: gcc version 10.2.0 (T-HEAD RISCV Tools V2.0.1 B20210512)
void csi_c906_memcpy(void *dst, const void *src, size_t n)
{
    asm volatile(
                "1:\n\t"
                "vsetvli    t0, %3, e8, m4\n\t"
                "vle.v      v4, (%2)\n\t"
                "add        %2, %2, t0\n\t"
                "sub        %3, %3, t0\n\t"
                "vse.v      v4, (%0)\n\t"
                "add        %0, %0, t0\n\t"
                "bnez       %3, 1b\n\t"

                :"=r"(dst)  // %0
                :"0"(dst),  // %1
                "r"(src),   // %2
                "r"(n)      // %3
                : "t0", "v4", "v5", "v6", "v7"
    );
}



/*  params:
    input:          origin input data
    input_padded:   input data after pad
    inc:            origin input channel
    inh:            origin input height
    inw:            origin input width
    padded_h:       input height after pad
    padded_w:       input width after pad
    pad_top:        origin pad top
    pad_left:       origin pad left
*/
void csi_c906_pad_input(const float *input, float *input_padded, int inc, int inh, int inw,
                        int padded_h, int padded_w, int pad_top, int pad_left)
{
    int padded_hw = padded_h * padded_w;

    float *pad_ptr = input_padded;
    float *inp_ptr = (float *)input;
    int resi_h = padded_h - pad_top - inh;  // remain to pad on h (pad_down)
    int resi_w = padded_w - pad_left - inw; // remain to pad on w (pad_right)

#if __riscv_vector == 128

    asm volatile(
        "vsetvli        zero, zero, e32, m2\n\t"
        "vmv.v.x        v2, zero\n\t"       // clear v2
        "mulw           t1, %6, %7\n\t"     // pad_top * padded_w
        "mulw           t2, %6, %9\n\t"     // pad_down * padded_w

    "1:\n\t"     // channel loop
        "mv             t5, %3\n\t"         // t5 = in_h
        "beqz           %7, 3f\n\t"         // if pad_top = 0
        "mv             t3, t1\n\t"         // t3 = num to memset 0

        "2:\n\t"    // pad h_top
            "vsetvli        t0, t3, e32, m2\n\t"
            "vsw.v          v2, (%1)\n\t"
            "sub            t3, t3, t0\n\t"
            "slli           t0, t0, 2\n\t"
            "add            %1, %1, t0\n\t"
            "bnez           t3, 2b\n\t"

        "3:\n\t"    // pad h_mid

            "mv             t4, %4\n\t"     // t4 = in_w
            "beqz           %8, 5f\n\t"     // if pad_left = 0
            "mv             t3, %8\n\t"     // t3 = pad_left

            "4:\n\t"    // pad w_left
                "vsetvli        t0, t3, e32, m2\n\t"
                "vsw.v          v2, (%1)\n\t"
                "sub            t3, t3, t0\n\t"
                "slli           t0, t0, 2\n\t"
                "add            %1, %1, t0\n\t"
                "bnez           t3, 4b\n\t"

            "5:\n\t"    // pad w_mid
                "vsetvli        t0, t4, e32, m2\n\t"
                "vlw.v          v4, (%0)\n\t"   // load from input_data
                "sub            t4, t4, t0\n\t"
                "slli           t0, t0, 2\n\t"
                "add            %0, %0, t0\n\t"
                "vsw.v          v4, (%1)\n\t"   // store to padded_buf
                "add            %1, %1, t0\n\t"
                "bnez           t4, 5b\n\t"

                "beqz           %10, 7f\n\t"    // if pad_right = 0
                "mv             t3, %10\n\t"

            "6:\n\t"    // pad w_right
                "vsetvli        t0, t3, e32, m2\n\t"
                "vsw.v          v2, (%1)\n\t"
                "sub            t3, t3, t0\n\t"
                "slli           t0, t0, 2\n\t"
                "add            %1, %1, t0\n\t"
                "bnez           t3, 6b\n\t"

        "7:\n\t"

            "addi           t5, t5, -1\n\t"
            "bnez           t5, 3b\n\t"

        "beqz           %9, 9f\n\t"     // if pad_down = 0
        "mv             t3, t2\n\t"     // t3 = num to memset 0

        "8:\n\t"     // pad h_down
            "vsetvli        t0, t3, e32, m2\n\t"
            "vsw.v          v2, (%1)\n\t"
            "sub            t3, t3, t0\n\t"
            "slli           t0, t0, 2\n\t"
            "add            %1, %1, t0\n\t"
            "bnez           t3, 8b\n\t"

    "9:\n\t"
        "addi           %2, %2, -1\n\t"
        "bnez           %2, 1b\n\t"

        :"=r"(inp_ptr),     // %0
        "=r"(pad_ptr),      // %1
        "=r"(inc),          // %2
        "=r"(inh),          // %3
        "=r"(inw),          // %4
        "=r"(padded_hw),    // %5
        "=r"(padded_w),     // %6
        "=r"(pad_top),      // %7
        "=r"(pad_left),     // %8
        "=r"(resi_h),       // %9
        "=r"(resi_w)        // %10
        :"0"(inp_ptr),
        "1"(pad_ptr),
        "2"(inc),
        "3"(inh),
        "4"(inw),
        "5"(padded_hw),
        "6"(padded_w),
        "7"(pad_top),
        "8"(pad_left),
        "9"(resi_h),
        "10"(resi_w)
        :"cc", "memory", "v2", "v3", "v4", "v5",
         "t0", "t1", "t2", "t3", "t4", "t5"

    );
#else
    for (int c = 0; c < inc; c++) {
        pad_ptr = input_padded + c * padded_hw;
        // pad h_top
        memset(pad_ptr, 0, padded_w * pad_top * sizeof(float));
        pad_ptr += pad_top * padded_w;
        // pad h_mid
        for (int h = 0; h < inh; h++) {
            // pad w_left
            memset(pad_ptr, 0, pad_left * sizeof(float));
            // pad w_mid
            memcpy(pad_ptr + pad_left, inp_ptr, inw * sizeof(float));
            // pad w_end
            memset(pad_ptr + pad_left + inw, 0, resi_w * sizeof(float));

            inp_ptr += inw;
            pad_ptr += padded_w;
        }
        // pad h_bottom
        memset(pad_ptr, 0, padded_w * resi_h * sizeof(float));
    }
#endif  // __riscv_vector
}


void csi_c906_pad_input_fp16(const __fp16 *input, __fp16 *input_padded, int inc, int inh, int inw,
                             int padded_h, int padded_w, int pad_top, int pad_left)
{
    int padded_hw = padded_h * padded_w;

    __fp16 *pad_ptr = input_padded;
    __fp16 *inp_ptr = (__fp16 *)input;
    int resi_h = padded_h - pad_top - inh;  // remain to pad on h (pad_down)
    int resi_w = padded_w - pad_left - inw; // remain to pad on w (pad_right)

    asm volatile(
        "vsetvli        zero, zero, e16, m2\n\t"
        "vmv.v.x        v2, zero\n\t"       // clear v2
        "mulw           t1, %6, %7\n\t"     // pad_top * padded_w
        "mulw           t2, %6, %9\n\t"     // pad_down * padded_w

    "1:\n\t"     // channel loop
        "mv             t5, %3\n\t"         // t5 = in_h
        "beqz           %7, 3f\n\t"         // if pad_top = 0
        "mv             t3, t1\n\t"         // t3 = num to memset 0

        "2:\n\t"    // pad h_top
            "vsetvli        t0, t3, e16, m2\n\t"
            "vse.v          v2, (%1)\n\t"
            "sub            t3, t3, t0\n\t"
            "slli           t0, t0, 1\n\t"
            "add            %1, %1, t0\n\t"
            "bnez           t3, 2b\n\t"

        "3:\n\t"    // pad h_mid

            "mv             t4, %4\n\t"     // t4 = in_w
            "beqz           %8, 5f\n\t"     // if pad_left = 0
            "mv             t3, %8\n\t"     // t3 = pad_left

            "4:\n\t"    // pad w_left
                "vsetvli        t0, t3, e16, m2\n\t"
                "vse.v          v2, (%1)\n\t"
                "sub            t3, t3, t0\n\t"
                "slli           t0, t0, 1\n\t"
                "add            %1, %1, t0\n\t"
                "bnez           t3, 4b\n\t"

            "5:\n\t"    // pad w_mid
                "vsetvli        t0, t4, e16, m2\n\t"
                "vle.v          v4, (%0)\n\t"   // load from input_data
                "sub            t4, t4, t0\n\t"
                "slli           t0, t0, 1\n\t"
                "add            %0, %0, t0\n\t"
                "vse.v          v4, (%1)\n\t"   // store to padded_buf
                "add            %1, %1, t0\n\t"
                "bnez           t4, 5b\n\t"

                "beqz           %10, 7f\n\t"    // if pad_right = 0
                "mv             t3, %10\n\t"

            "6:\n\t"    // pad w_right
                "vsetvli        t0, t3, e16, m2\n\t"
                "vse.v          v2, (%1)\n\t"
                "sub            t3, t3, t0\n\t"
                "slli           t0, t0, 1\n\t"
                "add            %1, %1, t0\n\t"
                "bnez           t3, 6b\n\t"

        "7:\n\t"

            "addi           t5, t5, -1\n\t"
            "bnez           t5, 3b\n\t"

        "beqz           %9, 9f\n\t"     // if pad_down = 0
        "mv             t3, t2\n\t"     // t4 = num to memset 0

        "8:\n\t"     // pad h_down
            "vsetvli        t0, t3, e16, m2\n\t"
            "vse.v          v2, (%1)\n\t"
            "sub            t3, t3, t0\n\t"
            "slli           t0, t0, 1\n\t"
            "add            %1, %1, t0\n\t"
            "bnez           t3, 8b\n\t"

    "9:\n\t"
        "addi           %2, %2, -1\n\t"
        "bnez           %2, 1b\n\t"

        :"=r"(inp_ptr),     // %0
        "=r"(pad_ptr),      // %1
        "=r"(inc),          // %2
        "=r"(inh),          // %3
        "=r"(inw),          // %4
        "=r"(padded_hw),    // %5
        "=r"(padded_w),     // %6
        "=r"(pad_top),      // %7
        "=r"(pad_left),     // %8
        "=r"(resi_h),       // %9
        "=r"(resi_w)        // %10
        :"0"(inp_ptr),
        "1"(pad_ptr),
        "2"(inc),
        "3"(inh),
        "4"(inw),
        "5"(padded_hw),
        "6"(padded_w),
        "7"(pad_top),
        "8"(pad_left),
        "9"(resi_h),
        "10"(resi_w)
        :"cc", "memory", "v2", "v3", "v4", "v5",
         "t0", "t1", "t2", "t3", "t4", "t5"

    );
}


/*  params:
    output_trans:   transflorm output after dot
    output：        final output data
    out_c:          final output channel
    out_h:          final output height
    out_w:          final output width
    wino_h:         winograd conv out_h, alignment with 2/4/6
    wino_w：        winograd conv out_w, alignment with 2/4/6
*/
void csi_c906_crop_output(float *output_trans, float *output, int out_c, int out_h, int out_w,
                          int wino_h, int wino_w)
{
    int resi_h = wino_h - out_h;
    int resi_w = wino_w - out_w;
    float *out_ptr = output;
    for(int c = 0; c < out_c; c++) {

        float *crop_ptr = output_trans + c * wino_h * wino_w;

        for(int h = 0; h < out_h; h++) {
            memcpy(out_ptr, crop_ptr, out_w * sizeof(float));
            out_ptr += out_w;
            crop_ptr += wino_w;
        }
    }
}

void csi_c906_crop_output_fp16(__fp16 *output_trans, __fp16 *output, int out_c, int out_h, int out_w,
                               int wino_h, int wino_w)
{
    int resi_h = wino_h - out_h;
    int resi_w = wino_w - out_w;
    __fp16 *out_ptr = output;
    for(int c = 0; c < out_c; c++) {

        __fp16 *crop_ptr = output_trans + c * wino_h * wino_w;

        for(int h = 0; h < out_h; h++) {
            memcpy(out_ptr, crop_ptr, out_w * sizeof(__fp16));
            out_ptr += out_w;
            crop_ptr += wino_w;
        }
    }
}
