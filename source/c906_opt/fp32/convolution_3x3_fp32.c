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

/*
    the conditions for using winograd convolution
    in_channel >= 16
    out_channel >= 16
    input_height <= 120
    input_width <= 120
*/

#include "c906/c906.h"

/*
    padding input for winograd input transform , and change memory layout to [n c/4 h w 4]
    input layout: [n c h w]
    input_padded layout: [n c/4 h w 4]
    constrain: input channel % 4 = 0
*/
void shl_c906_pad_input_pack1to4(const float *input, float *input_padded, int inc, int inh, int inw,
                                 int padded_h, int padded_w, int pad_top, int pad_left)
{
    int inc4 = inc / 4;
    int padded_hw = padded_h * padded_w;

    float *pad_ptr = input_padded;
    float *inp_ptr = (float *)input;
    int resi_h = padded_h - pad_top - inh;   // remain to pad on h (pad_down)
    int resi_w = padded_w - pad_left - inw;  // remain to pad on w (pad_right)

    asm volatile(
        "vsetvli        zero, zero, e32, m1\n\t"
        "vmv.v.x        v2, zero\n\t"    // clear v2, for memset value 0
        "mulw           t1, %6, %7\n\t"  // pad_top * padded_w
        "mulw           t2, %6, %9\n\t"  // pad_down * padded_w
        "mulw           t0, %3, %4\n\t"  // input_size per_channel
        "slli           t0, t0, 2\n\t"   // load stride = input_size * 4
        "slli           t6, t0, 2\n\t"   // t6 = input_size * 4(channel) * 4 bytes

        "1:\n\t"                     // channel loop [inc/8]
        "mv             a0, %0\n\t"  // update input_addr
        "mv             t5, %3\n\t"  // t5 = in_h
        "beqz           %7, 3f\n\t"  // if pad_top = 0
        "mv             t3, t1\n\t"  // t3 = num to memset

        "2:\n\t"  // pad h_top
        "vse.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 2b\n\t"

        "3:\n\t"                     // pad h_mid
        "mv             t4, %4\n\t"  // t4 = in_w
        "beqz           %8, 5f\n\t"  // if pad_left = 0
        "mv             t3, %8\n\t"  // t3 = pad_left

        "4:\n\t"  // pad w_left
        "vse.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 4b\n\t"

        "5:\n\t"  // pad w_mid
        "vlse.v         v4, (a0), t0\n\t"
        "addi           a0, a0, 4\n\t"
        "vse.v          v4, (%1)\n\t"
        "addi           %1, %1, 16\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 5b\n\t"

        "beqz           %10, 7f\n\t"  // if pad_right = 0
        "mv             t3, %10\n\t"  // t3 = pad_right

        "6:\n\t"  // pad w_right
        "vse.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 6b\n\t"

        "7:\n\t"
        "addi           t5, t5, -1\n\t"
        "bnez           t5, 3b\n\t"

        "beqz           %9, 9f\n\t"  // if pad_down = 0
        "mv             t3, t2\n\t"  // t3 = num to memset 0

        "8:\n\t"  // pad h_down
        "vse.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 8b\n\t"

        "9:\n\t"
        "add            %0, %0, t6\n\t"  // input_data jump to next 4 channel

        "addi           %2, %2, -1\n\t"
        "bnez           %2, 1b\n\t"

        : "=r"(inp_ptr),    // %0
          "=r"(pad_ptr),    // %1
          "=r"(inc4),       // %2
          "=r"(inh),        // %3
          "=r"(inw),        // %4
          "=r"(padded_hw),  // %5
          "=r"(padded_w),   // %6
          "=r"(pad_top),    // %7
          "=r"(pad_left),   // %8
          "=r"(resi_h),     // %9
          "=r"(resi_w)      // %10
        : "0"(inp_ptr), "1"(pad_ptr), "2"(inc4), "3"(inh), "4"(inw), "5"(padded_hw), "6"(padded_w),
          "7"(pad_top), "8"(pad_left), "9"(resi_h), "10"(resi_w)
        : "cc", "memory", "v2", "v4", "a0", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
}

void shl_c906_crop_output_pack4to1(const float *output_trans, float *output, int out_c, int out_h,
                                   int out_w, int wino_h, int wino_w)
{
    int out_c4 = out_c / 4;
    float *out_tm_ptr = (float *)output_trans;
    float *out_ptr = output;

    asm volatile(
        "vsetvli        zero, zero, e32, m1\n\t"

        "mulw           t0, %3, %4\n\t"  // output_size per_channel
        "slli           t0, t0, 2\n\t"   // store_stride = output_size * 4

        "slli           t3, t0, 2\n\t"  // t3 = output_size * 4(channel) * 4bytes
        "slli           t4, %6, 4\n\t"  // t4 = wino_w * 4(channel) * 4

        "mulw           t5, %5, %6\n\t"  // crop_size per_channel
        "slli           t5, t5, 4\n\t"   // t5 = crop_size * 4(channel) * 4

        "1:\n\t"                     // channel loop [out_ch / 4]
        "mv             a1, %1\n\t"  // update output_addr
        "mv             a0, %0\n\t"  // update crop_addr per-channel

        "mv             t1, %3\n\t"  // t1 = out_h

        "2:\n\t"                     // crop h
        "mv             t2, %4\n\t"  // t2 = out_w
        "mv             s1, a0\n\t"  // update crop_addr per-row

        "3:\n\t"  // crop w
        "vle.v          v2, (s1)\n\t"
        "addi           s1, s1, 16\n\t"
        "vsse.v         v2, (a1), t0\n\t"
        "addi           a1, a1, 4\n\t"

        "addi           t2, t2, -1\n\t"
        "bnez           t2, 3b\n\t"

        "add            a0, a0, t4\n\t"  // crop-data jump to next row

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        "4:\n\t"
        "add            %1, %1, t3\n\t"  // output_data jump to next 4 channel
        "add            %0, %0, t5\n\t"  // crop-data jump to next 4 channel

        "addi           %2, %2, -1\n\t"
        "bnez           %2, 1b\n\t"

        : "=r"(out_tm_ptr),  // %0
          "=r"(out_ptr),     // %1
          "=r"(out_c4),      // %2
          "=r"(out_h),       // %3
          "=r"(out_w),       // %4
          "=r"(wino_h),      // %5
          "=r"(wino_w)       // %6
        : "0"(out_tm_ptr), "1"(out_ptr), "2"(out_c4), "3"(out_h), "4"(out_w), "5"(wino_h),
          "6"(wino_w)
        : "cc", "memory", "v2", "v3", "a0", "a1", "s1", "t0", "t1", "t2", "t3", "t4", "t5"

    );
}

/*
    constrain: output channel % 4 = 0
               input channel % 4 = 0
    kernel before:  [O I 3*3]
    kernel after :  [O/4 8*8 I 4]
*/
void shl_c906_conv3x3s1_winograd64_transform_kernel_pack4(struct csinn_tensor *o_kernel,
                                                          struct csinn_tensor *t_kernel)
{
    int32_t outch = o_kernel->dim[0];
    int32_t inch = o_kernel->dim[1];

    float *kernel_data = (float *)o_kernel->data;
    // for kernel transform buf, 3x3 --> 8x8
    float *kernel_tm = (float *)shl_mem_alloc(outch * inch * 8 * 8 * sizeof(float));
    // kernel transform matrix: G
    const float ktm[8][3] = {{1.0f, 0.0f, 0.0f},
                             {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                             {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                             {1.0f / 90, 1.0f / 45, 2.0f / 45},
                             {1.0f / 90, -1.0f / 45, 2.0f / 45},
                             {1.0f / 45, 1.0f / 90, 1.0f / 180},
                             {1.0f / 45, -1.0f / 90, 1.0f / 180},
                             {0.0f, 0.0f, 1.0f}};

    // const float ktm[8][3] = {
    //     {1.0f, 0.0f, 0.0f},
    //     {-2.0f / 9, -2.0f / 9, -2.0f / 9},
    //     {-2.0f / 9, 2.0f / 9, -2.0f / 9},
    //     {1.0f / 90, 1.0f / 45, 2.0f / 45},
    //     {1.0f / 90, -1.0f / 45, 2.0f / 45},
    //     {32.0f / 45, 16.0f / 45, 8.0f / 45},
    //     {32.0f / 45, -16.0f / 45, 8.0f / 45},
    //     {0.0f, 0.0f, 1.0f}
    // };

    csinn_tensor_copy(t_kernel, o_kernel);
    t_kernel->dim_count = 5;
    t_kernel->dim[0] = outch / 4;
    t_kernel->dim[1] = 8;
    t_kernel->dim[2] = 8;
    t_kernel->dim[3] = inch;
    t_kernel->dim[4] = 4;
    t_kernel->layout = CSINN_LAYOUT_O1HWIO0;

    for (int p = 0; p < outch; p++) {
        for (int q = 0; q < inch; q++) {
            const float *kernel0 = kernel_data + p * inch * 9 + q * 9;
            float *kernel_tmp = kernel_tm + p * inch * 64 + q * 64;

            // transform kernel
            const float *k0 = kernel0;
            const float *k1 = kernel0 + 3;
            const float *k2 = kernel0 + 6;

            // h : first compute the transport matrix tmp = (g * GT)T
            float tmp[8][3];
            for (int i = 0; i < 8; i++) {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 8; j++) {
                float *tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++) {
                    kernel_tmp[j * 8 + i] =
                        tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
    // optimized layout for winograd64
    float *kernel_tm_pack4 = (float *)shl_mem_alloc(outch * inch * 8 * 8 * sizeof(float));
    t_kernel->data = kernel_tm_pack4;

    for (int oc = 0; oc < outch / 4; oc++) {
        float *g0 = kernel_tm_pack4 + oc * 64 * inch * 4;

        const float *k0 = kernel_tm + oc * 64 * inch * 4;
        const float *k1 = k0 + 64 * inch;
        const float *k2 = k1 + 64 * inch;
        const float *k3 = k2 + 64 * inch;

        for (int k = 0; k < 64; k++) {
            float *g00 = g0 + k * inch * 4;

            for (int ic = 0; ic < inch / 4; ic++) {
                for (int i = 0; i < 4; i++) {
                    const float *k00 = k0 + (ic * 4 + i) * 64;
                    const float *k10 = k1 + (ic * 4 + i) * 64;
                    const float *k20 = k2 + (ic * 4 + i) * 64;
                    const float *k30 = k3 + (ic * 4 + i) * 64;

                    g00[0] = k00[k];
                    g00[1] = k10[k];
                    g00[2] = k20[k];
                    g00[3] = k30[k];

                    g00 += 4;
                }
            }
        }
    }
    o_kernel->data = NULL;
    shl_mem_free(kernel_tm);
}

/*
    constrain: output channel % 4 = 0
               input channel % 4 = 0
*/
int shl_c906_conv3x3s1_winograd64_pack4(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)params->conv_extra.kernel_tm->data;
    float *bias_data = (float *)bias->data;

    // param
    int kernel_h = kernel->dim[2];
    int kernel_w = kernel->dim[3];
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int dilation_h = params->dilation_height;
    int dilation_w = params->dilation_width;
    int pad_left = params->pad_left;
    int pad_top = params->pad_top;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int input_size = in_c * in_h * in_w;
    int kernel_size = in_c * kernel_h * kernel_w;

    int out_c = kernel->dim[0];
    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int output_size = out_c * out_h * out_w;

    // winograd param
    int block_h = (out_h + 5) / 6;
    int block_w = (out_w + 5) / 6;

    int padded_in_h =
        block_h * 6 +
        2;  // block * 4 for alignment with 4，kernel = 3 * 3 ，stride = 1，thus input_size + 2
    int padded_in_w = block_w * 6 + 2;
    int padded_in_hw = padded_in_h * padded_in_w;  // element size after padding per channel

    /****************************** bias *****************************/
    bool flag_bias = 1;  // default: conv2d layer include bias
    if (bias_data == NULL) {
        flag_bias = 0;
        bias_data = (float *)shl_mem_alloc(out_c * sizeof(float));
    }

    for (int n = 0; n < batch; n++) {
        // pad buffer: [in_c/8 h w 8]
        float *input_padd_buf = (float *)shl_mem_alloc(in_c * padded_in_hw * sizeof(float));

        // pad input
        shl_c906_pad_input_pack1to4(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h,
                                    padded_in_w, pad_top, pad_left);
        input_data += input_size;

        // input transform buffer1: [in_ch/8, 64, blocks, 8]
        float *input_tm1_buf =
            (float *)shl_mem_alloc(in_c * block_h * block_w * 8 * 8 * sizeof(float));

        /****************************** transform input *****************************/
        /*
        BT = {
            { 1   0    -5.25    0    5.25     0    -1  0 };
            { 0   1      1    -4.25  -4.25    1    1   0 };
            { 0   -1     1    4.25   -4.25   -1    1   0 };
            { 0  0.5    0.25   -2.5   -1.25     2    1   0 };
            { 0  -0.5   0.25    2.5   -1.25    -2    1   0 };
            { 0   2      4    -2.5    -5     0.5   1   0 };
            { 0   -2     4     2.5    -5    -0.5   1   0 };
            { 0   -1     0    5.25     0    -5.25  0   1 }
        };
        */

        // int in_h_tm = block_h * 8;  // input height after transform
        // int in_w_tm = block_w * 8;

        int tiles = block_h * block_w;

#pragma omp parallel for num_threads(1)
        for (int q = 0; q < in_c / 4; q++) {
            float *img0 = input_padd_buf + q * padded_in_h * padded_in_w *
                                               4;  // feature map after padding - q channel
            float *img0_tm =
                input_tm1_buf + q * 64 * tiles * 4;  // transform and interleave - q channel

            float *tmp = (float *)shl_mem_alloc(8 * 8 * 4 * sizeof(float));

            for (int i = 0; i < block_h; i++) {
                for (int j = 0; j < block_w; j++) {
                    float *r0 = img0 + (i * padded_in_w * 6 + j * 6) *
                                           4;  // feature map after padding 8*8 start addr
                    float *r0_tm =
                        img0_tm + (i * block_w + j) * 4;  // input_tm1 8*8 block start addr

                    float ratio[] = {5.25, -4.25, 0.25, -1.25, 4.0,
                                     0.5,  -2.5,  2.0};  // note: in fact cannot be output constrain
                    float *ratio_ptr = ratio;

                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "li             t0, 8\n\t"      // m = 8
                        "mv             t5, %2\n\t"     // t5 = tmp start addr
                        "slli           t1, %4, 4\n\t"  // t1 = padded_in_w * 4 * 4bytes

                        "flw            fa0, 0(%3)\n\t"   // fa0 = 5.25
                        "flw            fa1, 4(%3)\n\t"   // fa1 = -4.25
                        "flw            fa2, 8(%3)\n\t"   // fa2 = 0.25
                        "flw            fa3, 12(%3)\n\t"  // fa3 = -1.25
                        "flw            fa4, 16(%3)\n\t"  // fa4 = 4.0
                        "flw            fa5, 20(%3)\n\t"  // fa5 = 0.5
                        "flw            fa6, 24(%3)\n\t"  // fa6 = -2.5
                        "flw            fa7, 28(%3)\n\t"  // fa7 = 2.0

                        "1:\n\t"
                        "mv             s1, %0\n\t"  // s1 = r00 addr

                        "mv             a0, t5\n\t"       // tmp[0][m]
                        "addi           a1, a0, 128\n\t"  // tmp[1][m]
                        "addi           a2, a1, 128\n\t"  // tmp[2][m]
                        "addi           a3, a2, 128\n\t"  // tmp[3][m]
                        "addi           a4, a3, 128\n\t"  // tmp[4][m]
                        "addi           a5, a4, 128\n\t"  // tmp[5][m]
                        "addi           a6, a5, 128\n\t"  // tmp[6][m]
                        "addi           a7, a6, 128\n\t"  // tmp[7][m]

                        "vle.v          v0, (s1)\n\t"  // r00
                        "addi           s1, s1, 16\n\t"
                        "vle.v          v1, (s1)\n\t"  // r01
                        "addi           s1, s1, 16\n\t"
                        "vle.v          v2, (s1)\n\t"  // r02
                        "addi           s1, s1, 16\n\t"
                        "vle.v          v3, (s1)\n\t"  // r03
                        "addi           s1, s1, 16\n\t"
                        "vle.v          v4, (s1)\n\t"  // r04
                        "addi           s1, s1, 16\n\t"
                        "vle.v          v5, (s1)\n\t"  // r05
                        "addi           s1, s1, 16\n\t"
                        "vle.v          v6, (s1)\n\t"  // r06
                        "addi           s1, s1, 16\n\t"
                        "vle.v          v7, (s1)\n\t"  // r07
                        "addi           s1, s1, 16\n\t"

                        "vmv.v.v        v10, v6\n\t"

                        //---------------------------------------------
                        "vfsub.vv       v8, v4, v2\n\t"  // r04 - r02
                        "vfsub.vv       v9, v3, v5\n\t"  // r03 - r05

                        "vfsub.vv       v24, v0, v6\n\t"  // r00 - r06
                        "vfsub.vv       v31, v7, v1\n\t"  // r07 - r01

                        "vfmacc.vf      v10, fa2, v2\n\t"  // r06 + r02 * 0.25f

                        "vfmul.vf       v11, v1, fa5\n\t"  // r01 * 0.5f
                        "vfmul.vf       v12, v1, fa7\n\t"  // r01 * 2.0f

                        "vfmacc.vf      v24, fa0, v8\n\t"  // r00 - r06 + 5.25 * (r04 - r02) =
                                                           // tmp[0][m]
                        "vfmacc.vf      v31, fa0, v9\n\t"  // r07 - r01 + 5.25 * (r03 - r05) =
                                                           // tmp[7][m]

                        //---------------------------------------------
                        "vfadd.vv       v8, v2, v6\n\t"  // r02 + r06
                        "vfadd.vv       v9, v1, v5\n\t"  // r01 + r05

                        "vfmacc.vf      v11, fa6, v3\n\t"  // r01 * 0.5f - r03 * 2.5f
                        "vfmacc.vf      v12, fa6, v3\n\t"  // r01 * 2.f - r03 * 2.5f

                        "vfmacc.vf      v2, fa3, v4\n\t"   // r02 - r04 * 1.25f
                        "vfmacc.vf      v10, fa3, v4\n\t"  // r06 + r02 * 0.25f - r04 * 1.25f =
                                                           // tmp34a

                        "vfmacc.vf      v8, fa1, v4\n\t"  // r02 + r06 - r04 * 4.25f = tmp12a
                        "vfmacc.vf      v9, fa1, v3\n\t"  // r01 + r05 - r03 * 4.25f = tmp12b

                        "vfmacc.vf      v11, fa7, v5\n\t"  // r01 * 0.5f - r03 * 2.5f + r05 * 2.0 =
                                                           // tmp34b
                        "vfmacc.vf      v12, fa5, v5\n\t"  // r01 * 2.f - r03 * 2.5f + r05 * 0.5 =
                                                           // tmp56b

                        "vse.v          v24, (a0)\n\t"
                        "vse.v          v31, (a7)\n\t"

                        "vfadd.vv       v25, v8, v9\n\t"  // tmp12a + tmp12b = tmp[1][m]
                        "vfsub.vv       v26, v8, v9\n\t"  // tmp12a - tmp12b = tmp[2][m]

                        //---------------------------------------------
                        "vfmacc.vf      v6, fa4, v2\n\t"  // r06 + (r02 - r04 * 1.25f) * 4 = tmp56a

                        "vfadd.vv       v27, v10, v11\n\t"  // tmp34a + tmp34b = tmp[3][m]
                        "vfsub.vv       v28, v10, v11\n\t"  // tmp34a - tmp34b = tmp[4][m]

                        "vfadd.vv       v29, v6, v12\n\t"  // tmp56a + tmp56b = tmp[5][m]
                        "vfsub.vv       v30, v6, v12\n\t"  // tmp56a - tmp56b = tmp[6][m]

                        "vse.v          v25, (a1)\n\t"
                        "vse.v          v26, (a2)\n\t"
                        "vse.v          v27, (a3)\n\t"
                        "vse.v          v28, (a4)\n\t"
                        "vse.v          v29, (a5)\n\t"
                        "vse.v          v30, (a6)\n\t"

                        //---------------------------------------------

                        "add            %0, %0, t1\n\t"  // padding feature map 8*8 next line addr
                        "addi           t5, t5, 16\n\t"  // tmp[0][0] --> tmp[0][1]

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        "2:\n\t"

                        "mv             t5, %2\n\t"  // tmp start addr
                        "li             t0, 8\n\t"   // m = 8

                        "slli           t1, %5, 4\n\t"  // t1 = tiles * 4 * 4 bytes
                        "slli           t2, %5, 7\n\t"  // t2 = tiles * 8 * 4 * 4 bytes

                        "3:\n\t"

                        "mv             a0, %1\n\t"      // r0_tm_0
                        "add            a1, a0, t1\n\t"  // r0_tm_1
                        "add            a2, a1, t1\n\t"  // r0_tm_2
                        "add            a3, a2, t1\n\t"  // r0_tm_3
                        "add            a4, a3, t1\n\t"  // r0_tm_4
                        "add            a5, a4, t1\n\t"  // r0_tm_5
                        "add            a6, a5, t1\n\t"  // r0_tm_6
                        "add            a7, a6, t1\n\t"  // r0_tm_7

                        "vle.v          v0, (t5)\n\t"  // tmp[m][0]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v1, (t5)\n\t"  // tmp[m][1]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v2, (t5)\n\t"  // tmp[m][2]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v3, (t5)\n\t"  // tmp[m][3]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v4, (t5)\n\t"  // tmp[m][4]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v5, (t5)\n\t"  // tmp[m][5]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v6, (t5)\n\t"  // tmp[m][6]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v7, (t5)\n\t"  // tmp[m][7]
                        "addi           t5, t5, 16\n\t"

                        "vmv.v.v        v10, v6\n\t"

                        //---------------------------------------------
                        "vfsub.vv       v8, v4, v2\n\t"  // tmp04 - tmp02 (tmp[m][4] - tmp[m][2])
                        "vfsub.vv       v9, v3, v5\n\t"  // tmp03 - tmp05

                        "vfsub.vv       v24, v0, v6\n\t"  // tmp00 - tmp06
                        "vfsub.vv       v31, v7, v1\n\t"  // tmp07 - tmp01

                        "vfmacc.vf      v10, fa2, v2\n\t"  // tmp06 + tmp02 * 0.25f

                        "vfmul.vf       v11, v1, fa5\n\t"  // tmp01 * 0.5f
                        "vfmul.vf       v12, v1, fa7\n\t"  // tmp01 * 2.0f

                        "vfmacc.vf      v24, fa0, v8\n\t"  // tmp00 - tmp06 + 5.25 * (tmp04 - tmp02)
                                                           // = r0_tm_0[m]
                        "vfmacc.vf      v31, fa0, v9\n\t"  // tmp07 - tmp01 + 5.25 * (tmp03 - tmp05)
                                                           // = r0_tm_7[m]

                        //---------------------------------------------
                        "vfadd.vv       v8, v2, v6\n\t"  // tmp02 + tmp06
                        "vfadd.vv       v9, v1, v5\n\t"  // tmp01 + tmp05

                        "vfmacc.vf      v11, fa6, v3\n\t"  // tmp01 * 0.5f - tmp03 * 2.5f
                        "vfmacc.vf      v12, fa6, v3\n\t"  // tmp01 * 2.f - tmp03 * 2.5f

                        "vfmacc.vf      v2, fa3, v4\n\t"   // tmp02 - tmp04 * 1.25f
                        "vfmacc.vf      v10, fa3, v4\n\t"  // tmp06 + tmp02 * 0.25f - tmp04 * 1.25f
                                                           // = tmp34a

                        "vfmacc.vf      v8, fa1, v4\n\t"  // tmp02 + tmp06 - tmp04 * 4.25f = tmp12a
                        "vfmacc.vf      v9, fa1, v3\n\t"  // tmp01 + tmp05 - tmp03 * 4.25f = tmp12b

                        "vfmacc.vf      v11, fa7, v5\n\t"  // tmp01 * 0.5f - tmp03 * 2.5f + tmp05
                                                           // * 2.0 = tmp34b
                        "vfmacc.vf      v12, fa5, v5\n\t"  // tmp01 * 2.f - tmp03 * 2.5f + tmp05 *
                                                           // 0.5 = tmp56b

                        "vse.v          v24, (a0)\n\t"
                        "vse.v          v31, (a7)\n\t"

                        "vfadd.vv       v25, v8, v9\n\t"  // tmp12a + tmp12b = r0_tm_1[m]
                        "vfsub.vv       v26, v8, v9\n\t"  // tmp12a - tmp12b = r0_tm_2[m]

                        //---------------------------------------------
                        "vfmacc.vf      v6, fa4, v2\n\t"  // tmp06 + (tmp02 - tmp04 * 1.25f) * 4 =
                                                          // tmp56a

                        "vfadd.vv       v27, v10, v11\n\t"  // tmp34a + tmp34b = r0_tm_3[m]
                        "vfsub.vv       v28, v10, v11\n\t"  // tmp34a - tmp34b = r0_tm_4[m]

                        "vfadd.vv       v29, v6, v12\n\t"  // tmp56a + tmp56b = r0_tm_5[m]
                        "vfsub.vv       v30, v6, v12\n\t"  // tmp56a - tmp56b = r0_tm_6[m]

                        "vse.v          v25, (a1)\n\t"
                        "vse.v          v26, (a2)\n\t"
                        "vse.v          v27, (a3)\n\t"
                        "vse.v          v28, (a4)\n\t"
                        "vse.v          v29, (a5)\n\t"
                        "vse.v          v30, (a6)\n\t"

                        "add            %1, %1, t2\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 3b"

                        : "=r"(r0),           // %0
                          "=r"(r0_tm),        // %1
                          "=r"(tmp),          // %2
                          "=r"(ratio_ptr),    // %3
                          "=r"(padded_in_w),  // %4
                          "=r"(tiles)         // %5
                        : "0"(r0), "1"(r0_tm), "2"(tmp), "3"(ratio_ptr), "4"(padded_in_w),
                          "5"(tiles)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                          "v9", "v10", "v11", "v12", "v24", "v25", "v26", "v27", "v28", "v29",
                          "v30", "v31", "t0", "t1", "t2", "t5", "s1", "a0", "a1", "a2", "a3", "a4",
                          "a5", "a6", "a7", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7");
                }
            }
            shl_mem_free(tmp);
        }
        shl_mem_free(input_padd_buf);

        /*********************************** dot ***************************************/
        // reorder input_tm1_buf
        float *input_tm2_buf = (float *)shl_mem_alloc(64 * tiles * in_c * sizeof(float));

#pragma omp parallel for num_threads(1)
        for (int r = 0; r < 64; r++) {
            float *img_tm2 = input_tm2_buf + r * tiles * in_c;  // input_tm2 r channel data

            int t = 0;
            for (; t + 7 < tiles; t += 8) {
                float *tm2 = img_tm2 + t * in_c;  // img_tm2 row data
                float *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * 4;

                //-----------------
                // for (int q = 0; q < in_c / 4; q++) {
                //     for (int l = 0; l < 4; l++) {
                //         tm2[0] = tm1[l];
                //         tm2[1] = tm1[l + 4 * 1];
                //         tm2[2] = tm1[l + 4 * 2];
                //         tm2[3] = tm1[l + 4 * 3];
                //         tm2[4] = tm1[l + 4 * 4];
                //         tm2[5] = tm1[l + 4 * 5];
                //         tm2[6] = tm1[l + 4 * 6];
                //         tm2[7] = tm1[l + 4 * 7];
                //         tm2 += 8;
                //     }
                //     tm1 += 64 * tiles * 4;
                // }
                asm volatile(
                    "vsetvli        zero, zero, e32, m1\n\t"
                    "slli           t1, %2, 10\n\t"  // 64 * tiles * 4 * 4 bytes
                    "srai           t2, %3, 2\n\t"   // in_ch4

                    "1:\n\t"  // in_ch loop4

                    "mv             a0, %1\n\t"  // updata tm1 addr

                    "vle.v          v0, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v1, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v2, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v3, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v4, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v5, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v6, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v7, (a0)\n\t"

                    "vsseg8e.v      v0, (%0)\n\t"

                    "add            %1, %1, t1\n\t"
                    "addi           %0, %0, 128\n\t"

                    "addi           t2, t2, -1\n\t"
                    "bnez           t2, 1b\n\t"

                    : "=r"(tm2),    // %0
                      "=r"(tm1),    // %1
                      "=r"(tiles),  // %2
                      "=r"(in_c)    // %3
                    : "0"(tm2), "1"(tm1), "2"(tiles), "3"(in_c)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "a0", "t1",
                      "t2");
            }
            for (; t + 3 < tiles; t += 4) {
                float *tm2 = img_tm2 + t * in_c;  // img_tm2 row data
                float *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * 4;

                // for (int q = 0; q < in_c / 4; q++) {
                //     for (int l = 0; l < 4; l++) {
                //         tm2[0] = tm1[l];
                //         tm2[1] = tm1[l + 4 * 1];
                //         tm2[2] = tm1[l + 4 * 2];
                //         tm2[3] = tm1[l + 4 * 3];
                //         tm2 += 4;
                //     }
                //     tm1 += 64 * tiles * 4;
                // }
                asm volatile(
                    "vsetvli        zero, zero, e32, m1\n\t"
                    "slli           t1, %2, 10\n\t"  // 64 * tiles * 4 * 4 bytes
                    "srai           t2, %3, 2\n\t"   // in_ch4

                    "1:\n\t"  // in_ch loop4

                    "mv             a0, %1\n\t"  // updata tm1 addr

                    "vle.v          v0, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v1, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v2, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v3, (a0)\n\t"

                    "vsseg4e.v      v0, (%0)\n\t"

                    "add            %1, %1, t1\n\t"
                    "addi           %0, %0, 64\n\t"

                    "addi           t2, t2, -1\n\t"
                    "bnez           t2, 1b\n\t"

                    : "=r"(tm2),    // %0
                      "=r"(tm1),    // %1
                      "=r"(tiles),  // %2
                      "=r"(in_c)    // %3
                    : "0"(tm2), "1"(tm1), "2"(tiles), "3"(in_c)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "a0", "t1", "t2");
            }
            for (; t + 1 < tiles; t += 2) {
                float *tm2 = img_tm2 + t * in_c;  // img_tm2 row data
                float *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * 4;
                // for (int q = 0; q < in_c / 4; q++) {
                //     for (int l = 0; l < 4; l++) {
                //         tm2[0] = tm1[l];
                //         tm2[1] = tm1[l + 4];
                //         tm2 += 2;
                //     }
                //     tm1 += 64 * tiles * 4;
                // }

                asm volatile(
                    "vsetvli        zero, zero, e32, m1\n\t"
                    "slli           t1, %2, 10\n\t"  // 64 * tiles * 4 * 4 bytes
                    "srai           t2, %3, 2\n\t"   // in_ch4

                    "1:\n\t"  // in_ch loop4

                    "mv             a0, %1\n\t"  // updata tm1 addr

                    "vle.v          v0, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"
                    "vle.v          v1, (a0)\n\t"

                    "vsseg2e.v      v0, (%0)\n\t"

                    "add            %1, %1, t1\n\t"
                    "addi           %0, %0, 32\n\t"

                    "addi           t2, t2, -1\n\t"
                    "bnez           t2, 1b\n\t"

                    : "=r"(tm2),    // %0
                      "=r"(tm1),    // %1
                      "=r"(tiles),  // %2
                      "=r"(in_c)    // %3
                    : "0"(tm2), "1"(tm1), "2"(tiles), "3"(in_c)
                    : "cc", "memory", "v0", "v1", "a0", "t1", "t2");
            }
            for (; t < tiles; t++) {
                float *tm2 = img_tm2 + t * in_c;  // img_tm2 row data
                float *tm1 = input_tm1_buf;

                tm1 += (r * tiles + t) * 4;
                // for (int q = 0; q < in_c / 4; q++) {
                //     for (int l = 0; l < 4; l++) {
                //         tm2[0] = tm1[l];
                //         tm2++;
                //     }
                //     tm1 += 64 * tiles * 4;
                // }

                asm volatile(
                    "vsetvli        zero, zero, e32, m1\n\t"
                    "slli           t1, %2, 10\n\t"  // 64 * tiles * 4 * 4 bytes
                    "srai           t2, %3, 2\n\t"   // in_ch4

                    "1:\n\t"  // in_ch loop4

                    "mv             a0, %1\n\t"  // updata tm1 addr

                    "vle.v          v0, (a0)\n\t"
                    "addi           a0, a0, 16\n\t"

                    "vse.v          v0, (%0)\n\t"

                    "add            %1, %1, t1\n\t"
                    "addi           %0, %0, 16\n\t"

                    "addi           t2, t2, -1\n\t"
                    "bnez           t2, 1b\n\t"

                    : "=r"(tm2),    // %0
                      "=r"(tm1),    // %1
                      "=r"(tiles),  // %2
                      "=r"(in_c)    // %3
                    : "0"(tm2), "1"(tm1), "2"(tiles), "3"(in_c)
                    : "cc", "memory", "v0", "a0", "t1", "t2");
            }
        }

        shl_mem_free(input_tm1_buf);

        // output_dot_buf： [out_c/4, 64, blocks, 4]
        float *output_dot_buf =
            (float *)shl_mem_alloc(out_c * block_h * block_w * 8 * 8 * sizeof(float));

#pragma omp parallel for num_threads(1)
        for (int p = 0; p < out_c / 4; p++) {
            float *output0_tm = output_dot_buf + p * 64 * tiles * 4;  // 4 channel dot output
            float *kernel0_tm = kernel_data + p * 64 * in_c * 4;      // 4 channel kernel

            for (int r = 0; r < 64; r++) {
                float *img_tm2 = input_tm2_buf + r * tiles * in_c;  // img_tm2 第r个channel

                int t = 0;
                for (; t + 7 < tiles; t += 8) {
                    float *r0 = img_tm2 + t * in_c;
                    float *k0 = kernel0_tm + r * in_c * 4;

                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %3\n\t"  // t0 = in_c

                        "vmv.v.x        v0, zero\n\t"
                        "vmv.v.x        v1, zero\n\t"
                        "vmv.v.x        v2, zero\n\t"
                        "vmv.v.x        v3, zero\n\t"
                        "vmv.v.x        v4, zero\n\t"
                        "vmv.v.x        v5, zero\n\t"
                        "vmv.v.x        v6, zero\n\t"
                        "vmv.v.x        v7, zero\n\t"  // clear

                        "1:\n\t"

                        "flw            fa0, (%0)\n\t"
                        "flw            fa1, 4(%0)\n\t"
                        "flw            fa2, 8(%0)\n\t"
                        "flw            fa3, 12(%0)\n\t"
                        "flw            fa4, 16(%0)\n\t"
                        "flw            fa5, 20(%0)\n\t"
                        "flw            fa6, 24(%0)\n\t"
                        "flw            fa7, 28(%0)\n\t"
                        "addi           %0, %0, 32\n\t"

                        "vle.v          v8, (%1)\n\t"
                        "addi           %1, %1, 16\n\t"

                        "vfmacc.vf      v0, fa0, v8\n\t"
                        "vfmacc.vf      v1, fa1, v8\n\t"
                        "vfmacc.vf      v2, fa2, v8\n\t"
                        "vfmacc.vf      v3, fa3, v8\n\t"
                        "vfmacc.vf      v4, fa4, v8\n\t"
                        "vfmacc.vf      v5, fa5, v8\n\t"
                        "vfmacc.vf      v6, fa6, v8\n\t"
                        "vfmacc.vf      v7, fa7, v8\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        "vse.v          v0, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v1, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v2, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v3, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v4, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v5, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v6, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v7, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"

                        : "=r"(r0),          // %0
                          "=r"(k0),          // %1
                          "=r"(output0_tm),  // %2
                          "=r"(in_c)         // %3
                        : "0"(r0), "1"(k0), "2"(output0_tm), "3"(in_c)

                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                          "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "t0"

                    );
                }
                for (; t + 3 < tiles; t += 4) {
                    float *r0 = img_tm2 + t * in_c;
                    float *k0 = kernel0_tm + r * in_c * 4;

                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %3\n\t"  // t0 = in_c
                        "vmv.v.x        v0, zero\n\t"
                        "vmv.v.x        v1, zero\n\t"
                        "vmv.v.x        v2, zero\n\t"
                        "vmv.v.x        v3, zero\n\t"  // clear

                        "1:\n\t"

                        "flw            fa0, (%0)\n\t"
                        "flw            fa1, 4(%0)\n\t"
                        "flw            fa2, 8(%0)\n\t"
                        "flw            fa3, 12(%0)\n\t"
                        "addi           %0, %0, 16\n\t"

                        "vle.v          v4, (%1)\n\t"
                        "addi           %1, %1, 16\n\t"

                        "vfmacc.vf      v0, fa0, v4\n\t"
                        "vfmacc.vf      v1, fa1, v4\n\t"
                        "vfmacc.vf      v2, fa2, v4\n\t"
                        "vfmacc.vf      v3, fa3, v4\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        "vse.v          v0, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v1, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v2, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v3, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"

                        : "=r"(r0),          // %0
                          "=r"(k0),          // %1
                          "=r"(output0_tm),  // %2
                          "=r"(in_c)         // %3
                        : "0"(r0), "1"(k0), "2"(output0_tm), "3"(in_c)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "fa0", "fa1", "fa2", "fa3",
                          "t0");
                }
                for (; t + 1 < tiles; t += 2) {
                    float *r0 = img_tm2 + t * in_c;
                    float *k0 = kernel0_tm + r * in_c * 4;

                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %3\n\t"  // t0 = in_c
                        "vmv.v.x        v0, zero\n\t"
                        "vmv.v.x        v1, zero\n\t"  // clear

                        "1:\n\t"

                        "flw            fa0, (%0)\n\t"
                        "flw            fa1, 4(%0)\n\t"
                        "addi           %0, %0, 8\n\t"

                        "vle.v          v2, (%1)\n\t"
                        "addi           %1, %1, 16\n\t"

                        "vfmacc.vf      v0, fa0, v2\n\t"
                        "vfmacc.vf      v1, fa1, v2\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        "vse.v          v0, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"
                        "vse.v          v1, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"

                        : "=r"(r0),          // %0
                          "=r"(k0),          // %1
                          "=r"(output0_tm),  // %2
                          "=r"(in_c)         // %3
                        : "0"(r0), "1"(k0), "2"(output0_tm), "3"(in_c)
                        : "cc", "memory", "v0", "v1", "v2", "fa0", "fa1", "t0");
                }
                for (; t < tiles; t++) {
                    float *r0 = img_tm2 + t * in_c;
                    float *k0 = kernel0_tm + r * in_c * 4;

                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %3\n\t"    // t0 = in_c
                        "vmv.v.x        v0, zero\n\t"  // clear

                        "1:\n\t"

                        "flw            fa0, (%0)\n\t"
                        "addi           %0, %0, 4\n\t"

                        "vle.v          v1, (%1)\n\t"
                        "addi           %1, %1, 16\n\t"

                        "vfmacc.vf      v0, fa0, v1\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        "vse.v          v0, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"

                        : "=r"(r0),          // %0
                          "=r"(k0),          // %1
                          "=r"(output0_tm),  // %2
                          "=r"(in_c)         // %3
                        : "0"(r0), "1"(k0), "2"(output0_tm), "3"(in_c)
                        : "cc", "memory", "v0", "v1", "fa0", "t0");
                }
            }
        }

        shl_mem_free(input_tm2_buf);
        /*************************** transform output ****************************/
        // output_tm1_buf: [out_c/4, out_h6, out_w6, 4]
        float *output_tm1_buf =
            (float *)shl_mem_alloc(out_c * block_h * block_w * 6 * 6 * sizeof(float));

        /*
        AT = {
            { 1  1  1   1    1    1      1    0 };
            { 0  1  -1  2   -2   1/2   -1/2   0 };
            { 0  1  1   4    4   1/4    1/4   0 };
            { 0  1  -1  8   -8   1/8   -1/8   0 };
            { 0  1  1   16  16   1/16  1/16   0 };
            { 0  1  -1  32  -32  1/32  -1/32  1 }
        };
        AT = {
            { 1  1  1   1    1   32    32   0 };
            { 0  1  -1  2   -2   16   -16   0 };
            { 0  1  1   4    4   8     8    0 };
            { 0  1  -1  8   -8   4    -4    0 };
            { 0  1  1   16  16   2     2    0 };
            { 0  1  -1  32  -32  1    -1    1 }
        };
        */

#pragma omp parallel for num_threads(1)
        for (int p = 0; p < out_c / 4; p++) {
            float *bias_tmp = bias_data + p * 4;

            float *out0_tm =
                output_dot_buf + p * 64 * block_h * block_w * 4;  // 输出转换前/dot后 第p个channel
            float *out0 =
                output_tm1_buf + p * 6 * block_h * 6 * block_w * 4;  // 转换后输出 第p个channel

            float *tmp1 = (float *)shl_mem_alloc(6 * 8 * 4 * sizeof(float));
            int out_w6 = block_w * 6;

            for (int i = 0; i < block_h; i++) {
                for (int j = 0; j < block_w; j++) {
                    float *output0_tm_0 = out0_tm + (i * block_w + j) * 4;  // 8*8 起始地址

                    float *output0 =
                        out0 + (i * block_w * 6 * 6 + j * 6) * 4;  // 输出 6*6 的起始地址

                    float ratio[] = {2.0, 4.0, 8.0, 16.0, 32.0};
                    float *ratio_ptr = ratio;

                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "li             t0, 8\n\t"      // m = 8
                        "mv             t5, %2\n\t"     // t5 = tmp start addr
                        "slli           t1, %4, 4\n\t"  // t1 = tiles * 4 * 4
                        "slli           t2, %4, 7\n\t"  // t2 = tiles * 8 * 4 * 4 bytes

                        "flw            fa0, 0(%3)\n\t"   // fa0 = 2
                        "flw            fa1, 4(%3)\n\t"   // fa1 = 4
                        "flw            fa2, 8(%3)\n\t"   // fa2 = 8
                        "flw            fa3, 12(%3)\n\t"  // fa3 = 16
                        "flw            fa4, 16(%3)\n\t"  // fa4 = 32

                        "mv             s1, %0\n\t"

                        "1:\n\t"  // shape : [6 * 8] * [8 * 8] = [6 * 8]

                        "mv             a0, t5\n\t"       // tmp[0][m]
                        "addi           a1, a0, 128\n\t"  // tmp[1][m]
                        "addi           a2, a1, 128\n\t"  // tmp[2][m]
                        "addi           a3, a2, 128\n\t"  // tmp[3][m]
                        "addi           a4, a3, 128\n\t"  // tmp[4][m]
                        "addi           a5, a4, 128\n\t"  // tmp[5][m]

                        "vle.v          v0, (s1)\n\t"  // r00
                        "add            s1, s1, t1\n\t"
                        "vle.v          v1, (s1)\n\t"  // r01
                        "add            s1, s1, t1\n\t"
                        "vle.v          v2, (s1)\n\t"  // r02
                        "add            s1, s1, t1\n\t"
                        "vle.v          v3, (s1)\n\t"  // r03
                        "add            s1, s1, t1\n\t"
                        "vle.v          v4, (s1)\n\t"  // r04
                        "add            s1, s1, t1\n\t"
                        "vle.v          v5, (s1)\n\t"  // r05
                        "add            s1, s1, t1\n\t"
                        "vle.v          v6, (s1)\n\t"  // r06
                        "add            s1, s1, t1\n\t"
                        "vle.v          v7, (s1)\n\t"  // r07
                        "add            s1, s1, t1\n\t"

                        //---------------------------------------------
                        "vfadd.vv       v8, v1, v2\n\t"  // r01 + r02 = tmp024a
                        "vfsub.vv       v9, v1, v2\n\t"  // r01 - r02 = tmp135a

                        "vfadd.vv       v10, v3, v4\n\t"  // r03 + r04 = tmp024b
                        "vfsub.vv       v11, v3, v4\n\t"  // r03 - r04 = tmp135b

                        "vfadd.vv       v12, v5, v6\n\t"  // r05 + r06 = tmp024c
                        "vfsub.vv       v13, v5, v6\n\t"  // r05 - r06 = tmp135c

                        "vfadd.vv       v0, v0, v8\n\t"  // r00 + tmp024a
                        "vfadd.vv       v7, v7, v9\n\t"  // r07 + tmp135a
                        "vmv.v.v        v14, v10\n\t"    // v14 = tmp024b

                        "vmv.v.v        v26, v8\n\t"  // v26 = tmp024a
                        "vmv.v.v        v28, v8\n\t"  // v28 = tmp024a

                        "vfmacc.vf      v26, fa1, v10\n\t"  // tmp024a + tmp024b * 4
                        "vfmacc.vf      v14, fa4, v12\n\t"  // tmp024b + tmp024c * 32
                        "vfmacc.vf      v28, fa3, v10\n\t"  // tmp024a + tmp024b * 16

                        "vmv.v.v        v15, v13\n\t"      // v15 = tmp135c
                        "vmv.v.v        v25, v9\n\t"       // v25 = tmp135a
                        "vmv.v.v        v27, v9\n\t"       // v27 = tmp135a
                        "vfadd.vv       v24, v0, v14\n\t"  // r00 + tmp024a + tmp024b + tmp024c * 32
                                                           // = tmp[0][m]

                        "vfmacc.vf      v25, fa0, v11\n\t"  // tmp135a + tmp135b * 2
                        "vfmacc.vf      v27, fa2, v11\n\t"  // tmp135a + tmp135b * 8

                        //---------------------------------------------
                        "vse.v          v24, (a0)\n\t"

                        "vfmacc.vf      v26, fa2, v12\n\t"  // tmp024a + tmp024b * 4 + tmp024c * 8 =
                                                            // tmp[2][m]
                        "vfmacc.vf      v28, fa0, v12\n\t"  // tmp024a + tmp024b * 16 + tmp024c +
                                                            // tmp024c = tmp[4][m]
                        "vfmacc.vf      v15, fa4, v11\n\t"  // tmp135b * 32 + tmp135c

                        "vse.v          v26, (a2)\n\t"
                        "vse.v          v28, (a4)\n\t"

                        //---------------------------------------------
                        "vfmacc.vf      v25, fa3, v13\n\t"  // tmp135a + tmp135b * 2 + tmp135c * 16
                                                            // = tmp[1][m]
                        "vfmacc.vf      v27, fa1, v13\n\t"  // tmp135a + tmp135b * 8 + tmp135c * 4 =
                                                            // tmp[3][m]

                        "vfadd.vv       v29, v7, v15\n\t"  // r07 + tmp135a + tmp135b * 32 + tmp135c

                        "vse.v          v25, (a1)\n\t"
                        "vse.v          v27, (a3)\n\t"
                        "vse.v          v29, (a5)\n\t"

                        "addi           t5, t5, 16\n\t"  // tmp[0][0] --> tmp[0][1]

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        "2:\n\t"

                        "mv             t5, %2\n\t"     // tmp start addr
                        "li             t0, 6\n\t"      // m = 6
                        "slli           t1, %5, 4\n\t"  // t1 = out_w6 * 4 * 4bytes
                        "vle.v          v16, (%6)\n\t"  // load 4 channel bias data

                        "3:\n\t"  // shape : [6 * 8] * [6 * 8] = [6 * 6]

                        "mv             a0, %1\n\t"
                        "addi           a1, a0, 16\n\t"
                        "addi           a2, a1, 16\n\t"
                        "addi           a3, a2, 16\n\t"
                        "addi           a4, a3, 16\n\t"
                        "addi           a5, a4, 16\n\t"

                        "vle.v          v0, (t5)\n\t"  // tmp[m][0]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v1, (t5)\n\t"  // tmp[m][1]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v2, (t5)\n\t"  // tmp[m][2]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v3, (t5)\n\t"  // tmp[m][3]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v4, (t5)\n\t"  // tmp[m][4]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v5, (t5)\n\t"  // tmp[m][5]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v6, (t5)\n\t"  // tmp[m][6]
                        "addi           t5, t5, 16\n\t"
                        "vle.v          v7, (t5)\n\t"  // tmp[m][7]
                        "addi           t5, t5, 16\n\t"

                        //---------------------------------------------
                        "vfadd.vv       v8, v1, v2\n\t"  // tmp[m][1] + tmp[m][2] = tmp024a
                        "vfsub.vv       v9, v1, v2\n\t"  // tmp[m][1] - tmp[m][2] = tmp135a

                        "vfadd.vv       v10, v3, v4\n\t"  // tmp[m][3] + tmp[m][4] = tmp024b
                        "vfsub.vv       v11, v3, v4\n\t"  // tmp[m][3] - tmp[m][4] = tmp135b

                        "vfadd.vv       v12, v5, v6\n\t"  // tmp[m][5] + tmp[m][6] = tmp024c
                        "vfsub.vv       v13, v5, v6\n\t"  // tmp[m][5] - tmp[m][6] = tmp135c

                        "vfadd.vv       v0, v0, v8\n\t"  // tmp[m][0] + tmp024a
                        "vfadd.vv       v7, v7, v9\n\t"  // tmp[m][7] + tmp135a
                        "vmv.v.v        v14, v10\n\t"    // v14 = tmp024b

                        "vmv.v.v        v26, v8\n\t"  // v26 = tmp024a
                        "vmv.v.v        v28, v8\n\t"  // v28 = tmp024a

                        "vfmacc.vf      v26, fa1, v10\n\t"  // tmp024a + tmp024b * 4
                        "vfmacc.vf      v14, fa4, v12\n\t"  // tmp024b + tmp024c * 32
                        "vfmacc.vf      v28, fa3, v10\n\t"  // tmp024a + tmp024b * 16

                        "vmv.v.v        v15, v13\n\t"      // v15 = tmp135c
                        "vmv.v.v        v25, v9\n\t"       // v25 = tmp135a
                        "vmv.v.v        v27, v9\n\t"       // v27 = tmp135a
                        "vfadd.vv       v24, v0, v14\n\t"  // tmp[m][0] + tmp024a + tmp024b +
                                                           // tmp024c * 32 = tmp[0][m]

                        "vfmacc.vf      v25, fa0, v11\n\t"  // tmp135a + tmp135b * 2
                        "vfmacc.vf      v27, fa2, v11\n\t"  // tmp135a + tmp135b * 8

                        //---------------------------------------------
                        "vfadd.vv       v24, v24, v16\n\t"  // + bias

                        "vfmacc.vf      v26, fa2, v12\n\t"  // tmp024a + tmp024b * 4 + tmp024c * 8 =
                                                            // tmp[2][m]
                        "vfmacc.vf      v28, fa0, v12\n\t"  // tmp024a + tmp024b * 16 + tmp024c +
                                                            // tmp024c = tmp[4][m]
                        "vfmacc.vf      v15, fa4, v11\n\t"  // tmp135b * 32 + tmp135c

                        "vse.v          v24, (a0)\n\t"

                        "vfmacc.vf      v25, fa3, v13\n\t"  // tmp135a + tmp135b * 2 + tmp135c * 16
                                                            // = tmp[1][m]
                        "vfmacc.vf      v27, fa1, v13\n\t"  // tmp135a + tmp135b * 8 + tmp135c * 4 =
                                                            // tmp[3][m]

                        "vfadd.vv       v26, v26, v16\n\t"  // + bias
                        "vfadd.vv       v28, v28, v16\n\t"  // + bias

                        "vfadd.vv       v29, v7, v15\n\t"  // tmp[m][7] + tmp135a + tmp135b * 32 +
                                                           // tmp135c

                        "vse.v          v26, (a2)\n\t"
                        "vse.v          v28, (a4)\n\t"

                        //---------------------------------------------

                        "vfadd.vv       v25, v25, v16\n\t"  // + bias
                        "vfadd.vv       v27, v27, v16\n\t"  // + bias
                        "vfadd.vv       v29, v29, v16\n\t"  // + bias

                        "vse.v          v25, (a1)\n\t"
                        "vse.v          v27, (a3)\n\t"
                        "vse.v          v29, (a5)\n\t"

                        "add            %1, %1, t1\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 3b"

                        : "=r"(output0_tm_0),  // %0
                          "=r"(output0),       // %1
                          "=r"(tmp1),          // %2
                          "=r"(ratio_ptr),     // %3
                          "=r"(tiles),         // %4
                          "=r"(out_w6),        // %5
                          "=r"(bias_tmp)       // %6
                        : "0"(output0_tm_0), "1"(output0), "2"(tmp1), "3"(ratio_ptr), "4"(tiles),
                          "5"(out_w6), "6"(bias_tmp)

                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v24", "v25",
                          "v26", "v27", "v28", "v29", "t0", "t1", "t2", "t5", "s1", "a0", "a1",
                          "a2", "a3", "a4", "a5", "fa0", "fa1", "fa2", "fa3", "fa4");
                }
            }
            shl_mem_free(tmp1);
        }

        shl_mem_free(output_dot_buf);
        // crop the output after transform: cut extra part (right , bottom)
        shl_c906_crop_output_pack4to1(output_tm1_buf, output_data, out_c, out_h, out_w, block_h * 6,
                                      block_w * 6);
        output_data += output_size;
        shl_mem_free(output_tm1_buf);
    }

    if (!flag_bias) {
        shl_mem_free(bias_data);
        bias_data = NULL;
    }
    return CSINN_TRUE;
}
