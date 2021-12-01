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

#include "csi_c906.h"

static int csi_c906_prelu_nhwc_f32(struct csi_tensor *input,
                                   struct csi_tensor *alpha,
                                   struct csi_tensor *output,
                                   struct prelu_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *alpha_data = (float *)alpha->data;
    int outer_size = output->dim[0] * output->dim[1] * output->dim[2];
    int inner_size = output->dim[3];
    float gata = 0.0f;
    asm volatile(
                "mv         a0, %0\n\t"     // a0 = outpt_data
                "mv         a2, %2\n\t"     // a2 = input_data
                "outer_loop:\n\t"
                "mv         t1, %5\n\t"     // t1 = inner_size
                "mv         a1, %3\n\t"     // a1 = alpha_data
                "inner_loop:\n\t"
                "vsetvli    t0, t1, e32, m1\n\t"
                "vlw.v      v2, (a2)\n\t"   // load input_data to v2,v3
                "sub        t1, t1, t0\n\t"
                "slli       t0, t0, 2\n\t"
                "add        a2, a2, t0\n\t"
                "vlw.v      v4, (a1)\n\t"   // load alpha_data to v4,v5
                "add        a1, a1, t0\n\t"
                "vmflt.vf   v0, v2, %6\n\t"
                "vfmul.vv   v2, v2, v4, v0.t\n\t"
                "vsw.v      v2, (a0)\n\t"
                "add        a0, a0, t0\n\t"
                "bnez       t1, inner_loop\n\t" // finish all channel

                "addi       %4, %4, -1\n\t"
                "bnez       %4, outer_loop\n\t"

                :"=r"(output_data)  // %0
                :"0"(output_data),  // %1
                "r"(input_data),    // %2
                "r"(alpha_data),    // %3
                "r"(outer_size),    // %4
                "r"(inner_size),    // %5
                "f"(gata)           // %6
                : "v0", "v2", "v3", "v4", "v5", "t0", "t1", "a0", "a1", "a2"
    );

    // for (int b = 0; b < output->dim[0]; ++b) {
    //     for (int y = 0; y < output->dim[1]; ++y) {
    //         for (int x = 0; x < output->dim[2]; ++x) {
    //             for (int c = 0; c < output->dim[3]; ++c) {
    //                 int output_index = csi_ref_get_index(output->dim, b, y, x, c);
    //                 int input_index = csi_ref_get_index(input->dim, b, y, x, c);
    //                 float input_value = input_data[input_index];
    //                 if (input_value >= 0) {
    //                     output_data[output_index] = input_data[input_index];
    //                 } else {
    //                     output_data[output_index] = input_value * alpha_data[c];
    //                 }
    //             }
    //         }
    //     }
    // }
    return CSINN_TRUE;
}

static int csi_c906_prelu_nchw_f32(struct csi_tensor *input,
                                   struct csi_tensor *alpha,
                                   struct csi_tensor *output,
                                   struct prelu_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *alpha_data = (float *)alpha->data;

    int batch = output->dim[0];
    int channel = output->dim[1];
    int size = output->dim[2] * output->dim[3];
    float gata = 0.0f;

    asm volatile(
                "mv         t3, %4\n\t"     // t3 = batch
                "mv         a2, %2\n\t"     // a2 = input_data
                "mv         a0, %0\n\t"     // a0 = output_data
                "loop3:\n\t"
                "mv         t2, %5\n\t"     // t2 = channel
                "mv         a1, %3\n\t"     // a1 = alpha_data
                "loop2:\n\t"
                "mv         t1, %6\n\t"     // t1 = size;
                "flw        ft0, (a1)\n\t"
                "addi       a1, a1, 4\n\t"
                "loop1:\n\t"
                "vsetvli    t0, t1, e32, m2\n\t"
                "vlw.v      v2, (a2)\n\t"
                "sub        t1, t1, t0\n\t"
                "slli       t0, t0, 2\n\t"
                "add        a2, a2, t0\n\t"
                "vmflt.vf   v0, v2, %7\n\t"
                "vfmul.vf   v2, v2, ft0, v0.t\n\t"
                "vsw.v      v2, (a0)\n\t"
                "add        a0, a0, t0\n\t"
                "bnez       t1, loop1\n\t"

                "addi       t2, t2, -1\n\t"
                "bnez       t2, loop2\n\t"

                "addi       t3, t3, -1\n\t"
                "bnez       t3, loop3\n\t"

                :"=r"(output_data)  // %0
                :"0"(output_data),  // %1
                "r"(input_data),    // %2
                "r"(alpha_data),    // %3
                "r"(batch),         // %4
                "r"(channel),       // %5
                "r"(size),          // %6
                "f"(gata)           // %7
                : "v0", "v2", "v3", "t0", "t1", "t2", "t3", "a0", "a1", "a2", "ft0"
    );

    return CSINN_TRUE;
}

int csi_c906_prelu_f32(struct csi_tensor *input,
                       struct csi_tensor *alpha,
                       struct csi_tensor *output,
                       struct prelu_params *params)
{
    if (params->base.layout == CSINN_NCHW) {
        csi_c906_prelu_nchw_f32(input, alpha, output, params);
    } else if (params->base.layout == CSINN_NHWC) {
        csi_c906_prelu_nhwc_f32(input, alpha, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}
