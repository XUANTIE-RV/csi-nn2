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

/* CSI-NN2 version 1.13.x */

#include "./valid_data/dwconv2d.dat"
#include "csi_nn.h"
#include "csi_thead_rvv.h"
#include "math_snr.h"
#include "test_utils.h"

void verify_dwconv2d(void *input_data, void *kernel_data, void *bias_data, void *ref_data,
                     int (*func)(), int in_c, int in_h, int in_w, int out_c, int out_h, int out_w,
                     int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
                     enum csinn_dtype_enum dtype)
{
    struct csi_tensor *input = csi_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_c;
    input->dim[2] = in_h;
    input->dim[3] = in_w;
    input->dim_count = 4;
    input->name = "input";
    int in_size = csi_tensor_size(input);

    struct csi_tensor *kernel = csi_alloc_tensor(NULL);
    kernel->dim[0] = in_c;
    kernel->dim[1] = 1;
    kernel->dim[2] = kernel_h;
    kernel->dim[3] = kernel_w;
    kernel->dim_count = 4;
    kernel->name = "kernel";

    struct csi_tensor *bias = csi_alloc_tensor(NULL);
    bias->dim[0] = in_c;
    bias->dim_count = 1;
    bias->name = "bias";

    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = input->dim[0];
    output->dim[1] = out_c;
    output->dim[2] = out_h;
    output->dim[3] = out_w;
    output->dim_count = 4;
    output->name = "output";
    int out_size = csi_tensor_size(output);

    struct conv2d_params params;
    params.base.name = "params";
    params.stride_height = stride_h;
    params.stride_width = stride_w;
    params.pad_left = pad_w;
    params.pad_right = pad_w;
    params.pad_top = pad_h;
    params.pad_down = pad_h;

    input->data = input_data;
    kernel->data = kernel_data;
    bias->data = bias_data;
    output->data = csi_mem_alloc(out_size * sizeof(float));

    func(input, output, kernel, bias, &params);

    evaluate_error(output->data, ref_data, out_size, dtype);

    csi_free_tensor(input);
    csi_mem_free(output->data);
    csi_free_tensor(output);
    csi_free_tensor(kernel);
    csi_free_tensor(bias);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of depthwise_convolution for RVV.\n");
    verify_dwconv2d(dwconv3x3s1_fp32_in, dwconv3x3s1_fp32_ker, dwconv3x3s1_fp32_bias,
                    dwconv3x3s1_fp32_out, csi_nn_rvv_dwconv3x3s1_fp32, 2, 4, 10, 2, 4, 10, 3, 3, 1,
                    1, 1, 1, CSINN_DTYPE_FLOAT32);
    verify_dwconv2d(dwconv3x3s1_fp16_in, dwconv3x3s1_fp16_ker, dwconv3x3s1_fp16_bias,
                    dwconv3x3s1_fp16_out, csi_nn_rvv_dwconv3x3s1_fp16, 2, 4, 10, 2, 4, 10, 3, 3, 1,
                    1, 1, 1, CSINN_DTYPE_FLOAT16);
    // verify_dwconv2d(dwconv3x3s1_int8_in, dwconv3x3s1_int8_ker, dwconv3x3s1_int8_bias,
    //                 dwconv3x3s1_int8_out, csi_nn_rvv_dwconv3x3s1_int8, 2, 4, 10, 2, 4, 10, 3, 3,
    //                 1, 1, 1, 1, CSINN_DTYPE_INT8);

    verify_dwconv2d(dwconv3x3s2_fp32_in, dwconv3x3s2_fp32_ker, dwconv3x3s2_fp32_bias,
                    dwconv3x3s2_fp32_out, csi_nn_rvv_dwconv3x3s2_fp32, 2, 6, 18, 2, 3, 9, 3, 3, 2,
                    2, 1, 1, CSINN_DTYPE_FLOAT32);
    verify_dwconv2d(dwconv3x3s2_fp16_in, dwconv3x3s2_fp16_ker, dwconv3x3s2_fp16_bias,
                    dwconv3x3s2_fp16_out, csi_nn_rvv_dwconv3x3s2_fp16, 2, 6, 18, 2, 3, 9, 3, 3, 2,
                    2, 1, 1, CSINN_DTYPE_FLOAT16);
    // verify_dwconv2d(dwconv3x3s2_int8_in, dwconv3x3s2_int8_ker, dwconv3x3s2_int8_bias,
    //                 dwconv3x3s2_int8_out, csi_nn_rvv_dwconv3x3s2_int8, 2, 6, 18, 2, 3, 9, 3, 3,
    //                 2, 2, 1, 1, CSINN_DTYPE_INT8);

    return done_testing();
}
