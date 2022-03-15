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

#include "./valid_data/avgpool.dat"
#include "csi_nn.h"
#include "csi_thead_rvv.h"
#include "math_snr.h"
#include "test_utils.h"

void verify_avgpool2d(void *input_data, void *ref_data, int (*func)(), int in_c, int in_h, int in_w,
                      int out_c, int out_h, int out_w, int kernel_h, int kernel_w, int stride_h,
                      int stride_w, int pad_h, int pad_w, enum csinn_dtype_enum dtype)
{
    struct csi_tensor *input = csi_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_c;
    input->dim[2] = in_h;
    input->dim[3] = in_w;
    input->dim_count = 4;
    input->name = "input";
    int in_size = csi_tensor_size(input);

    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = input->dim[0];
    output->dim[1] = out_c;
    output->dim[2] = out_h;
    output->dim[3] = out_w;
    output->dim_count = 4;
    output->name = "output";
    int out_size = csi_tensor_size(output);

    struct pool_params params;
    params.base.name = "params";
    params.ceil_mode = 0;
    params.stride_height = stride_h;
    params.stride_width = stride_w;
    params.filter_height = kernel_h;
    params.filter_width = kernel_w;
    params.pad_left = pad_w;
    params.pad_right = pad_w;
    params.pad_top = pad_h;
    params.pad_down = pad_h;
    params.count_include_pad = 1;

    input->data = input_data;
    output->data = csi_mem_alloc(out_size * sizeof(float));

    func(input, output, &params);

    evaluate_error(output->data, ref_data, out_size, dtype);

    csi_free_tensor(input);
    csi_mem_free(output->data);
    csi_free_tensor(output);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of avgpool for RVV.\n");
    verify_avgpool2d(avgpool2x2s2_fp32_in, avgpool2x2s2_fp32_out, csi_nn_rvv_avgpool2x2s2_fp32, 2,
                     6, 18, 2, 3, 9, 2, 2, 2, 2, 0, 0, CSINN_DTYPE_FLOAT32);
    verify_avgpool2d(avgpool2x2s2_fp16_in, avgpool2x2s2_fp16_out, csi_nn_rvv_avgpool2x2s2_fp16, 2,
                     6, 18, 2, 3, 9, 2, 2, 2, 2, 0, 0, CSINN_DTYPE_FLOAT16);

    verify_avgpool2d(avgpool2x2s2_p1_fp32_in, avgpool2x2s2_p1_fp32_out,
                     csi_nn_rvv_avgpool2x2s2_p1_fp32, 2, 7, 19, 2, 4, 10, 2, 2, 2, 2, 1, 1,
                     CSINN_DTYPE_FLOAT32);
    verify_avgpool2d(avgpool2x2s2_p1_fp16_in, avgpool2x2s2_p1_fp16_out,
                     csi_nn_rvv_avgpool2x2s2_p1_fp16, 2, 7, 19, 2, 4, 10, 2, 2, 2, 2, 1, 1,
                     CSINN_DTYPE_FLOAT16);

    verify_avgpool2d(avgpool3x3s2_fp32_in, avgpool3x3s2_fp32_out, csi_nn_rvv_avgpool3x3s2_fp32, 2,
                     7, 19, 2, 3, 9, 3, 3, 2, 2, 0, 0, CSINN_DTYPE_FLOAT32);
    verify_avgpool2d(avgpool3x3s2_fp16_in, avgpool3x3s2_fp16_out, csi_nn_rvv_avgpool3x3s2_fp16, 2,
                     7, 19, 2, 3, 9, 3, 3, 2, 2, 0, 0, CSINN_DTYPE_FLOAT16);

    verify_avgpool2d(avgpool3x3s2_p1_fp32_in, avgpool3x3s2_p1_fp32_out,
                     csi_nn_rvv_avgpool3x3s2_p1_fp32, 2, 6, 18, 2, 3, 9, 3, 3, 2, 2, 1, 1,
                     CSINN_DTYPE_FLOAT32);
    verify_avgpool2d(avgpool3x3s2_p1_fp16_in, avgpool3x3s2_p1_fp16_out,
                     csi_nn_rvv_avgpool3x3s2_p1_fp16, 2, 6, 18, 2, 3, 9, 3, 3, 2, 2, 1, 1,
                     CSINN_DTYPE_FLOAT16);

    verify_avgpool2d(avgpool3x3s1_p1_fp32_in, avgpool3x3s1_p1_fp32_out,
                     csi_nn_rvv_avgpool3x3s1_p1_fp32, 2, 3, 10, 2, 3, 10, 3, 3, 1, 1, 1, 1,
                     CSINN_DTYPE_FLOAT32);
    verify_avgpool2d(avgpool3x3s1_p1_fp16_in, avgpool3x3s1_p1_fp16_out,
                     csi_nn_rvv_avgpool3x3s1_p1_fp16, 2, 3, 10, 2, 3, 10, 3, 3, 1, 1, 1, 1,
                     CSINN_DTYPE_FLOAT16);

    verify_avgpool2d(global_avgpool_fp32_in, global_avgpool_fp32_out,
                     csi_nn_rvv_global_avgpool2d_fp32, 3, 7, 7, 3, 1, 1, 7, 7, 1, 1, 0, 0,
                     CSINN_DTYPE_FLOAT32);
    verify_avgpool2d(global_avgpool_fp16_in, global_avgpool_fp16_out,
                     csi_nn_rvv_global_avgpool2d_fp16, 3, 7, 7, 3, 1, 1, 7, 7, 1, 1, 0, 0,
                     CSINN_DTYPE_FLOAT16);

    return done_testing();
}