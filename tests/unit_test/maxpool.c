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

#include "./valid_data/maxpool.dat"

#include "csi_nn.h"
#include "shl_thead_rvv.h"
#include "test_utils.h"

void verify_maxpool2d(void *input_data, void *ref_data, int (*func)(), int in_c, int in_h, int in_w,
                      int out_c, int out_h, int out_w, int kernel_h, int kernel_w, int stride_h,
                      int stride_w, int pad_h, int pad_w, enum csinn_dtype_enum dtype)
{
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_c;
    input->dim[2] = in_h;
    input->dim[3] = in_w;
    input->dim_count = 4;
    input->name = "input";
    int in_size = csinn_tensor_size(input);

    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = input->dim[0];
    output->dim[1] = out_c;
    output->dim[2] = out_h;
    output->dim[3] = out_w;
    output->dim_count = 4;
    output->name = "output";
    int out_size = csinn_tensor_size(output);

    struct csinn_pool_params *params = csinn_alloc_params(sizeof(struct csinn_pool_params), NULL);
    params->base.name = "params";
    params->ceil_mode = 0;
    params->stride_height = stride_h;
    params->stride_width = stride_w;
    params->filter_height = kernel_h;
    params->filter_width = kernel_w;
    params->pad_left = pad_w;
    params->pad_right = pad_w;
    params->pad_top = pad_h;
    params->pad_down = pad_h;

    input->data = input_data;
    output->data = shl_mem_alloc(out_size * sizeof(float));

    func(input, output, params);

    evaluate_error(output->data, ref_data, out_size, dtype);

    csinn_free_tensor(input);
    shl_mem_free(output->data);
    csinn_free_tensor(output);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of maxpool for RVV.\n");
    verify_maxpool2d(maxpool2x2s2_fp32_in, maxpool2x2s2_fp32_out, shl_rvv_maxpool2x2s2_fp32, 2, 6,
                     18, 2, 3, 9, 2, 2, 2, 2, 0, 0, CSINN_DTYPE_FLOAT32);
    verify_maxpool2d(maxpool2x2s2_fp16_in, maxpool2x2s2_fp16_out, shl_rvv_maxpool2x2s2_fp16, 2, 6,
                     18, 2, 3, 9, 2, 2, 2, 2, 0, 0, CSINN_DTYPE_FLOAT16);
    verify_maxpool2d(maxpool2x2s2_int8_in, maxpool2x2s2_int8_out, shl_rvv_maxpool2x2s2_int8, 2, 6,
                     18, 2, 3, 9, 2, 2, 2, 2, 0, 0, CSINN_DTYPE_INT8);

    verify_maxpool2d(maxpool2x2s2_p1_fp32_in, maxpool2x2s2_p1_fp32_out,
                     shl_rvv_maxpool2x2s2_p1_fp32, 2, 7, 19, 2, 4, 10, 2, 2, 2, 2, 1, 1,
                     CSINN_DTYPE_FLOAT32);
    verify_maxpool2d(maxpool2x2s2_p1_fp16_in, maxpool2x2s2_p1_fp16_out,
                     shl_rvv_maxpool2x2s2_p1_fp16, 2, 7, 19, 2, 4, 10, 2, 2, 2, 2, 1, 1,
                     CSINN_DTYPE_FLOAT16);
    verify_maxpool2d(maxpool2x2s2_p1_int8_in, maxpool2x2s2_p1_int8_out,
                     shl_rvv_maxpool2x2s2_p1_int8, 2, 7, 19, 2, 4, 10, 2, 2, 2, 2, 1, 1,
                     CSINN_DTYPE_INT8);

    verify_maxpool2d(maxpool3x3s2_fp32_in, maxpool3x3s2_fp32_out, shl_rvv_maxpool3x3s2_fp32, 2, 7,
                     19, 2, 3, 9, 3, 3, 2, 2, 0, 0, CSINN_DTYPE_FLOAT32);
    verify_maxpool2d(maxpool3x3s2_fp16_in, maxpool3x3s2_fp16_out, shl_rvv_maxpool3x3s2_fp16, 2, 7,
                     19, 2, 3, 9, 3, 3, 2, 2, 0, 0, CSINN_DTYPE_FLOAT16);
    verify_maxpool2d(maxpool3x3s2_int8_in, maxpool3x3s2_int8_out, shl_rvv_maxpool3x3s2_int8, 2, 7,
                     19, 2, 3, 9, 3, 3, 2, 2, 0, 0, CSINN_DTYPE_INT8);

    verify_maxpool2d(maxpool3x3s2_p1_fp32_in, maxpool3x3s2_p1_fp32_out,
                     shl_rvv_maxpool3x3s2_p1_fp32, 2, 6, 18, 2, 3, 9, 3, 3, 2, 2, 1, 1,
                     CSINN_DTYPE_FLOAT32);
    verify_maxpool2d(maxpool3x3s2_p1_fp16_in, maxpool3x3s2_p1_fp16_out,
                     shl_rvv_maxpool3x3s2_p1_fp16, 2, 6, 18, 2, 3, 9, 3, 3, 2, 2, 1, 1,
                     CSINN_DTYPE_FLOAT16);
    verify_maxpool2d(maxpool3x3s2_p1_int8_in, maxpool3x3s2_p1_int8_out,
                     shl_rvv_maxpool3x3s2_p1_int8, 2, 6, 18, 2, 3, 9, 3, 3, 2, 2, 1, 1,
                     CSINN_DTYPE_INT8);

    verify_maxpool2d(maxpool3x3s1_p1_fp32_in, maxpool3x3s1_p1_fp32_out,
                     shl_rvv_maxpool3x3s1_p1_fp32, 2, 3, 10, 2, 3, 10, 3, 3, 1, 1, 1, 1,
                     CSINN_DTYPE_FLOAT32);
    verify_maxpool2d(maxpool3x3s1_p1_fp16_in, maxpool3x3s1_p1_fp16_out,
                     shl_rvv_maxpool3x3s1_p1_fp16, 2, 3, 10, 2, 3, 10, 3, 3, 1, 1, 1, 1,
                     CSINN_DTYPE_FLOAT16);
    verify_maxpool2d(maxpool3x3s1_p1_int8_in, maxpool3x3s1_p1_int8_out,
                     shl_rvv_maxpool3x3s1_p1_int8, 2, 3, 10, 2, 3, 10, 3, 3, 1, 1, 1, 1,
                     CSINN_DTYPE_INT8);

    verify_maxpool2d(global_maxpool_fp32_in, global_maxpool_fp32_out, shl_rvv_global_maxpool2d_fp32,
                     3, 7, 7, 3, 1, 1, 7, 7, 1, 1, 0, 0, CSINN_DTYPE_FLOAT32);
    verify_maxpool2d(global_maxpool_fp16_in, global_maxpool_fp16_out, shl_rvv_global_maxpool2d_fp16,
                     3, 7, 7, 3, 1, 1, 7, 7, 1, 1, 0, 0, CSINN_DTYPE_FLOAT16);

    return done_testing();
}