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

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"
#include "./valid_data/q7_conv_RGB.dat"

extern void verify_conv2d_q7(void *input_data,
                             void *kernel_data,
                             void *bias_data,
                             void *ref_data,
                             uint16_t batch,
                             uint16_t in_h,
                             uint16_t in_w,
                             uint16_t in_c,
                             uint16_t out_h,
                             uint16_t out_w,
                             uint16_t out_c,
                             uint16_t kernel_h,
                             uint16_t kernel_w,
                             uint16_t stride_h,
                             uint16_t stride_w,
                             uint16_t pad_x,
                             uint16_t pad_y,
                             uint16_t bias_shift,
                             uint16_t out_shift,
                             float difference);


int main(int argc, char** argv)
{
    init_testsuite("Testing function of convolution RGB q7 for xt800.\n");

    verify_conv2d_q7(q7_conv_input_0, q7_conv_weight_0, q7_conv_bias_0, q7_conv_result_0,
                     1, 32, 32, 3, 30, 30, 16, 3, 3, 1, 1, 0, 0, 0, 11, 0.0f);

    verify_conv2d_q7(q7_conv_input_0, q7_conv_weight_0, q7_conv_bias_0, q7_conv_result_1,
                     1, 32, 32, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_2,
                     1, 32, 32, 3, 28, 28, 16, 5, 5, 1, 1, 0, 0, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_3,
                     1, 32, 32, 3, 32, 32, 16, 5, 5, 1, 1, 2, 2, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_4,
                     1, 32, 32, 3, 12, 12, 16, 5, 5, 3, 3, 3, 3, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_2, q7_conv_weight_2, q7_conv_bias_2, q7_conv_result_5,
                     1, 32, 32, 3, 26, 26, 16, 7, 7, 1, 1, 0, 0, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_2, q7_conv_weight_2, q7_conv_bias_2, q7_conv_result_6,
                     1, 32, 32, 3, 32, 32, 16, 7, 7, 1, 1, 3, 3, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_2, q7_conv_weight_2, q7_conv_bias_2, q7_conv_result_7,
                     1, 32, 32, 3, 10, 10, 16, 7, 7, 3, 3, 1, 1, 0, 12, 0.0f);

    /* leftover test */
    verify_conv2d_q7(q7_conv_input_0, q7_conv_weight_0, q7_conv_bias_0, q7_conv_result_8,
                     1, 31, 31, 3, 29, 29, 15, 3, 3, 1, 1, 0, 0, 0, 11, 0.0f);

    verify_conv2d_q7(q7_conv_input_0, q7_conv_weight_0, q7_conv_bias_0, q7_conv_result_9,
                     1, 31, 31, 3, 31, 31, 15, 3, 3, 1, 1, 1, 1, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_10,
                     1, 31, 31, 3, 27, 27, 15, 5, 5, 1, 1, 0, 0, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_11,
                     1, 31, 31, 3, 31, 31, 15, 5, 5, 1, 1, 2, 2, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_1, q7_conv_weight_1, q7_conv_bias_1, q7_conv_result_12,
                     1, 31, 31, 3, 1, 1, 15, 5, 5, 3, 3, 2, 2, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_2, q7_conv_weight_2, q7_conv_bias_2, q7_conv_result_13,
                     1, 31, 31, 3, 25, 25, 15, 7, 7, 1, 1, 0, 0, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_2, q7_conv_weight_2, q7_conv_bias_2, q7_conv_result_14,
                     1, 31, 31, 3, 31, 31, 15, 7, 7, 1, 1, 3, 3, 0, 12, 0.0f);

    verify_conv2d_q7(q7_conv_input_2, q7_conv_weight_2, q7_conv_bias_2, q7_conv_result_15,
                     1, 31, 31, 3, 9, 9, 15, 7, 7, 3, 3, 0, 0, 0, 12, 0.0f);
}
