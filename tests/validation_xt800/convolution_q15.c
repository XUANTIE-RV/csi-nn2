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

/* CSI-NN2 version 1.12.x */

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"
#include "./valid_data/q15_conv_basic.dat"


extern void verify_conv2d_q15(void *input_data,
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
    init_testsuite("Testing function of convolution q15 for xt800.\n");

    verify_conv2d_q15(q15_conv_input_3, q15_conv_weight_3, q15_conv_bias_3, q15_conv_result_16,
                      1, 16, 16, 8, 14, 14, 8, 3, 3, 1, 1, 0, 0, 0, 11, 0.0f);

    verify_conv2d_q15(q15_conv_input_3, q15_conv_weight_3, q15_conv_bias_3, q15_conv_result_17,
                      1, 16, 16, 8, 16, 16, 8, 3, 3, 1, 1, 1, 1, 0, 12, 0.0f);

    verify_conv2d_q15(q15_conv_input_4, q15_conv_weight_4, q15_conv_bias_4, q15_conv_result_18,
                      1, 16, 16, 8, 12, 12, 16, 5, 5, 1, 1, 0, 0, 0, 12, 0.0f);

    verify_conv2d_q15(q15_conv_input_4, q15_conv_weight_4, q15_conv_bias_4, q15_conv_result_19,
                      1, 16, 16, 8, 16, 16, 16, 5, 5, 1, 1, 2, 2, 0, 12, 0.0f);

    verify_conv2d_q15(q15_conv_input_4, q15_conv_weight_4, q15_conv_bias_4, q15_conv_result_20,
                      1, 16, 16, 8, 6, 6, 16, 5, 5, 3, 3, 2, 2, 0, 12, 0.0f);

    verify_conv2d_q15(q15_conv_input_5, q15_conv_weight_5, q15_conv_bias_5, q15_conv_result_21,
                      1, 16, 16, 8, 10, 10, 24, 7, 7, 1, 1, 0, 0, 0, 12, 0.0f);

    verify_conv2d_q15(q15_conv_input_5, q15_conv_weight_5, q15_conv_bias_5, q15_conv_result_22,
                      1, 16, 16, 8, 16, 16, 24, 7, 7, 1, 1, 3, 3, 0, 12, 0.0f);

    verify_conv2d_q15(q15_conv_input_5, q15_conv_weight_5, q15_conv_bias_5, q15_conv_result_23,
                      1, 16, 16, 8, 6, 6, 24, 7, 7, 3, 3, 3, 3, 0, 12, 0.0f);



    // FIXME: ld: region `DATA' overflowed by 41200 bytes
    // verify_conv2d_q15(q15_conv_input_0, q15_conv_weight_0, q15_conv_bias_0, q15_conv_result_0,
    //                   1, 32, 32, 16, 30, 30, 32, 3, 3, 1, 1, 0, 0, 0, 11, 0.0f);

    // verify_conv2d_q15(q15_conv_input_0, q15_conv_weight_0, q15_conv_bias_0, q15_conv_result_1,
    //                   1, 32, 32, 16, 32, 32, 32, 3, 3, 1, 1, 1, 1, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_1, q15_conv_weight_1, q15_conv_bias_1, q15_conv_result_2,
    //                   1, 32, 32, 16, 28, 28, 16, 5, 5, 1, 1, 0, 0, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_1, q15_conv_weight_1, q15_conv_bias_1, q15_conv_result_3,
    //                   1, 32, 32, 16, 32, 32, 16, 5, 5, 1, 1, 2, 2, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_1, q15_conv_weight_1, q15_conv_bias_1, q15_conv_result_4,
    //                   1, 32, 32, 16, 12, 12, 16, 5, 5, 3, 3, 3, 3, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_2, q15_conv_weight_2, q15_conv_bias_2, q15_conv_result_5,
    //                   1, 32, 32, 16, 26, 26, 16, 7, 7, 1, 1, 0, 0, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_2, q15_conv_weight_2, q15_conv_bias_2, q15_conv_result_6,
    //                   1, 32, 32, 16, 32, 32, 16, 7, 7, 1, 1, 3, 3, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_2, q15_conv_weight_2, q15_conv_bias_2, q15_conv_result_7,
    //                   1, 32, 32, 16, 10, 10, 16, 7, 7, 3, 3, 1, 1, 0, 12, 0.0f);

    /* ------------- leftover ------------------*/
    // verify_conv2d_q15(q15_conv_input_0, q15_conv_weight_0, q15_conv_bias_0, q15_conv_result_8,
    //                   1, 31, 31, 15, 29, 29, 30, 3, 3, 1, 1, 0, 0, 0, 11, 0.0f);

    // verify_conv2d_q15(q15_conv_input_0, q15_conv_weight_0, q15_conv_bias_0, q15_conv_result_9,
    //                   1, 31, 31, 15, 31, 31, 30, 3, 3, 1, 1, 1, 1, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_1, q15_conv_weight_1, q15_conv_bias_1, q15_conv_result_10,
    //                   1, 31, 31, 15, 27, 27, 15, 5, 5, 1, 1, 0, 0, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_1, q15_conv_weight_1, q15_conv_bias_1, q15_conv_result_11,
    //                   1, 31, 31, 15, 31, 31, 15, 5, 5, 1, 1, 2, 2, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_1, q15_conv_weight_1, q15_conv_bias_1, q15_conv_result_12,
    //                   1, 31, 31, 15, 11, 11, 15, 5, 5, 3, 3, 2, 2, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_2, q15_conv_weight_2, q15_conv_bias_2, q15_conv_result_13,
    //                   1, 31, 31, 15, 25, 25, 15, 7, 7, 1, 1, 0, 0, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_2, q15_conv_weight_2, q15_conv_bias_2, q15_conv_result_14,
    //                   1, 31, 31, 15, 31, 31, 15, 7, 7, 1, 1, 3, 3, 0, 12, 0.0f);

    // verify_conv2d_q15(q15_conv_input_2, q15_conv_weight_2, q15_conv_bias_2, q15_conv_result_15,
    //                   1, 31, 31, 15, 9, 9, 15, 7, 7, 3, 3, 0, 0, 0, 12, 0.0f);
}

