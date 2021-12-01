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
#include "./valid_data/q7_1x1_conv.dat"


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
    init_testsuite("First testing function of convolution 1x1 q7 for xt800.\n");


    /* -------------- conv2d 1x1 --------------- */
    verify_conv2d_q7(q7_1x1_conv_input_0, q7_1x1_conv_weight_0, q7_1x1_conv_bias_0, q7_1x1_conv_result_0,
                     1, 32, 32, 16, 32, 32, 32, 1, 1, 1, 1, 0, 0, 0, 12, 0.0f);

    /* leftover test */
    verify_conv2d_q7(q7_1x1_conv_input_0, q7_1x1_conv_weight_0, q7_1x1_conv_bias_0, q7_1x1_conv_result_3,
                     1, 31, 31, 12, 31, 31, 30, 1, 1, 1, 1, 0, 0, 0, 12, 0.0f);

    verify_conv2d_q7(q7_1x1_conv_input_1, q7_1x1_conv_weight_1, q7_1x1_conv_bias_1, q7_1x1_conv_result_1,
                     1, 64, 16, 16, 64, 16, 16, 1, 1, 1, 1, 0, 0, 0, 12, 0.0f);

    /* leftover test */
    verify_conv2d_q7(q7_1x1_conv_input_1, q7_1x1_conv_weight_1, q7_1x1_conv_bias_1, q7_1x1_conv_result_4,
                     1, 63, 15, 12, 63, 15, 12, 1, 1, 1, 1, 0, 0, 0, 12, 0.0f);


    // TODO: ld: region `DATA' overflowed by 41200 bytes
    // verify_conv2d_q7(q7_1x1_conv_input_2, q7_1x1_conv_weight_2, q7_1x1_conv_bias_2, q7_1x1_conv_result_2,
    //                  1, 16, 64, 16, 16, 64, 48, 1, 1, 1, 1, 0, 0, 0, 12, 0.0f);

    // // /* leftover test */
    // verify_conv2d_q7(q7_1x1_conv_input_2, q7_1x1_conv_weight_2, q7_1x1_conv_bias_2, q7_1x1_conv_result_5,
    //                  1, 15, 63, 12, 15, 63, 40, 1, 1, 1, 1, 0, 0, 0, 12, 0.0f);
}








