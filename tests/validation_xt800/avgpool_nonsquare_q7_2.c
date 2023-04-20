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

#include "./valid_data/pool_data.dat"
#include "csi_nn.h"
#include "test_utils.h"

extern void verify_avgpool2d_q7(void *input_data, void *output_data, uint16_t batch, uint16_t in_h,
                                uint16_t in_w, uint16_t in_c, uint16_t out_h, uint16_t out_w,
                                uint16_t out_c, uint16_t kernel_h, uint16_t kernel_w,
                                uint16_t stride_h, uint16_t stride_w, uint16_t pad_x,
                                uint16_t pad_y, uint16_t out_lshift, float difference);

int main(int argc, char **argv)
{
    init_testsuite("Second testing function of avgpool nonsquare q7 for xt800.\n");

    /* ---------------- leftover ------------------------*/
    verify_avgpool2d_q7(pooling_input_00, avepool_nonsquare_result_9, 1, 63, 15, 4, 61, 13, 4, 3, 3,
                        1, 1, 0, 0, 1, 3.0f);

    verify_avgpool2d_q7(pooling_input_01, avepool_nonsquare_result_10, 1, 63, 15, 4, 29, 6, 4, 7, 5,
                        2, 2, 0, 0, 1, 3.0f);

    verify_avgpool2d_q7(pooling_input_02, avepool_nonsquare_result_11, 1, 31, 31, 4, 8, 6, 4, 5, 7,
                        4, 5, 1, 2, 1, 3.0f);

    verify_avgpool2d_q7(pooling_input_10, avepool_nonsquare_result_12, 1, 31, 127, 1, 29, 125, 1, 3,
                        3, 1, 1, 0, 0, 0, 3.0f);

    verify_avgpool2d_q7(pooling_input_11, avepool_nonsquare_result_13, 1, 127, 31, 1, 26, 13, 1, 5,
                        7, 5, 2, 0, 3, 0, 3.0f);

    verify_avgpool2d_q7(pooling_input_12, avepool_nonsquare_result_14, 1, 63, 63, 1, 29, 30, 1, 8,
                        6, 2, 2, 1, 1, 0, 3.0f);

    verify_avgpool2d_q7(pooling_input_20, avepool_nonsquare_result_15, 1, 31, 7, 16, 29, 5, 16, 5,
                        3, 1, 1, 0, 2, 2, 3.0f);

    verify_avgpool2d_q7(pooling_input_21, avepool_nonsquare_result_16, 1, 7, 31, 16, 7, 14, 16, 3,
                        5, 1, 2, 0, 2, 2, 3.0f);

    verify_avgpool2d_q7(pooling_input_22, avepool_nonsquare_result_17, 1, 15, 15, 16, 7, 5, 16, 3,
                        5, 2, 3, 2, 0, 2, 3.0f);
}
