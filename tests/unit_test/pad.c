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

#include "./valid_data/pad.dat"

#include "csi_nn.h"
#include "shl_thead_rvv.h"
#include "test_utils.h"

void verify_pad(void *input_data, void *ref_data, void (*func)(), int in_c, int in_h, int in_w,
                int pad_top, int pad_left, int pad_down, int pad_right, enum csinn_dtype_enum dtype)
{
    int padded_h = in_h + pad_top + pad_down;
    int padded_w = in_w + pad_left + pad_right;
    int out_size = in_c * padded_h * padded_w;

    float *out = shl_mem_alloc(out_size * sizeof(float));

    if (dtype == CSINN_DTYPE_INT8) {
        func(input_data, out, in_c, in_h, in_w, padded_h, padded_w, pad_top, pad_left, (int8_t)0);
    } else {
        func(input_data, out, in_c, in_h, in_w, padded_h, padded_w, pad_top, pad_left);
    }

    evaluate_error(out, ref_data, out_size, dtype);

    shl_mem_free(out);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of pad for RVV.\n");
    verify_pad(pad_fp32_in, pad_fp32_out, shl_rvv_pad_input_fp32, 3, 4, 19, 1, 1, 1, 1,
               CSINN_DTYPE_FLOAT32);
    verify_pad(pad_fp16_in, pad_fp16_out, shl_rvv_pad_input_fp16, 3, 4, 19, 1, 1, 1, 1,
               CSINN_DTYPE_FLOAT16);
    verify_pad(pad_int8_in, pad_int8_out, shl_rvv_pad_input_int8, 3, 4, 19, 1, 1, 1, 1,
               CSINN_DTYPE_INT8);

    return done_testing();
}
