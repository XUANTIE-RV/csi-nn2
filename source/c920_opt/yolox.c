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

#include "c920/c920.h"

int shl_c920_yolox_preprocess(struct csinn_tensor *input, struct csinn_tensor *output)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;

    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int out_c = output->dim[1];
    int out_h = output->dim[2];
    int out_w = output->dim[3];

    int stride = 2 * sizeof(uint8_t);

    uint8_t *out_ptr = output_data;
    for (int c = 0; c < in_c; c++) {
        for (int h = 0; h < in_h; h += 2) {
            uint8_t *in_ptr = input_data + (c * in_h + h) * in_w;
            int w = 0;
            while (w < out_w) {
                int vl = vsetvl_e8m8(out_w - w);
                vuint8m8_t _u8 = vlse8_v_u8m8(in_ptr, stride, vl);
                vse8_v_u8m8(out_ptr, _u8, vl);
                w += vl;
                in_ptr += vl * 2;
                out_ptr += vl;
            }
        }
    }

    for (int c = 0; c < in_c; c++) {
        for (int h = 1; h < in_h; h += 2) {
            uint8_t *in_ptr = input_data + (c * in_h + h) * in_w;
            int w = 0;
            while (w < out_w) {
                int vl = vsetvl_e8m8(out_w - w);
                vuint8m8_t _u8 = vlse8_v_u8m8(in_ptr, stride, vl);
                vse8_v_u8m8(out_ptr, _u8, vl);
                w += vl;
                in_ptr += vl * 2;
                out_ptr += vl;
            }
        }
    }

    for (int c = 0; c < in_c; c++) {
        for (int h = 0; h < in_h; h += 2) {
            uint8_t *in_ptr = input_data + (c * in_h + h) * in_w + 1;
            int w = 0;
            while (w < out_w) {
                int vl = vsetvl_e8m8(out_w - w);
                vuint8m8_t _u8 = vlse8_v_u8m8(in_ptr, stride, vl);
                vse8_v_u8m8(out_ptr, _u8, vl);
                w += vl;
                in_ptr += vl * 2;
                out_ptr += vl;
            }
        }
    }

    for (int c = 0; c < in_c; c++) {
        for (int h = 1; h < in_h; h += 2) {
            uint8_t *in_ptr = input_data + (c * in_h + h) * in_w + 1;
            int w = 0;
            while (w < out_w) {
                int vl = vsetvl_e8m8(out_w - w);
                vuint8m8_t _u8 = vlse8_v_u8m8(in_ptr, stride, vl);
                vse8_v_u8m8(out_ptr, _u8, vl);
                w += vl;
                in_ptr += vl * 2;
                out_ptr += vl;
            }
        }
    }

    return CSINN_TRUE;
}
