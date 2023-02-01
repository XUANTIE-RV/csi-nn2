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

/* SHL version 2.1.x */

#include "shl_thead_rvv.h"

/*************************************************************
 * note: support flexible vlen
 *************************************************************/
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static void maxpool_w8_fp16_nhwc(const __fp16 *src, __fp16 *dst, struct csinn_pool_params *params,
                                 int oh, int ow, int idx_h_start, int idx_h_end, int in_w,
                                 int out_w, int in_c)
{
    int kernel_w = params->filter_width;
    int stride_w = params->stride_width;
    int pad_left = params->pad_left;

    const int idx_w_start = -pad_left + ow * stride_w;
    const int idx_w_end = idx_w_start + kernel_w;

    vfloat16m1_t _max0, _max1, _max2, _max3;
    vfloat16m1_t _max4, _max5, _max6, _max7;

    int vl;
    int c = 0;
    while (c < in_c) {
        vl = vsetvl_e16m1(in_c - c);
        _max0 = vfmv_v_f_f16m1(-__FLT_MAX__, vl);
        _max1 = _max2 = _max3 = _max4 = _max0;
        _max5 = _max6 = _max7 = _max0;

        for (int h = idx_h_start; h < idx_h_end; h++) {
            for (int w = idx_w_start; w < idx_w_end; w++) {
                const __fp16 *in_ptr = src + (h * in_w + w) * in_c + c;
                _max0 = vfmax_vv_f16m1(_max0, vle16_v_f16m1(in_ptr + 0 * stride_w * in_c, vl), vl);
                _max1 = vfmax_vv_f16m1(_max1, vle16_v_f16m1(in_ptr + 1 * stride_w * in_c, vl), vl);
                _max2 = vfmax_vv_f16m1(_max2, vle16_v_f16m1(in_ptr + 2 * stride_w * in_c, vl), vl);
                _max3 = vfmax_vv_f16m1(_max3, vle16_v_f16m1(in_ptr + 3 * stride_w * in_c, vl), vl);
                _max4 = vfmax_vv_f16m1(_max4, vle16_v_f16m1(in_ptr + 4 * stride_w * in_c, vl), vl);
                _max5 = vfmax_vv_f16m1(_max5, vle16_v_f16m1(in_ptr + 5 * stride_w * in_c, vl), vl);
                _max6 = vfmax_vv_f16m1(_max6, vle16_v_f16m1(in_ptr + 6 * stride_w * in_c, vl), vl);
                _max7 = vfmax_vv_f16m1(_max7, vle16_v_f16m1(in_ptr + 7 * stride_w * in_c, vl), vl);
            }
        }
        __fp16 *out_ptr = dst + (oh * out_w + ow) * in_c + c;
        vse16_v_f16m1(out_ptr + 0 * in_c, _max0, vl);
        vse16_v_f16m1(out_ptr + 1 * in_c, _max1, vl);
        vse16_v_f16m1(out_ptr + 2 * in_c, _max2, vl);
        vse16_v_f16m1(out_ptr + 3 * in_c, _max3, vl);
        vse16_v_f16m1(out_ptr + 4 * in_c, _max4, vl);
        vse16_v_f16m1(out_ptr + 5 * in_c, _max5, vl);
        vse16_v_f16m1(out_ptr + 6 * in_c, _max6, vl);
        vse16_v_f16m1(out_ptr + 7 * in_c, _max7, vl);
        c += vl;
    }
}

static void maxpool_border_fp16_nhwc(const __fp16 *src, __fp16 *dst,
                                     struct csinn_pool_params *params, int oh, int ow,
                                     int idx_h_start, int idx_h_end, int in_w, int out_w, int in_c)
{
    int kernel_w = params->filter_width;
    int stride_w = params->stride_width;
    int pad_left = params->pad_left;

    int i_w_start = -pad_left + ow * stride_w;
    int i_w_end = i_w_start + kernel_w;
    const int idx_w_start = max(i_w_start, 0);
    const int idx_w_end = min(i_w_end, in_w);

    int vl;
    int c = 0;
    while (c < in_c) {
        vl = vsetvl_e16m1(in_c - c);
        vfloat16m1_t _max = vfmv_v_f_f16m1(-__FLT_MAX__, vl);
        for (int h = idx_h_start; h < idx_h_end; h++) {
            for (int w = idx_w_start; w < idx_w_end; w++) {
                const __fp16 *in_ptr = src + (h * in_w + w) * in_c + c;
                _max = vfmax_vv_f16m1(_max, vle16_v_f16m1(in_ptr, vl), vl);
            }
        }
        __fp16 *out_ptr = dst + (oh * out_w + ow) * in_c + c;
        vse16_v_f16m1(out_ptr, _max, vl);
        c += vl;
    }
}

int shl_rvv_maxpool_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int batch = input->dim[0];
    int in_h = input->dim[1];
    int in_w = input->dim[2];
    int in_c = input->dim[3];

    int out_h = output->dim[1];
    int out_w = output->dim[2];
    int out_c = output->dim[3];

    int kernel_h = params->filter_height;
    int kernel_w = params->filter_width;
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int pad_left = params->pad_left;
    int pad_top = params->pad_top;

    int dst_1x8_start_w = max((pad_left + stride_w - 1) / stride_w, 0);
    int dst_1x8_end_w = min((in_w + pad_left - kernel_w) / stride_w + 1, out_w);

    for (int b = 0; b < batch; b++) {
        __fp16 *in_ptr = input_data + b * in_h * in_w * in_c;
        __fp16 *out_ptr = output_data + b * out_h * out_w * out_c;

        for (int oh = 0; oh < out_h; oh++) {
            int i_h_start = -pad_top + oh * stride_h;
            int i_h_end = i_h_start + kernel_h;
            const int idx_h_start = max(i_h_start, 0);
            const int idx_h_end = min(i_h_end, in_h);
            int ow = 0;
            for (; ow < dst_1x8_start_w; ow++) {
                maxpool_border_fp16_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                         in_w, out_w, in_c);
            }
            for (; ow + 8 <= dst_1x8_end_w; ow += 8) {
                maxpool_w8_fp16_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end, in_w,
                                     out_w, in_c);
            }
            for (; ow < out_w; ow++) {
                maxpool_border_fp16_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                         in_w, out_w, in_c);
            }
        }
    }
    return CSINN_TRUE;
}
