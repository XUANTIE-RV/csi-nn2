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

static void maxpool_w8_fp32_packn(const float *src, float *dst, struct csinn_pool_params *params,
                                  int oh, int ow, int idx_h_start, int idx_h_end, int in_w,
                                  int out_w)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    int kernel_w = params->filter_width;
    int stride_w = params->stride_width;
    int pad_left = params->pad_left;

    const int idx_w_start = -pad_left + ow * stride_w;
    const int idx_w_end = idx_w_start + kernel_w;

    vfloat32m1_t _max0, _max1, _max2, _max3;
    vfloat32m1_t _max4, _max5, _max6, _max7;

    _max0 = vfmv_v_f_f32m1(-__FLT_MAX__, vl);
    _max1 = _max2 = _max3 = _max4 = _max0;
    _max5 = _max6 = _max7 = _max0;

    for (int h = idx_h_start; h < idx_h_end; h++) {
        for (int w = idx_w_start; w < idx_w_end; w++) {
            const float *in_ptr = src + (h * in_w + w) * packn;
            _max0 = vfmax_vv_f32m1(_max0, vle32_v_f32m1(in_ptr + 0 * stride_w * packn, vl), vl);
            _max1 = vfmax_vv_f32m1(_max1, vle32_v_f32m1(in_ptr + 1 * stride_w * packn, vl), vl);
            _max2 = vfmax_vv_f32m1(_max2, vle32_v_f32m1(in_ptr + 2 * stride_w * packn, vl), vl);
            _max3 = vfmax_vv_f32m1(_max3, vle32_v_f32m1(in_ptr + 3 * stride_w * packn, vl), vl);
            _max4 = vfmax_vv_f32m1(_max4, vle32_v_f32m1(in_ptr + 4 * stride_w * packn, vl), vl);
            _max5 = vfmax_vv_f32m1(_max5, vle32_v_f32m1(in_ptr + 5 * stride_w * packn, vl), vl);
            _max6 = vfmax_vv_f32m1(_max6, vle32_v_f32m1(in_ptr + 6 * stride_w * packn, vl), vl);
            _max7 = vfmax_vv_f32m1(_max7, vle32_v_f32m1(in_ptr + 7 * stride_w * packn, vl), vl);
        }
    }
    float *out_ptr = dst + (oh * out_w + ow) * packn;
    vse32_v_f32m1(out_ptr + 0 * packn, _max0, vl);
    vse32_v_f32m1(out_ptr + 1 * packn, _max1, vl);
    vse32_v_f32m1(out_ptr + 2 * packn, _max2, vl);
    vse32_v_f32m1(out_ptr + 3 * packn, _max3, vl);
    vse32_v_f32m1(out_ptr + 4 * packn, _max4, vl);
    vse32_v_f32m1(out_ptr + 5 * packn, _max5, vl);
    vse32_v_f32m1(out_ptr + 6 * packn, _max6, vl);
    vse32_v_f32m1(out_ptr + 7 * packn, _max7, vl);
}

static void maxpool_border_fp32_packn(const float *src, float *dst,
                                      struct csinn_pool_params *params, int oh, int ow,
                                      int idx_h_start, int idx_h_end, int in_w, int out_w)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    int kernel_w = params->filter_width;
    int stride_w = params->stride_width;
    int pad_left = params->pad_left;

    int i_w_start = -pad_left + ow * stride_w;
    int i_w_end = i_w_start + kernel_w;
    const int idx_w_start = max(i_w_start, 0);
    const int idx_w_end = min(i_w_end, in_w);

    vfloat32m1_t _max = vfmv_v_f_f32m1(-__FLT_MAX__, vl);
    for (int h = idx_h_start; h < idx_h_end; h++) {
        for (int w = idx_w_start; w < idx_w_end; w++) {
            const float *in_ptr = src + (h * in_w + w) * packn;
            _max = vfmax_vv_f32m1(_max, vle32_v_f32m1(in_ptr, vl), vl);
        }
    }
    float *out_ptr = dst + (oh * out_w + ow) * packn;
    vse32_v_f32m1(out_ptr, _max, vl);
}

int shl_rvv_maxpool_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];

    int out_h = output->dim[2];
    int out_w = output->dim[3];

    int kernel_h = params->filter_height;
    int kernel_w = params->filter_width;
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int pad_left = params->pad_left;
    int pad_top = params->pad_top;

    int dst_1x8_start_w = max((pad_left + stride_w - 1) / stride_w, 0);
    int dst_1x8_end_w = min((in_w + pad_left - kernel_w) / stride_w + 1, out_w);

    for (int bc = 0; bc < batch * in_c; bc += packn) {
        float *in_ptr = input_data + bc * in_h * in_w;
        float *out_ptr = output_data + bc * out_h * out_w;

        for (int oh = 0; oh < out_h; oh++) {
            int i_h_start = -pad_top + oh * stride_h;
            int i_h_end = i_h_start + kernel_h;
            const int idx_h_start = max(i_h_start, 0);
            const int idx_h_end = min(i_h_end, in_h);
            int ow = 0;
            for (; ow < dst_1x8_start_w; ow++) {
                maxpool_border_fp32_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                          in_w, out_w);
            }
            for (; ow + 8 <= dst_1x8_end_w; ow += 8) {
                maxpool_w8_fp32_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end, in_w,
                                      out_w);
            }
            for (; ow < out_w; ow++) {
                maxpool_border_fp32_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                          in_w, out_w);
            }
        }
    }
    return CSINN_TRUE;
}
