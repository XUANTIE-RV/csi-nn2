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

#include "shl_thead_rvv.h"

/*************************************************************
 * s2 * (q2 - z2) = avgpool_mxn{ s1 * (q1 - z1) }
 * q2 = s1/s2 * (∑(q1 - z1))/(m*n) + z2
 * constrain: input channel % packn = 0
 *************************************************************/
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static void avgpool_w8_int8_packn(const int8_t *src, int8_t *dst, struct csinn_pool_params *params,
                                  int oh, int ow, int idx_h_start, int idx_h_end, int in_w,
                                  int out_w, enum avgpool_loc_enum loc, __fp16 real_scale, int z1,
                                  int z2)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int kernel_w = params->filter_width;
    int stride_w = params->stride_width;
    const int idx_w_start = -params->pad_left + ow * stride_w;
    const int idx_w_end = idx_w_start + kernel_w;

    int window_size = shl_rvv_avgpool_get_window_size(params, idx_h_start, idx_h_end, idx_w_start,
                                                      idx_w_end, loc);
    __fp16 ratio = 1.0f / window_size;
    ratio *= real_scale;
    int z1xn = z1 * window_size;

    vint16m2_t _acc0, _acc1, _acc2, _acc3;
    vint16m2_t _acc4, _acc5, _acc6, _acc7;
    _acc0 = vmv_v_x_i16m2(0, vl);
    _acc1 = _acc2 = _acc3 = _acc0;
    _acc4 = _acc5 = _acc6 = _acc7 = _acc0;

    for (int h = idx_h_start; h < idx_h_end; h++) {
        for (int w = idx_w_start; w < idx_w_end; w++) {
            const int8_t *in_ptr = src + (h * in_w + w) * packn;
            _acc0 = vadd_vv_i16m2(
                _acc0, vwadd_vx_i16m2(vle8_v_i8m1(in_ptr + 0 * stride_w * packn, vl), 0, vl), vl);
            _acc1 = vadd_vv_i16m2(
                _acc1, vwadd_vx_i16m2(vle8_v_i8m1(in_ptr + 1 * stride_w * packn, vl), 0, vl), vl);
            _acc2 = vadd_vv_i16m2(
                _acc2, vwadd_vx_i16m2(vle8_v_i8m1(in_ptr + 2 * stride_w * packn, vl), 0, vl), vl);
            _acc3 = vadd_vv_i16m2(
                _acc3, vwadd_vx_i16m2(vle8_v_i8m1(in_ptr + 3 * stride_w * packn, vl), 0, vl), vl);
            _acc4 = vadd_vv_i16m2(
                _acc4, vwadd_vx_i16m2(vle8_v_i8m1(in_ptr + 4 * stride_w * packn, vl), 0, vl), vl);
            _acc5 = vadd_vv_i16m2(
                _acc5, vwadd_vx_i16m2(vle8_v_i8m1(in_ptr + 5 * stride_w * packn, vl), 0, vl), vl);
            _acc6 = vadd_vv_i16m2(
                _acc6, vwadd_vx_i16m2(vle8_v_i8m1(in_ptr + 6 * stride_w * packn, vl), 0, vl), vl);
            _acc7 = vadd_vv_i16m2(
                _acc7, vwadd_vx_i16m2(vle8_v_i8m1(in_ptr + 7 * stride_w * packn, vl), 0, vl), vl);
        }
    }

    vfloat16m2_t _tmp0 = vfcvt_f_x_v_f16m2(_acc0, vl);
    vfloat16m2_t _tmp1 = vfcvt_f_x_v_f16m2(_acc1, vl);
    vfloat16m2_t _tmp2 = vfcvt_f_x_v_f16m2(_acc2, vl);
    vfloat16m2_t _tmp3 = vfcvt_f_x_v_f16m2(_acc3, vl);
    vfloat16m2_t _tmp4 = vfcvt_f_x_v_f16m2(_acc4, vl);
    vfloat16m2_t _tmp5 = vfcvt_f_x_v_f16m2(_acc5, vl);
    vfloat16m2_t _tmp6 = vfcvt_f_x_v_f16m2(_acc6, vl);
    vfloat16m2_t _tmp7 = vfcvt_f_x_v_f16m2(_acc7, vl);
    _tmp0 = vfmul_vf_f16m2(_tmp0, ratio, vl);
    _tmp1 = vfmul_vf_f16m2(_tmp1, ratio, vl);
    _tmp2 = vfmul_vf_f16m2(_tmp2, ratio, vl);
    _tmp3 = vfmul_vf_f16m2(_tmp3, ratio, vl);
    _tmp4 = vfmul_vf_f16m2(_tmp4, ratio, vl);
    _tmp5 = vfmul_vf_f16m2(_tmp5, ratio, vl);
    _tmp6 = vfmul_vf_f16m2(_tmp6, ratio, vl);
    _tmp7 = vfmul_vf_f16m2(_tmp7, ratio, vl);

    vint16m2_t _res0 = vfcvt_x_f_v_i16m2(_tmp0, vl);
    vint16m2_t _res1 = vfcvt_x_f_v_i16m2(_tmp1, vl);
    vint16m2_t _res2 = vfcvt_x_f_v_i16m2(_tmp2, vl);
    vint16m2_t _res3 = vfcvt_x_f_v_i16m2(_tmp3, vl);
    vint16m2_t _res4 = vfcvt_x_f_v_i16m2(_tmp4, vl);
    vint16m2_t _res5 = vfcvt_x_f_v_i16m2(_tmp5, vl);
    vint16m2_t _res6 = vfcvt_x_f_v_i16m2(_tmp6, vl);
    vint16m2_t _res7 = vfcvt_x_f_v_i16m2(_tmp7, vl);
    _res0 = vadd_vx_i16m2(_res0, z2, vl);
    _res1 = vadd_vx_i16m2(_res1, z2, vl);
    _res2 = vadd_vx_i16m2(_res2, z2, vl);
    _res3 = vadd_vx_i16m2(_res3, z2, vl);
    _res4 = vadd_vx_i16m2(_res4, z2, vl);
    _res5 = vadd_vx_i16m2(_res5, z2, vl);
    _res6 = vadd_vx_i16m2(_res6, z2, vl);
    _res7 = vadd_vx_i16m2(_res7, z2, vl);

    int8_t *out_ptr = dst + (oh * out_w + ow) * packn;
    vse8_v_i8m1(out_ptr + 0 * packn, vnclip_wx_i8m1(_res0, 0, vl), vl);
    vse8_v_i8m1(out_ptr + 1 * packn, vnclip_wx_i8m1(_res1, 0, vl), vl);
    vse8_v_i8m1(out_ptr + 2 * packn, vnclip_wx_i8m1(_res2, 0, vl), vl);
    vse8_v_i8m1(out_ptr + 3 * packn, vnclip_wx_i8m1(_res3, 0, vl), vl);
    vse8_v_i8m1(out_ptr + 4 * packn, vnclip_wx_i8m1(_res4, 0, vl), vl);
    vse8_v_i8m1(out_ptr + 5 * packn, vnclip_wx_i8m1(_res5, 0, vl), vl);
    vse8_v_i8m1(out_ptr + 6 * packn, vnclip_wx_i8m1(_res6, 0, vl), vl);
    vse8_v_i8m1(out_ptr + 7 * packn, vnclip_wx_i8m1(_res7, 0, vl), vl);
}

static void avgpool_border_int8_packn(const int8_t *src, int8_t *dst,
                                      struct csinn_pool_params *params, int oh, int ow,
                                      int idx_h_start, int idx_h_end, int in_w, int out_w,
                                      enum avgpool_loc_enum loc, __fp16 real_scale, int z1, int z2)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int kernel_w = params->filter_width;
    int stride_w = params->stride_width;

    int i_w_start = -params->pad_left + ow * stride_w;
    int i_w_end = i_w_start + kernel_w;
    const int idx_w_start = max(i_w_start, 0);
    const int idx_w_end = min(i_w_end, in_w);

    int window_size = shl_rvv_avgpool_get_window_size(params, idx_h_start, idx_h_end, idx_w_start,
                                                      idx_w_end, loc);
    __fp16 ratio = 1.0f / window_size;
    ratio *= real_scale;
    int z1xn = z1 * window_size;

    vint16m2_t _acc = vmv_v_x_i16m2(0, vl);
    for (int h = idx_h_start; h < idx_h_end; h++) {
        for (int w = idx_w_start; w < idx_w_end; w++) {
            const int8_t *in_ptr = src + (h * in_w + w) * packn;
            _acc = vadd_vv_i16m2(_acc, vwadd_vx_i16m2(vle8_v_i8m1(in_ptr, vl), 0, vl), vl);
        }
    }

    _acc = vsub_vx_i16m2(_acc, z1xn, vl);
    vfloat16m2_t _tmp = vfcvt_f_x_v_f16m2(_acc, vl);
    _tmp = vfmul_vf_f16m2(_tmp, ratio, vl);
    vint16m2_t _res = vfcvt_x_f_v_i16m2(_tmp, vl);
    _res = vadd_vx_i16m2(_res, z2, vl);

    int8_t *out_ptr = dst + (oh * out_w + ow) * packn;
    vse8_v_i8m1(out_ptr, vnclip_wx_i8m1(_res, 0, vl), vl);
}

int shl_rvv_avgpool_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

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

    int pad_top = params->pad_top;
    int pad_left = params->pad_left;

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int dst_start_h = max((pad_top + stride_h - 1) / stride_h, 0);
    int dst_end_h = min((in_h + pad_top - kernel_h) / stride_h + 1, out_h);
    int dst_1x8_start_w = max((pad_left + stride_w - 1) / stride_w, 0);
    int dst_1x8_end_w = min((in_w + pad_left - kernel_w) / stride_w + 1, out_w);

    __fp16 real_scale = input->qinfo->scale / output->qinfo->scale;
    int z1 = input->qinfo->zero_point;
    int z2 = output->qinfo->zero_point;

    for (int bc = 0; bc < batch * in_c; bc += packn) {
        const int8_t *in_ptr = input_data + bc * in_h * in_w;
        int8_t *out_ptr = output_data + bc * out_h * out_w;

        int oh = 0;
        for (; oh < dst_start_h; oh++) {
            int in_h_start = -pad_top + oh * stride_h;
            int in_h_end = in_h_start + kernel_h;
            const int idx_h_start = max(in_h_start, 0);
            const int idx_h_end = min(in_h_end, in_h);
            int ow = 0;
            for (; ow < dst_1x8_start_w; ow++) {
                avgpool_border_int8_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                          in_w, out_w, AVGPOOL_LEFT_TOP, real_scale, z1, z2);
            }
            for (; ow + 7 < dst_1x8_end_w; ow += 8) {
                avgpool_w8_int8_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end, in_w,
                                      out_w, AVGPOOL_TOP, real_scale, z1, z2);
            }
            for (; ow < out_w; ow++) {
                avgpool_border_int8_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                          in_w, out_w, AVGPOOL_RIGHT_TOP, real_scale, z1, z2);
            }
        }
        for (; oh < dst_end_h; oh++) {
            int in_h_start = -pad_top + oh * stride_h;
            int in_h_end = in_h_start + kernel_h;
            const int idx_h_start = max(in_h_start, 0);
            const int idx_h_end = min(in_h_end, in_h);
            int ow = 0;
            for (; ow < dst_1x8_start_w; ow++) {
                avgpool_border_int8_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                          in_w, out_w, AVGPOOL_LEFT, real_scale, z1, z2);
            }
            for (; ow + 7 < dst_1x8_end_w; ow += 8) {
                avgpool_w8_int8_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end, in_w,
                                      out_w, AVGPOOL_CENTER, real_scale, z1, z2);
            }
            for (; ow < out_w; ow++) {
                avgpool_border_int8_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                          in_w, out_w, AVGPOOL_RIGHT, real_scale, z1, z2);
            }
        }
        for (; oh < out_h; oh++) {
            int in_h_start = -pad_top + oh * stride_h;
            int in_h_end = in_h_start + kernel_h;
            const int idx_h_start = max(in_h_start, 0);
            const int idx_h_end = min(in_h_end, in_h);
            int ow = 0;
            for (; ow < dst_1x8_start_w; ow++) {
                avgpool_border_int8_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                          in_w, out_w, AVGPOOL_LEFT_BOTTOM, real_scale, z1, z2);
            }
            for (; ow + 7 < dst_1x8_end_w; ow += 8) {
                avgpool_w8_int8_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end, in_w,
                                      out_w, AVGPOOL_BOTTOM, real_scale, z1, z2);
            }
            for (; ow < out_w; ow++) {
                avgpool_border_int8_packn(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                          in_w, out_w, AVGPOOL_RIGHT_BOTTOM, real_scale, z1, z2);
            }
        }
    }
    return CSINN_TRUE;
}
