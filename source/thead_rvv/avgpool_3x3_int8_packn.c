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
 * s2 * (q2 - z2) = avgpool_3x3{ s1 * (q1 - z1) }
 * q2 = s1/s2 * (∑(q1 - z1))/9 + z2
 * constrain: input channel % packn = 0
 *************************************************************/
int shl_rvv_avgpool3x3s2_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int input_size = in_c * in_h * in_w;

    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int output_size = in_c * out_h * out_w;

    int padded_in_h = in_h + params->pad_top + params->pad_down;
    int padded_in_w = in_w + params->pad_left + params->pad_right;
    int padded_in_hw = padded_in_w * padded_in_h;

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int8_t *input_ncxhwx = (int8_t *)shl_mem_alloc(in_c * padded_in_hw * sizeof(int8_t));
    int tailstep = (padded_in_w - 2 * out_w + padded_in_w) * packn;

    // 将ratio合并进s1/s2
    float real_scale = input->qinfo->scale / output->qinfo->scale / 9;
    int32_t multiplier, shift;
    shl_quantize_multiplier(real_scale, &multiplier, &shift);
    int32_t z1x9 = input->qinfo->zero_point * 9;

    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_packn_int8(input_data, input_ncxhwx, in_c, in_h, in_w, padded_in_h,
                                     padded_in_w, params->pad_top, params->pad_left,
                                     input->qinfo->zero_point);

        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            const int8_t *line0 = input_ncxhwx + c * padded_in_h * padded_in_w;
            const int8_t *line1 = line0 + padded_in_w * packn;
            const int8_t *line2 = line1 + padded_in_w * packn;
            int8_t *out = output_data + c * out_h * out_w;

            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    vint16m2_t _acc0 =
                        vwadd_vv_i16m2(vle8_v_i8m1(line0, vl), vle8_v_i8m1(line0 + packn, vl), vl);
                    vint16m2_t _acc1 = vwadd_vv_i16m2(vle8_v_i8m1(line0 + packn * 2, vl),
                                                      vle8_v_i8m1(line1, vl), vl);
                    vint16m2_t _acc2 = vwadd_vv_i16m2(vle8_v_i8m1(line1 + packn, vl),
                                                      vle8_v_i8m1(line1 + packn * 2, vl), vl);
                    vint16m2_t _acc3 =
                        vwadd_vv_i16m2(vle8_v_i8m1(line2, vl), vle8_v_i8m1(line2 + packn, vl), vl);
                    vint16m2_t _acc4 = vwadd_vx_i16m2(vle8_v_i8m1(line2 + packn * 2, vl), 0, vl);
                    _acc0 = vadd_vv_i16m2(_acc0, _acc1, vl);
                    _acc0 = vadd_vv_i16m2(_acc0, _acc2, vl);
                    _acc0 = vadd_vv_i16m2(_acc0, _acc3, vl);
                    _acc0 = vadd_vv_i16m2(_acc0, _acc4, vl);
                    _acc0 = vsub_vx_i16m2(_acc0, (int16_t)z1x9, vl);

                    vint32m4_t _tmp = vwadd_vx_i32m4(_acc0, 0, vl);
                    vint32m4_t _mulh;
                    if (shift >= 0) {
                        // mulh 无 round 过程, 左移时多移1位，mulh
                        // 后再用带round的右移1位来实现类似round的功能
                        _mulh = vsll_vx_i32m4(_tmp, shift + 2, vl);
                        _mulh = vmulh_vx_i32m4(_mulh, multiplier, vl);
                        _mulh = vssra_vx_i32m4(_mulh, 1, vl);
                    } else {
                        _mulh = vmulh_vx_i32m4(_tmp, multiplier, vl);
                        _mulh = vssra_vx_i32m4(_mulh, -shift - 1, vl);
                    }

                    vint32m4_t _res0 = vadd_vx_i32m4(_mulh, output->qinfo->zero_point, vl);
                    vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);
                    vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);

                    vse8_v_i8m1(out, _res2, vl);

                    line0 += packn * 2;
                    line1 += packn * 2;
                    line2 += packn * 2;
                    out += packn;
                }
                line0 += tailstep;
                line1 += tailstep;
                line2 += tailstep;
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    shl_mem_free(input_ncxhwx);
    return CSINN_TRUE;
}

int shl_rvv_avgpool3x3s1_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int input_size = in_c * in_h * in_w;

    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int output_size = in_c * out_h * out_w;

    int padded_in_h = in_h + params->pad_top + params->pad_down;
    int padded_in_w = in_w + params->pad_left + params->pad_right;
    int padded_in_hw = padded_in_w * padded_in_h;

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int8_t *input_ncxhwx = (int8_t *)shl_mem_alloc(in_c * padded_in_hw * sizeof(int8_t));

    // 将ratio合并进s1/s2
    float real_scale = input->qinfo->scale / output->qinfo->scale / 9;
    int32_t multiplier, shift;
    shl_quantize_multiplier(real_scale, &multiplier, &shift);
    int32_t z1x9 = input->qinfo->zero_point * 9;

    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_packn_int8(input_data, input_ncxhwx, in_c, in_h, in_w, padded_in_h,
                                     padded_in_w, params->pad_top, params->pad_left,
                                     input->qinfo->zero_point);

        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            const int8_t *line0 = input_ncxhwx + c * padded_in_h * padded_in_w;
            const int8_t *line1 = line0 + padded_in_w * packn;
            const int8_t *line2 = line1 + padded_in_w * packn;
            int8_t *out = output_data + c * out_h * out_w;

            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    vint16m2_t _acc0 =
                        vwadd_vv_i16m2(vle8_v_i8m1(line0, vl), vle8_v_i8m1(line0 + packn, vl), vl);
                    vint16m2_t _acc1 = vwadd_vv_i16m2(vle8_v_i8m1(line0 + packn * 2, vl),
                                                      vle8_v_i8m1(line1, vl), vl);
                    vint16m2_t _acc2 = vwadd_vv_i16m2(vle8_v_i8m1(line1 + packn, vl),
                                                      vle8_v_i8m1(line1 + packn * 2, vl), vl);
                    vint16m2_t _acc3 =
                        vwadd_vv_i16m2(vle8_v_i8m1(line2, vl), vle8_v_i8m1(line2 + packn, vl), vl);
                    vint16m2_t _acc4 = vwadd_vx_i16m2(vle8_v_i8m1(line2 + packn * 2, vl), 0, vl);
                    _acc0 = vadd_vv_i16m2(_acc0, _acc1, vl);
                    _acc0 = vadd_vv_i16m2(_acc0, _acc2, vl);
                    _acc0 = vadd_vv_i16m2(_acc0, _acc3, vl);
                    _acc0 = vadd_vv_i16m2(_acc0, _acc4, vl);
                    _acc0 = vsub_vx_i16m2(_acc0, (int16_t)z1x9, vl);

                    vint32m4_t _tmp = vwadd_vx_i32m4(_acc0, 0, vl);
                    vint32m4_t _mulh;
                    if (shift >= 0) {
                        // mulh 无 round 过程, 左移时多移1位，mulh
                        // 后再用带round的右移1位来实现类似round的功能
                        _mulh = vsll_vx_i32m4(_tmp, shift + 2, vl);
                        _mulh = vmulh_vx_i32m4(_mulh, multiplier, vl);
                        _mulh = vssra_vx_i32m4(_mulh, 1, vl);
                    } else {
                        _mulh = vmulh_vx_i32m4(_tmp, multiplier, vl);
                        _mulh = vssra_vx_i32m4(_mulh, -shift - 1, vl);
                    }

                    vint32m4_t _res0 = vadd_vx_i32m4(_mulh, output->qinfo->zero_point, vl);
                    vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);
                    vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);

                    vse8_v_i8m1(out, _res2, vl);

                    line0 += packn * 1;
                    line1 += packn * 1;
                    line2 += packn * 1;
                    out += packn;
                }
                line0 += packn * 2;
                line1 += packn * 2;
                line2 += packn * 2;
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    shl_mem_free(input_ncxhwx);
    return CSINN_TRUE;
}