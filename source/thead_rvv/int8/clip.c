/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "rvv/rvv.h"

/*************************************************************
 * note: support flexible vlen
 *************************************************************/

int shl_rvv_clip_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_clip_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    // TODO: move to init api
    // real_scale > 1 =>  output->qinfo->shift > 0  ==> shift left
    float real_scale = input->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale, &output->qinfo->multiplier, &output->qinfo->shift);
    /* ignore overflow, limit truncation */
    int8_t clip_min = params->min_value / output->qinfo->scale + output->qinfo->zero_point;
    int8_t clip_max = params->max_value / output->qinfo->scale + output->qinfo->zero_point;

    int size = csinn_tensor_size(input);
    while (size > 0) {
        int vl = vsetvl_e8m1(size);

        vint8m1_t _input = vle8_v_i8m1(input_data, vl);
        vint16m2_t _input1 = vwsub_vx_i16m2(_input, input->qinfo->zero_point, vl);  // widden 8->16
        vint32m4_t _input2 = vwadd_vx_i32m4(_input1, 0, vl);                        // widden 16->32

        // mulh 无 round 过程, 左移时多移1位，mulh 后再用带round的右移1位来实现类似round的功能
        vint32m4_t _tmp = vsll_vx_i32m4(_input2, output->qinfo->shift + 2, vl);
        vint32m4_t _mulh = vmulh_vx_i32m4(_tmp, output->qinfo->multiplier, vl);
        _mulh = vssra_vx_i32m4(_mulh, 1, vl);

        vint32m4_t _res0 = vadd_vx_i32m4(_mulh, output->qinfo->zero_point, vl);  // +z2 (z2 = -128)
        vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);                        // narrow 32->16
        vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);                          // narrow 16->8
        vint8m1_t _res = vmax_vx_i8m1(_res2, clip_min, vl);
        _res = vmin_vx_i8m1(_res, clip_max, vl);

        vse8_v_i8m1(output_data, _res, vl);
        input_data += vl;
        output_data += vl;
        size -= vl;
    }
    output->layout = input->layout;
    output->dim_count = input->dim_count;
    for (int i = 0; i < output->dim_count; i++) {
        output->dim[i] = input->dim[i];
    }
    return CSINN_TRUE;
}
