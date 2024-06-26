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

int shl_rvv_reduce_sum_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
    }

    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    // TODO: move to init api
    float real_scale = input->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale, &output->qinfo->multiplier, &output->qinfo->shift);

    if (*(params->axis) == -1) {
        int size = 1;
        for (int i = 0; i < input->dim_count; i++) {
            size = size * input->dim[i];
        }
        float res = 0;
        for (int j = 0; j < size; j++) {
            float temp = (input_data[j] - input->qinfo->zero_point) * input->qinfo->scale;
            res = res + temp;
        }
        float ret = round(res / output->qinfo->scale) + output->qinfo->zero_point;
        if (ret > 127)
            ret = 127;
        else if (ret < -128)
            ret = -128;
        *output_data = (int8_t)ret;
    } else {
        int axis = *(params->axis);
        int64_t outer_size = 1;
        for (int i = 0; i < axis; i++) {
            outer_size *= input->dim[i];
        }
        int64_t inner_size = 1;
        for (int i = axis + 1; i < input->dim_count; i++) {
            inner_size *= input->dim[i];
        }
        int cnt = input->dim[axis];

        for (int i = 0; i < outer_size; i++) {
            int8_t *in_shadow_ptr = input_data;
            int k = inner_size;
            while (k > 0) {
                int vl = vsetvl_e8m1(k);
                int8_t *in_ptr = in_shadow_ptr;
                vint32m4_t _acc = vmv_v_x_i32m4(0, vl);
                for (int j = 0; j < cnt; j++) {
                    vint8m1_t _input = vle8_v_i8m1(in_ptr, vl);
                    vint16m2_t _input1 = vwadd_vx_i16m2(_input, 0, vl);   // widden 8->16
                    vint32m4_t _input2 = vwadd_vx_i32m4(_input1, 0, vl);  // widden 16->32

                    vint32m4_t _tmp = vsub_vx_i32m4(_input2, input->qinfo->zero_point, vl);
                    _acc = vadd_vv_i32m4(_acc, _tmp, vl);
                    in_ptr += inner_size;
                }
                vint32m4_t _mulh = vmulh_vx_i32m4(_acc, output->qinfo->multiplier, vl);
                vint32m4_t _res;

                if (output->qinfo->shift < 0) {
                    _res = vssra_vx_i32m4(_mulh, -output->qinfo->shift - 1, vl);
                } else {
                    _res = vsll_vx_i32m4(_mulh, output->qinfo->shift + 1, vl);
                }

                vint32m4_t _res0 =
                    vadd_vx_i32m4(_res, output->qinfo->zero_point, vl);  // +z2 (z2 = -128)
                vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);        // narrow 32->16
                vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);          // narrow 16->8
                vse8_v_i8m1(output_data, _res2, vl);
                output_data += vl;
                in_shadow_ptr += vl;
                k -= vl;
            }
            input_data += inner_size * cnt;
        }
    }
    return CSINN_TRUE;
}
