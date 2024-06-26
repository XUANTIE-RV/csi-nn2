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

static int shl_rvv_concat_ndarray_int8(struct csinn_tensor **input, struct csinn_tensor *output,
                                       struct csinn_concat_params *params)
{
    /* update output tensor */
    output->layout = input[0]->layout;
    output->dim_count = input[0]->dim_count;
    for (int i = 0; i < output->dim_count; i++) {
        output->dim[i] = input[0]->dim[i];
    }
    int axis_shape = 0;
    for (int i = 0; i < params->inputs_count; i++) {
        axis_shape += input[i]->dim[params->axis];
    }
    output->dim[params->axis] = axis_shape;

    int64_t outer_size = 1;
    for (int i = 0; i < params->axis; ++i) {
        outer_size *= output->dim[i];
    }
    int64_t base_inner_size = 1;
    for (int i = params->axis + 1; i < output->dim_count; ++i) {
        base_inner_size *= output->dim[i];
    }
    for (int q = 0; q < params->inputs_count; q++) {
        struct csinn_tensor *input_item = input[q];
        shl_quantize_multiplier(input_item->qinfo->scale / output->qinfo->scale,
                                &input_item->qinfo->multiplier, &input_item->qinfo->shift);
    }
    int vl;
    int8_t *output_ptr = (int8_t *)output->data;
    for (int k = 0; k < outer_size; k++) {
        for (int i = 0; i < params->inputs_count; ++i) {
            struct csinn_tensor *input_item = input[i];
            int8_t *input_item_data = (int8_t *)input_item->data;
            int copy_size = input_item->dim[params->axis] * base_inner_size;
            const int8_t *input_ptr = input_item_data + k * copy_size;
            /* input has same quant_info with output */
            if (memcmp(input_item->qinfo, output->qinfo, sizeof(struct csinn_quant_info)) == 0) {
                while (copy_size > 0) {
                    vl = vsetvl_e8m2(copy_size);
                    vint8m2_t _input = vle8_v_i8m2(input_ptr, vl);
                    input_ptr += vl;
                    vse8_v_i8m2(output_ptr, _input, vl);
                    output_ptr += vl;
                    copy_size -= vl;
                }
            } else {
                /*
                while (copy_size > 0) {
                    vl = vsetvl_e8m1(copy_size);
                    vint8m1_t _input = vle8_v_i8m1(input_ptr, vl);
                    vint16m2_t _input1 = vwsub_vx_i16m2(_input, input_item->qinfo->zero_point, vl);
                    vint32m4_t _input2 = vwadd_vx_i32m4(_input1, 0, vl);  // widden 16->32
                    vint32m4_t _mulh;
                    if (input_item->qinfo->shift < 0) {
                        _mulh = vmulh_vx_i32m4(_input2, input_item->qinfo->multiplier, vl);
                        _mulh = vssra_vx_i32m4(_mulh, -input_item->qinfo->shift - 1, vl);
                    } else {
                        _mulh = vsll_vx_i32m4(_input2, input_item->qinfo->shift + 1, vl);
                        _mulh = vmulh_vx_i32m4(_mulh, input_item->qinfo->multiplier, vl);
                    }
                    vint32m4_t _res0 = vadd_vx_i32m4(_mulh, output->qinfo->zero_point, vl);
                    vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);
                    vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);
                    vse8_v_i8m1(output_ptr, _res2, vl);
                    input_ptr += vl;
                    output_ptr += vl;
                    copy_size -= vl;
                }
                */
                while (copy_size > 0) {
                    vl = vsetvl_e8m1(copy_size);
                    vint8m1_t _input = vle8_v_i8m1(input_ptr, vl);
                    vint16m2_t _input1 = vwsub_vx_i16m2(_input, input_item->qinfo->zero_point, vl);
                    vfloat16m2_t _inputf = vfcvt_f_x_v_f16m2(_input1, vl);
                    vfloat16m2_t _tmp = vfmul_vf_f16m2(
                        _inputf, input_item->qinfo->scale / output->qinfo->scale, vl);
                    vint16m2_t _res = vfcvt_x_f_v_i16m2(_tmp, vl);
                    _res = vadd_vx_i16m2(_res, output->qinfo->zero_point, vl);
                    vse8_v_i8m1(output_ptr, vnclip_wx_i8m1(_res, 0, vl), vl);
                    input_ptr += vl;
                    output_ptr += vl;
                    copy_size -= vl;
                }
            }
        }
    }
    return CSINN_TRUE;
}

int shl_rvv_concat_int8(struct csinn_tensor **input, struct csinn_tensor *output,
                        struct csinn_concat_params *params)
{
    int axis = params->axis;
    int ch_interleave = 0;
    int same_layout = 1;
    for (int i = 1; i < params->inputs_count; i++) {
        if (input[i]->layout != input[i - 1]->layout) {
            same_layout = 0;
            break;
        }
    }
    if (same_layout) {
        return shl_rvv_concat_ndarray_int8(input, output, params);
    } else {
        /* TODO: support more layout */
        if (axis == 1) {
            for (int i = 0; i < params->inputs_count; i++) {
                struct csinn_tensor *input_item = input[i];
                if (input_item->layout == CSINN_LAYOUT_NC1HWC0) {
                    shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input_item);
                } else if (input_item->layout == CSINN_LAYOUT_NCHW) {
                    continue;
                } else {
                    shl_debug_error("%s: unsupport layout %d\n", __func__, input_item->layout);
                    return CSINN_UNSUPPORT_LAYOUT;
                }
            }
            return shl_rvv_concat_ndarray_int8(input, output, params);
        } else {
            for (int i = 0; i < params->inputs_count; i++) {
                struct csinn_tensor *input_item = input[i];
                if (input_item->layout == CSINN_LAYOUT_NCHW) {
                    shl_rvv_tensor_ndarray_to_nc1xc0_replace_int8(input_item);
                } else if (input_item->layout == CSINN_LAYOUT_NC1HWC0) {
                    continue;
                } else {
                    shl_debug_error("%s: unsupport layout %d\n", __func__, input_item->layout);
                    return CSINN_UNSUPPORT_LAYOUT;
                }
            }
            return shl_rvv_concat_ndarray_int8(input, output, params);
        }
    }
    return CSINN_TRUE;
}
