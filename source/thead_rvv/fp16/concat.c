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
#include "shl_thead_rvv.h"

static int shl_rvv_concat_ndarray_fp16(struct csinn_tensor **input, struct csinn_tensor *output,
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
    int vl;
    __fp16 *output_ptr = output->data;
    __fp16 out_scale = output->qinfo->scale;
    for (int k = 0; k < outer_size; k++) {
        for (int i = 0; i < params->inputs_count; ++i) {
            struct csinn_tensor *input_item = input[i];
            __fp16 *input_item_data = input_item->data;
            __fp16 in_scale = input_item->qinfo->scale;
            int copy_size = input_item->dim[params->axis] * base_inner_size;
            __fp16 *input_ptr = input_item_data + k * copy_size;

            while (copy_size > 0) {
                vl = vsetvl_e16m2(copy_size);
                vfloat16m2_t _input = vle16_v_f16m2(input_ptr, vl);
                input_ptr += vl;
                vse16_v_f16m2(output_ptr, _input, vl);
                output_ptr += vl;
                copy_size -= vl;
            }
        }
    }
    return CSINN_TRUE;
}

int shl_rvv_concat_fp16(struct csinn_tensor **input, struct csinn_tensor *output,
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
        return shl_rvv_concat_ndarray_fp16(input, output, params);
    } else {
        /* TODO: support more layout */
        if (axis == 1) {
            for (int i = 0; i < params->inputs_count; i++) {
                struct csinn_tensor *input_item = input[i];
                if (input_item->layout == CSINN_LAYOUT_NC1HWC0) {
                    shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input_item);
                } else if (input_item->layout == CSINN_LAYOUT_NCHW) {
                    continue;
                } else {
                    shl_debug_error("%s: unsupport layout %d\n", __func__, input_item->layout);
                    return CSINN_UNSUPPORT_LAYOUT;
                }
            }
            return shl_rvv_concat_ndarray_fp16(input, output, params);
        } else {
            for (int i = 0; i < params->inputs_count; i++) {
                struct csinn_tensor *input_item = input[i];
                if (input_item->layout == CSINN_LAYOUT_NCHW) {
                    shl_rvv_tensor_ndarray_to_nc1xc0_replace_fp16(input_item);
                } else if (input_item->layout == CSINN_LAYOUT_NC1HWC0) {
                    continue;
                } else {
                    shl_debug_error("%s: unsupport layout %d\n", __func__, input_item->layout);
                    return CSINN_UNSUPPORT_LAYOUT;
                }
            }
            return shl_rvv_concat_ndarray_fp16(input, output, params);
        }
    }
    return CSINN_TRUE;
}
