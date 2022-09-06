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

/* CSI-NN2 version 2.0.x */

#include "shl_ref.h"

static void shl_ref_resize_bilinear_nhwc_f32(struct csinn_tensor *input,
                                             struct csinn_tensor *output, bool align_corners)
{
    float *input_data = input->data;
    float *output_data = output->data;
    int32_t batches = input->dim[0];
    int32_t input_height = input->dim[1];
    int32_t input_width = input->dim[2];
    int32_t depth = input->dim[3];

    int32_t output_height = output->dim[1];
    int32_t output_width = output->dim[2];

    float height_scale = 0;
    float width_scale = 0;

    if (align_corners) {
        height_scale = (float)(input_height - 1) / (output_height - 1);
        width_scale = (float)(input_width - 1) / (output_width - 1);
    } else {
        height_scale = (float)(input_height) / output_height;
        width_scale = (float)(input_width) / output_width;
    }

    for (int b = 0; b < batches; ++b) {
        for (int y = 0; y < output_height; ++y) {
            float input_y = y * height_scale;
            int32_t y0 = (int32_t)(floor(input_y));
            int32_t y1 = shl_ref_min_internal_s32(y0 + 1, input_height - 1);
            for (int x = 0; x < output_width; ++x) {
                float input_x = x * width_scale;
                int32_t x0 = (int32_t)(floor(input_x));
                int32_t x1 = shl_ref_min_internal_s32(x0 + 1, input_width - 1);
                for (int c = 0; c < depth; ++c) {
                    float interpolation =
                        (float)(input_data[shl_ref_get_index(input->dim, b, y0, x0, c)] *
                                    (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                                input_data[shl_ref_get_index(input->dim, b, y1, x0, c)] *
                                    (input_y - y0) * (1 - (input_x - x0)) +
                                input_data[shl_ref_get_index(input->dim, b, y0, x1, c)] *
                                    (1 - (input_y - y0)) * (input_x - x0) +
                                input_data[shl_ref_get_index(input->dim, b, y1, x1, c)] *
                                    (input_y - y0) * (input_x - x0));
                    output_data[shl_ref_get_index(output->dim, b, y, x, c)] = interpolation;
                }
            }
        }
    }
}

/*reference
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/resize_nearest_neighbor.h
 */
static void shl_ref_resize_nearest_neighbor_f32(struct csinn_tensor *input,
                                                struct csinn_tensor *output, bool align_corners)
{
    float *input_data = input->data;
    float *output_data = output->data;
    int32_t batches = input->dim[0];
    int32_t input_height = input->dim[1];
    int32_t input_width = input->dim[2];
    int32_t depth = input->dim[3];

    int32_t output_height = output->dim[1];
    int32_t output_width = output->dim[2];

    float height_scale = 0;
    float width_scale = 0;

    if (align_corners) {
        height_scale = (float)(input_height - 1) / (output_height - 1);
        width_scale = (float)(input_width - 1) / (output_width - 1);
    } else {
        height_scale = (float)(input_height) / output_height;
        width_scale = (float)(input_width) / output_width;
    }

    const int col_offset = input->dim[3];
    const int row_offset = input->dim[2] * col_offset;
    const int batch_offset = input->dim[1] * row_offset;

    const float *input_ptr = input_data;
    float *output_ptr = output_data;
    for (int b = 0; b < batches; ++b) {
        for (int y = 0; y < output_height; ++y) {
            int32_t in_y =
                shl_ref_min_internal_s32(align_corners ? (int32_t)(round(y * height_scale))
                                                       : (int32_t)(floor(y * height_scale)),
                                         input_height - 1);
            const float *y_input_ptr = input_ptr + in_y * row_offset;
            for (int x = 0; x < output_width; ++x) {
                int32_t in_x =
                    shl_ref_min_internal_s32(align_corners ? (int32_t)(round(x * width_scale))
                                                           : (int32_t)(floor(x * width_scale)),
                                             input_width - 1);
                const float *x_input_ptr = y_input_ptr + in_x * col_offset;
                memcpy(output_ptr, x_input_ptr, depth * sizeof(float));
                output_ptr += depth;
            }
        }
        input_ptr += batch_offset;
    }
}

static void shl_ref_resize_nearest_neighbor_nchw_f32(struct csinn_tensor *o_input,
                                                     struct csinn_tensor *o_output,
                                                     bool align_corners)
{
    struct csinn_tensor *input = shl_ref_nchw_to_nhwc_f32(o_input);
    struct csinn_tensor *output = shl_ref_nchw_to_nhwc_f32(o_output);
    shl_ref_resize_nearest_neighbor_f32(input, output, align_corners);
    shl_ref_nhwc_to_nchw_f32(o_output, output);
    shl_ref_free_float_tensor(input);
}

static void shl_ref_resize_bilinear_nchw_f32(struct csinn_tensor *o_input,
                                             struct csinn_tensor *o_output, bool align_corners)
{
    struct csinn_tensor *input = shl_ref_nchw_to_nhwc_f32(o_input);
    struct csinn_tensor *output = shl_ref_nchw_to_nhwc_f32(o_output);
    shl_ref_resize_bilinear_nhwc_f32(input, output, align_corners);
    shl_ref_nhwc_to_nchw_f32(o_output, output);
    shl_ref_free_float_tensor(input);
}

int shl_ref_resize_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_resize_params *params)
{
    if (params->resize_mode == CSINN_RESIZE_BILINEAR) {
        if (params->base.layout == CSINN_LAYOUT_NCHW) {
            shl_ref_resize_bilinear_nchw_f32(input, output, params->align_corners);
        } else {
            shl_ref_resize_bilinear_nhwc_f32(input, output, params->align_corners);
        }
    } else if (params->resize_mode == CSINN_RESIZE_NEAREST_NEIGHBOR) {
        if (params->base.layout == CSINN_LAYOUT_NCHW) {
            shl_ref_resize_nearest_neighbor_nchw_f32(input, output, params->align_corners);
        } else {
            shl_ref_resize_nearest_neighbor_f32(input, output, params->align_corners);
        }
    } else {
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

int shl_ref_resize_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_resize_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_resize_f32);
}
