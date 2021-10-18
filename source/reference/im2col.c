/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_nn.h"
#include "csi_utils.h"

// input_data layout:NCHW
// https://github.com/pjreddie/darknet/blob/master/src/im2col.c
// output_data: row = channels*ksize_h*ksize_w, col = batch*height_col*width_col
static int csi_im2col_nchw_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct im2col_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int32_t batch = input->dim[0];
    int32_t channel = input->dim[1];
    int32_t height = input->dim[2];
    int32_t width = input->dim[3];
    int32_t ksize_h = params->kernel_h;
    int32_t ksize_w = params->kernel_w;
    int32_t stride_h = params->stride_h;
    int32_t stride_w = params->stride_w;

    int height_col = (height + params->pad_top + params->pad_down - ksize_h) / stride_h + 1;    // output_height
    int width_col  = (width + params->pad_left + params->pad_right - ksize_w) / stride_w + 1;   // output_width,  batch * output_height * output_width = matrix_col
    int channel_col = channel * ksize_h * ksize_w;

    for(int c = 0; c < channel_col; ++c) {
        int w_offset = c % ksize_w;
        int h_offset = c / ksize_w % ksize_h;
        int c_im = c / ksize_h / ksize_w;
        for(int b = 0; b < batch; ++b) {
            for(int h = 0; h < height_col; ++h) {
                for(int w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h * stride_h;
                    int im_col = w_offset + w * stride_w;
                    int col_index = ((c * batch + b) * height_col + h) * width_col + w;
                    im_row = im_row - params->pad_top;
                    im_col = im_col - params->pad_left;
                    if(im_row < 0 || im_col < 0 || im_row >= height || im_col >=width) {
                        output_data[col_index] = 0.0f;
                    } else {
                        output_data[col_index] = input_data[csi_get_index(input->dim, b, c_im, im_row, im_col)];
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_im2col_nchw_u8(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct im2col_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;
    int32_t batch = input->dim[0];
    int32_t channel = input->dim[1];
    int32_t height = input->dim[2];
    int32_t width = input->dim[3];
    int32_t ksize_h = params->kernel_h;
    int32_t ksize_w = params->kernel_w;
    int32_t stride_h = params->stride_h;
    int32_t stride_w = params->stride_w;

    int height_col = (height + params->pad_top + params->pad_down - ksize_h) / stride_h + 1;
    int width_col  = (width + params->pad_left + params->pad_right - ksize_w) / stride_w + 1;
    int channel_col = channel * ksize_h * ksize_w;

    for(int c = 0; c < channel_col; ++c) {
        int w_offset = c % ksize_h;
        int h_offset = c / ksize_w % ksize_h;
        int c_im = c / ksize_h / ksize_w;
        for(int b = 0; b < batch; ++b) {
            for(int h = 0; h < height_col; ++h) {
                for(int w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h * stride_h;
                    int im_col = w_offset + w * stride_w;
                    int col_index = ((c * batch + b) * height_col + h) * width_col + w;
                    im_row = im_row - params->pad_top;
                    im_col = im_col - params->pad_left;
                    if(im_row < 0 || im_col < 0 || im_row >= height || im_col >=width) {
                        output_data[col_index] = csi_quantize_f32_to_u8(0.0f, input->zero_point, input->multiplier, input->shift);
                    }
                    output_data[col_index] = input_data[csi_get_index(input->dim, b, c_im, im_row, im_col)];
                }
            }
        }
    }
    return CSINN_TRUE;
}


// input_data layout:NHWC
// output_data: row = batch*height_col*width_col, col = channels*ksize_h*ksize_w
static int csi_im2col_nhwc_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct im2col_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int32_t batch = input->dim[0];
    int32_t channel = input->dim[3];
    int32_t height = input->dim[1];
    int32_t width = input->dim[2];
    int32_t ksize_h = params->kernel_h;
    int32_t ksize_w = params->kernel_w;
    int32_t stride_h = params->stride_h;
    int32_t stride_w = params->stride_w;


    int height_col = (height + params->pad_top + params->pad_down - ksize_h) / stride_h + 1;    // output_height
    int width_col  = (width + params->pad_left + params->pad_right - ksize_w) / stride_w + 1;   // output_width,  output_height * output_width = matrix_
    int channel_col = channel * ksize_h * ksize_w;

    for(int b = 0; b < batch; ++b) {
        for(int h = 0; h < height_col; ++h) {
            for(int w = 0; w < width_col; ++w) {
                for(int c = 0; c < channel_col; ++c) {
                    int w_offset = c % ksize_w;
                    int h_offset = c / ksize_w % ksize_h;
                    int c_im = c / ksize_h / ksize_w;

                    int im_row = h_offset + h * stride_h;
                    int im_col = w_offset + w * stride_w;
                    int col_index = ((b * height_col + h) * width_col + w) * channel_col + c;
                    im_row = im_row - params->pad_top;
                    im_col = im_col - params->pad_left;
                    if(im_row < 0 || im_col < 0 || im_row >= height || im_col >=width) {
                        output_data[col_index] = 0.0f;
                    } else {
                        output_data[col_index] = input_data[csi_get_index(input->dim, b, im_row, im_col, c_im)];
                    }
                }
            }
        }
    }

/*
output_data layout: row = channels*ksize_h*ksize_w, col = batch*height_col*width_col
*/
    // for(int c = 0; c < channel_col; ++c) {
    //     int w_offset = c % ksize_w;
    //     int h_offset = c / ksize_w % ksize_h;
    //     int c_im = c / ksize_h / ksize_w;
    //     for(int b = 0; b < batch; ++b) {
    //         for(int h = 0; h < height_col; ++h) {
    //             for(int w = 0; w < width_col; ++w) {
    //                 int im_row = h_offset + h * stride_h;
    //                 int im_col = w_offset + w * stride_w;
    //                 int col_index = ((c * batch + b) * height_col + h) * width_col + w;
    //                 im_row = im_row - params->pad_top;
    //                 im_col = im_col - params->pad_left;
    //                 if(im_row < 0 || im_col < 0 || im_row >= height || im_col >=width) {
    //                     output_data[col_index] = 0.0f;
    //                 } else {
    //                     output_data[col_index] = input_data[csi_get_index(input->dim, b, im_row, im_col, c_im)];
    //                 }
    //             }
    //         }
    //     }
    // }

    return CSINN_TRUE;
}

static int csi_im2col_nhwc_u8(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct im2col_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;
    int32_t batch = input->dim[0];
    int32_t channel = input->dim[3];
    int32_t height = input->dim[1];
    int32_t width = input->dim[2];
    int32_t ksize_h = params->kernel_h;
    int32_t ksize_w = params->kernel_w;
    int32_t stride_h = params->stride_h;
    int32_t stride_w = params->stride_w;

    int height_col = (height + params->pad_top + params->pad_down - ksize_h) / stride_h + 1;    // output_height
    int width_col  = (width + params->pad_left + params->pad_right - ksize_w) / stride_w + 1;   // output_width,  output_height * output_width = matrix_
    int channel_col = channel * ksize_h * ksize_w;

    for(int b = 0; b < batch; ++b) {
        for(int h = 0; h < height_col; ++h) {
            for(int w = 0; w < width_col; ++w) {
                for(int c = 0; c < channel_col; ++c) {
                    int w_offset = c % ksize_w;
                    int h_offset = c / ksize_w % ksize_h;
                    int c_im = c / ksize_h / ksize_w;

                    int im_row = h_offset + h * stride_h;
                    int im_col = w_offset + w * stride_w;
                    int col_index = ((b * height_col + h) * width_col + w) * channel_col + c;
                    im_row = im_row - params->pad_top;
                    im_col = im_col - params->pad_left;
                    if(im_row < 0 || im_col < 0 || im_row >= height || im_col >=width) {
                        output_data[col_index] = csi_quantize_f32_to_u8(0.0f, input->zero_point, input->multiplier, input->shift);
                    } else {
                        output_data[col_index] = input_data[csi_get_index(input->dim, b, im_row, im_col, c_im)];
                    }
                }
            }
        }
    }

    // for(int c = 0; c < channel_col; ++c) {
    //     int w_offset = c % ksize_w;
    //     int h_offset = c / ksize_w % ksize_h;
    //     int c_im = c / ksize_h / ksize_w;
    //     for(int b = 0; b < batch; ++b) {
    //         for(int h = 0; h < height_col; ++h) {
    //             for(int w = 0; w < width_col; ++w) {
    //                 int im_row = h_offset + h * stride_h;
    //                 int im_col = w_offset + w * stride_w;
    //                 int col_index = ((c * batch + b) * height_col + h) * width_col + w;
    //                 im_row = im_row - params->pad_top;
    //                 im_col = im_col - params->pad_left;
    //                 if(im_row < 0 || im_col < 0 || im_row >= height || im_col >=width) {
    //                     output_data[col_index] = csi_quantize_f32_to_u8(0.0f, input->zero_point, input->multiplier, input->shift);
    //                 } else {
    //                     output_data[col_index] = input_data[csi_get_index(input->dim, b, im_row, im_col, c_im)];
    //                 }
    //             }
    //         }
    //     }
    // }

    return CSINN_TRUE;
}


int csi_im2col_f32(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct im2col_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_im2col_nchw_f32(input, output, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_im2col_nhwc_f32(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int csi_im2col_u8(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct im2col_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_im2col_nchw_u8(input, output, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_im2col_nhwc_u8(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}


int csi_im2col_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct im2col_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_IM2COL, input->dtype);
    if(params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_im2col(struct csi_tensor *input,
               struct csi_tensor *output,
               struct im2col_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
