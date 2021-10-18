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

static int csi_pad_nhwc_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pad_params *params)
{
    const int output_batch = output->dim[0];
    const int output_height = output->dim[1];
    const int output_width = output->dim[2];
    const int output_depth = output->dim[3];

    const int left_b_padding = params->pad_before[0];
    const int left_h_padding = params->pad_before[1];
    const int left_w_padding = params->pad_before[2];
    const int left_d_padding = params->pad_before[3];

    const int right_b_padding = params->pad_after[0];
    const int right_h_padding = params->pad_after[1];
    const int right_w_padding = params->pad_after[2];
    const int right_d_padding = params->pad_after[3];

    const float *in_ptr = input->data;
    float *out_ptr = output->data;

    for (int out_b = 0; out_b < output_batch; ++out_b) {
        for (int out_h = 0; out_h < output_height; ++out_h) {
            for (int out_w = 0; out_w < output_width; ++out_w) {
                for (int out_d = 0; out_d < output_depth; ++out_d) {
                    if (out_b < left_b_padding || out_b >= output_batch - right_b_padding ||
                        out_h < left_h_padding || out_h >= output_height - right_h_padding ||
                        out_w < left_w_padding || out_w >= output_width - right_w_padding ||
                        out_d < left_d_padding || out_d >= output_depth - right_d_padding) {
                        if (params->pad_mode == CSINN_PAD_CONSTANT) {
                            *out_ptr = params->pad_value;
                            out_ptr++;
                        } else if (params->pad_mode = CSINN_PAD_EDGE) {
                            /* TODO */
                            assert(0);
                        } else if (params->pad_mode = CSINN_PAD_REFLECT) {
                            /* TODO */
                            assert(0);
                        }
                    } else {
                        *out_ptr = *in_ptr;
                        out_ptr++;
                        in_ptr++;
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_pad_nhwc_u8(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct pad_params *params)
{
    const int output_batch = output->dim[0];
    const int output_height = output->dim[1];
    const int output_width = output->dim[2];
    const int output_depth = output->dim[3];

    const int left_b_padding = params->pad_before[0];
    const int left_h_padding = params->pad_before[1];
    const int left_w_padding = params->pad_before[2];
    const int left_d_padding = params->pad_before[3];

    const int right_b_padding = params->pad_after[0];
    const int right_h_padding = params->pad_after[1];
    const int right_w_padding = params->pad_after[2];
    const int right_d_padding = params->pad_after[3];

    const uint8_t *in_ptr = input->data;
    uint8_t *out_ptr = output->data;
    for (int out_b = 0; out_b < output_batch; ++out_b) {
        for (int out_h = 0; out_h < output_height; ++out_h) {
            for (int out_w = 0; out_w < output_width; ++out_w) {
                for (int out_d = 0; out_d < output_depth; ++out_d) {
                    if (out_b < left_b_padding || out_b >= output_batch - right_b_padding ||
                        out_h < left_h_padding || out_h >= output_height - right_h_padding ||
                        out_w < left_w_padding || out_w >= output_width - right_w_padding ||
                        out_d < left_d_padding || out_d >= output_depth - right_d_padding) {
                        if (params->pad_mode == CSINN_PAD_CONSTANT) {
                            *out_ptr = params->pad_value;
                            out_ptr++;
                        } else if (params->pad_mode = CSINN_PAD_EDGE) {
                            /* TODO */
                            assert(0);
                        } else if (params->pad_mode = CSINN_PAD_REFLECT) {
                            /* TODO */
                            assert(0);
                        }
                    } else {
                        *out_ptr = *in_ptr;
                        out_ptr++;
                        in_ptr++;
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_pad_nchw_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pad_params *params)
{
    const int output_batch = output->dim[0];
    const int output_depth = output->dim[1];
    const int output_height = output->dim[2];
    const int output_width = output->dim[3];

    const int left_b_padding = params->pad_before[0];
    const int left_h_padding = params->pad_before[1];
    const int left_w_padding = params->pad_before[2];
    const int left_d_padding = params->pad_before[3];

    const int right_b_padding = params->pad_after[0];
    const int right_h_padding = params->pad_after[1];
    const int right_w_padding = params->pad_after[2];
    const int right_d_padding = params->pad_after[3];

    const float *in_ptr = input->data;
    float *out_ptr = output->data;

    for (int out_b = 0; out_b < output_batch; ++out_b) {
        for (int out_d = 0; out_d < output_depth; ++out_d) {
            for (int out_h = 0; out_h < output_height; ++out_h) {
                for (int out_w = 0; out_w < output_width; ++out_w) {
                    if (out_b < left_b_padding || out_b >= output_batch - right_b_padding ||
                        out_h < left_h_padding || out_h >= output_height - right_h_padding ||
                        out_w < left_w_padding || out_w >= output_width - right_w_padding ||
                        out_d < left_d_padding || out_d >= output_depth - right_d_padding) {
                        if (params->pad_mode == CSINN_PAD_CONSTANT) {
                            *out_ptr = params->pad_value;
                            out_ptr++;
                        } else if (params->pad_mode = CSINN_PAD_EDGE) {
                            /* TODO */
                            assert(0);
                        } else if (params->pad_mode = CSINN_PAD_REFLECT) {
                            /* TODO */
                            assert(0);
                        }
                    } else {
                        *out_ptr = *in_ptr;
                        out_ptr++;
                        in_ptr++;
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_pad_nchw_u8(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct pad_params *params)
{
    const int output_batch = output->dim[0];
    const int output_depth = output->dim[1];
    const int output_height = output->dim[2];
    const int output_width = output->dim[3];

    const int left_b_padding = params->pad_before[0];
    const int left_h_padding = params->pad_before[1];
    const int left_w_padding = params->pad_before[2];
    const int left_d_padding = params->pad_before[3];

    const int right_b_padding = params->pad_after[0];
    const int right_h_padding = params->pad_after[1];
    const int right_w_padding = params->pad_after[2];
    const int right_d_padding = params->pad_after[3];

    const uint8_t *in_ptr = input->data;
    uint8_t *out_ptr = output->data;
    for (int out_b = 0; out_b < output_batch; ++out_b) {
        for (int out_d = 0; out_d < output_depth; ++out_d) {
            for (int out_h = 0; out_h < output_height; ++out_h) {
                for (int out_w = 0; out_w < output_width; ++out_w) {
                    if (out_b < left_b_padding || out_b >= output_batch - right_b_padding ||
                        out_h < left_h_padding || out_h >= output_height - right_h_padding ||
                        out_w < left_w_padding || out_w >= output_width - right_w_padding ||
                        out_d < left_d_padding || out_d >= output_depth - right_d_padding) {
                        if (params->pad_mode == CSINN_PAD_CONSTANT) {
                            *out_ptr = params->pad_value;
                            out_ptr++;
                        } else if (params->pad_mode = CSINN_PAD_EDGE) {
                            /* TODO */
                            assert(0);
                        } else if (params->pad_mode = CSINN_PAD_REFLECT) {
                            /* TODO */
                            assert(0);
                        }
                    } else {
                        *out_ptr = *in_ptr;
                        out_ptr++;
                        in_ptr++;
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_pad_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct pad_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_pad_nchw_f32(input, output, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_pad_nhwc_f32(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_pad_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct pad_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_pad_nchw_u8(input, output, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_pad_nhwc_u8(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_pad_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct pad_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_PAD, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;     
}

int csi_pad(struct csi_tensor *input,
            struct csi_tensor *output,
            struct pad_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
