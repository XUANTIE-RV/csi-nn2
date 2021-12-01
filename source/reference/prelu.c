/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

#include "csi_ref.h"
#include "csi_utils.h"

static int csi_ref_prelu_nhwc_f32(struct csi_tensor *input,
                                  struct csi_tensor *alpha,
                                  struct csi_tensor *output,
                                  struct prelu_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *alpha_data = alpha->data;
    for (int b = 0; b < output->dim[0]; ++b) {
        for (int y = 0; y < output->dim[1]; ++y) {
            for (int x = 0; x < output->dim[2]; ++x) {
                for (int c = 0; c < output->dim[3]; ++c) {
                    int output_index = csi_ref_get_index(output->dim, b, y, x, c);
                    int input_index = csi_ref_get_index(input->dim, b, y, x, c);
                    float input_value = input_data[input_index];
                    if (input_value >= 0) {
                        output_data[output_index] = input_data[input_index];
                    } else {
                        output_data[output_index] = input_value * alpha_data[c];
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_ref_prelu_nchw_f32(struct csi_tensor *input,
                                  struct csi_tensor *alpha,
                                  struct csi_tensor *output,
                                  struct prelu_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *alpha_data = alpha->data;
    for (int b = 0; b < output->dim[0]; ++b) {
        for (int y = 0; y < output->dim[2]; ++y) {
            for (int x = 0; x < output->dim[3]; ++x) {
                for (int c = 0; c < output->dim[1]; ++c) {
                    int output_index = csi_ref_get_index(output->dim, b, c, y, x);
                    int input_index = csi_ref_get_index(input->dim, b, c, y, x);
                    float input_value = input_data[input_index];
                    if (input_value >= 0) {
                        output_data[output_index] = input_data[input_index];
                    } else {
                        output_data[output_index] = input_value * alpha_data[c];
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_ref_prelu_f32(struct csi_tensor *input,
                      struct csi_tensor *alpha,
                      struct csi_tensor *output,
                      struct prelu_params *params)
{
    if (params->base.layout == CSINN_NCHW) {
        csi_ref_prelu_nchw_f32(input, alpha, output, params);
    } else if (params->base.layout == CSINN_NHWC) {
        csi_ref_prelu_nhwc_f32(input, alpha, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_ref_prelu_quant(struct csi_tensor *input,
                        struct csi_tensor *alpha,
                        struct csi_tensor *output,
                        struct prelu_params *params)
{
    return csi_ref_diso_callback_base(input, alpha, output, params, csi_ref_prelu_f32);
}		
