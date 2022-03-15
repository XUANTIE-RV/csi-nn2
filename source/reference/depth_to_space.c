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

/* CSI-NN2 version 1.12.x */

#include "csi_ref.h"

// the input->data is a 4-D Tensor with shape [batch, depth, height, width].
int csi_ref_depth_to_space_nchw_f32(struct csi_tensor *input, struct csi_tensor *output,
                                    struct depth_to_space_params *params)
{
    if (params->mode == CSINN_DEPTHTOSPACE_CRD) return CSINN_FALSE;
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch = input->dim[0];
    int in_channel = input->dim[1];
    int in_height = input->dim[2];
    int in_width = input->dim[3];

    int block_size = params->block_size;
    int block_size2 = block_size * block_size;
    assert(in_channel % block_size2 == 0);

    int out_channel = output->dim[1];  // out_channel = in_channel/block_size2;
    int out_height = output->dim[2];   // out_weight = in_weight*block_size;
    int out_width = output->dim[3];    // out_width = in_width*block_size;

    for (int out_b = 0; out_b < batch; ++out_b) {
        for (int in_h = 0; in_h < in_height; ++in_h) {
            for (int in_w = 0; in_w < in_width; ++in_w) {
                for (int out_c = 0; out_c < out_channel; ++out_c) {
                    float *temp = (float *)csi_mem_alloc(block_size2 * sizeof(float));
                    int in_start_addr = csi_ref_get_index(input->dim, out_b, out_c, in_h, in_w);
                    for (int i = 0; i < block_size2; i++) {
                        temp[i] =
                            input_data[in_start_addr + i * out_channel * in_height * in_width];
                    }
                    int out_start_addr = csi_ref_get_index(output->dim, out_b, out_c,
                                                           in_h * block_size, in_w * block_size);
                    for (int h = 0; h < block_size; h++) {
                        for (int w = 0; w < block_size; w++) {
                            output_data[out_start_addr + h * out_width + w] =
                                temp[h * block_size + w];
                        }
                    }
                    csi_mem_free(temp);
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_ref_depth_to_space_nhwc_f32(struct csi_tensor *input, struct csi_tensor *output,
                                    struct depth_to_space_params *params)
{
    struct csi_tensor *t_input = csi_alloc_tensor(NULL);
    csi_tensor_copy(t_input, input);
    t_input->layout = CSINN_LAYOUT_NCHW;
    t_input->data = malloc(csi_tensor_size(input) * sizeof(float));
    t_input->dim[1] = input->dim[3];
    t_input->dim[2] = input->dim[1];
    t_input->dim[3] = input->dim[2];
    struct transpose_params pparams;
    pparams.permute_num = 4;
    pparams.base.layout = CSINN_LAYOUT_NCHW;
    pparams.base.api = CSINN_REF;
    pparams.base.run_mode = CSINN_RM_LAYER;
    pparams.base.name = params->base.name;
    pparams.permute = malloc(pparams.permute_num * sizeof(int32_t));
    pparams.permute[0] = 0;
    pparams.permute[1] = 3;
    pparams.permute[2] = 1;
    pparams.permute[3] = 2;
    csi_ref_transpose(input, t_input, &pparams);

    struct csi_tensor *t_output = csi_alloc_tensor(NULL);
    csi_tensor_copy(t_output, output);
    t_output->layout = CSINN_LAYOUT_NCHW;
    t_output->data = malloc(csi_tensor_size(output) * sizeof(float));
    t_output->dim[1] = output->dim[3];
    t_output->dim[2] = output->dim[1];
    t_output->dim[3] = output->dim[2];

    csi_ref_depth_to_space_nchw_f32(t_input, t_output, params);
    pparams.permute[0] = 0;
    pparams.permute[1] = 2;
    pparams.permute[2] = 3;
    pparams.permute[3] = 1;

    csi_ref_transpose(t_output, output, &pparams);

    csi_free_tensor(t_input);
    csi_free_tensor(t_output);
    free(pparams.permute);
    return CSINN_TRUE;
}

int csi_ref_depth_to_space_f32(struct csi_tensor *input, struct csi_tensor *output,
                               struct depth_to_space_params *params)
{
    if (input->layout == CSINN_LAYOUT_NCHW) {
        return csi_ref_depth_to_space_nchw_f32(input, output, params);
    } else if (input->layout == CSINN_LAYOUT_NHWC) {
        return csi_ref_depth_to_space_nhwc_f32(input, output, params);
    }
    return CSINN_FALSE;
}

int csi_ref_depth_to_space_quant(struct csi_tensor *input, struct csi_tensor *output,
                                 struct depth_to_space_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_depth_to_space_f32);
}
