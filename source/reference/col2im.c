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

/* CSI-NN2 version 1.8.x */

#include "csi_ref.h"

int csi_ref_col2im_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct csi_tensor *kernel,
                       struct col2im_params *params)
{
    int32_t height = input->dim[1];
    int32_t width = input->dim[2];
    int32_t depth = input->dim[3];
    int32_t filter_h = kernel->dim[1];
    int32_t filter_w = kernel->dim[2];
    int32_t pad_t = params->pad_h;
    int32_t pad_b = params->pad_h;
    int32_t pad_l = params->pad_w;
    int32_t pad_r = params->pad_w;
    float *col_data = input->data;
    int height_col = (height + pad_t + pad_b - filter_h) / params->stride_h + 1;
    int width_col = (width + pad_l + pad_r - filter_w) / params->stride_w + 1;
    int h_pad = -pad_t;
    for (int h = 0; h < height_col; ++h) {
        int w_pad = -pad_l;
        for (int w = 0; w < width_col; ++w) {
            float *im_patch_data = output->data + (h_pad * width + w_pad) * depth;
            for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
                for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        // TODO(andydavis) Vectorize this loop (if compiler does not).
                        for (int i = 0; i < depth; ++i) {
                            im_patch_data[i] += col_data[i];
                        }
                    }
                    im_patch_data += depth;
                    col_data += depth;
                }
                // Jump over remaining number of depth.
                im_patch_data += depth * (width - filter_w);
            }
            w_pad += params->stride_w;
        }
        h_pad += params->stride_h;
    }
    return CSINN_TRUE;
}
