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

int csi_ref_scatter_nd_f32(struct csi_tensor *input, struct csi_tensor *indices,
                           struct csi_tensor *updates, struct csi_tensor *output,
                           struct scatter_nd_params *params)
{
  if (input->dim_count != 5 && indices->dim[indices->dim_count - 1] != 5) {
    return CSINN_FALSE;
  }
  float* input_data = (float*)input->data;
  int32_t* indices_data = (int32_t*)indices->data;
  float* updates_data = (float*)updates->data;
  float* output_data = (float*)output->data;

  int size = 1;
  for (int i = 0; i < input->dim_count; i++) {
    size = size * input->dim[i];
  }
  for (int i = 0; i < size; i++) {
    output_data[i] = input_data[i];
  }

  for (int i = 0; i < indices->dim[0]; i++) {
    for (int j = 0; j < indices->dim[1]; j++) {
      for (int k = 0; k < indices->dim[2]; k++) {
        for (int l = 0; l < indices->dim[3]; l++) {
          for (int m = 0; m < indices->dim[4]; m++) {
            int indices_base =
                ((((i * indices->dim[1] + j) * indices->dim[2] + k) * indices->dim[3] + l) *
                     indices->dim[4] + m) * indices->dim[5];

            int output_index =
                csi_ref_get_index_5(input->dim, indices_data[indices_base],
                                indices_data[indices_base + 1], indices_data[indices_base + 2],
                                indices_data[indices_base + 3], indices_data[indices_base + 4]);

            int updates_index = csi_ref_get_index_5(updates->dim, i, j, k, l, m);
            output_data[output_index] = updates_data[updates_index];
          }
        }
      }
    }
  }

  return CSINN_TRUE;
}

int csi_ref_scatter_nd_quant(struct csi_tensor *input, struct csi_tensor *indices,
                             struct csi_tensor *updates, struct csi_tensor *output,
                             struct scatter_nd_params *params)
{
    struct csi_tensor *float_input = csi_ref_tensor_transform_f32(input);
    struct csi_tensor *float_updates = csi_ref_tensor_transform_f32(updates);
    struct csi_tensor *float_output = csi_ref_tensor_transform_f32(output);
    int ret = csi_ref_scatter_nd_f32(float_input, indices, float_updates, float_output, params);
    csi_tensor_data_convert(output, float_output);
    csi_ref_tensor_transform_free_f32(float_input);
    csi_ref_tensor_transform_free_f32(float_output);
    csi_ref_tensor_transform_free_f32(float_updates);
    return ret;
}
