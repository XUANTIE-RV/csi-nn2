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

/* SHL version 2.1.x */

#include "shl_ref.h"

int shl_ref_transpose_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_transpose_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    if (input->quant_channel == output->quant_channel) {
        int quant_size = input->quant_channel * sizeof(struct csinn_quant_info);
        int t = memcmp(input->qinfo, output->qinfo, quant_size);
        if (t == 0) {
            cb->exec = shl_ref_transpose;
            return CSINN_TRUE;
        }
    }
    cb->exec = shl_ref_transpose_quant;
    return CSINN_TRUE;
}

static void copy_element(struct csinn_tensor *input, struct csinn_tensor *output, int input_idx,
                         int output_idx)
{
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        float *src32 = input->data;
        float *dest32 = output->data;
        dest32[output_idx] = src32[input_idx];
    } else if (input->dtype == CSINN_DTYPE_UINT8 || input->dtype == CSINN_DTYPE_INT8) {
        int8_t *src8 = input->data;
        int8_t *dest8 = output->data;
        dest8[output_idx] = src8[input_idx];
    } else if (input->dtype == CSINN_DTYPE_INT16 || input->dtype == CSINN_DTYPE_FLOAT16) {
        int16_t *src16 = input->data;
        int16_t *dest16 = output->data;
        dest16[output_idx] = src16[input_idx];
    } else {
        shl_debug_error("Transpose unsupport dtype\n");
    }
}

static void swap(int32_t *out_idx, int32_t *in_idx, struct csinn_tensor *input,
                 struct csinn_tensor *output, int32_t *perm, int iter_count)
{
    for (out_idx[iter_count] = 0; out_idx[iter_count] < output->dim[iter_count];
         out_idx[iter_count]++) {
        in_idx[perm[iter_count]] = out_idx[iter_count];
        if (iter_count == 0) {
            int input_idx = shl_ref_get_index_iter(input->dim, input->dim_count - 1, in_idx);
            int output_idx = shl_ref_get_index_iter(output->dim, output->dim_count - 1, out_idx);
            copy_element(input, output, input_idx, output_idx);
        } else {
            swap(out_idx, in_idx, input, output, perm, iter_count - 1);
        }
    }
}

int shl_ref_transpose(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_transpose_params *params)
{
    const int unextended_output_size = output->dim_count;
    int32_t *o = shl_mem_alloc(unextended_output_size * sizeof(int32_t));
    int32_t *i = shl_mem_alloc(unextended_output_size * sizeof(int32_t));
    if (input->dtype != CSINN_DTYPE_FLOAT32 && input->qinfo->scale != output->qinfo->scale &&
        input->qinfo->zero_point != output->qinfo->zero_point) {
        int ret;
        struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
        ret = shl_ref_transpose(finput, foutput, params);
        csinn_tensor_data_convert(output, foutput);
        shl_ref_tensor_transform_free_f32(finput);
        shl_ref_tensor_transform_free_f32(foutput);
    } else {
        swap(o, i, input, output, params->permute, unextended_output_size - 1);
    }
    shl_mem_free(o);
    shl_mem_free(i);
    return CSINN_TRUE;
}

int shl_ref_transpose_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_transpose_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_transpose);
}
