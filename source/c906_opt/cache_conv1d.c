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

#include "csi_c906.h"

int csi_c906_cache_conv1d_init(struct csi_tensor *input, struct csi_tensor *output,
                               struct csi_tensor *weight, struct csi_tensor *bias,
                               struct cache_conv1d_params *params)
{
    size_t data_size =
        output->dim[0] * output->dim[1] * output->dim[2] * sizeof(__fp16);  // 512*13*2
    asr_buffer_init_c906(&params->asr_buffer, 2 * data_size, data_size);

    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        __fp16 *weight_data = (__fp16 *)weight->data;

        int n = weight->dim[0];  // out_nodes
        int k = weight->dim[1];  // in_nodes
        if (k % 16 != 0) {
            csi_debug_error("out_nodes num should be multiple of 16\n");
        }
        __fp16 *pa_reorder = (__fp16 *)csi_mem_alloc(n * k * sizeof(__fp16));
        csi_c906_reorder_weight_n16_fp16(weight_data, pa_reorder, n, k, k);

        csi_c906_memcpy(weight_data, pa_reorder, n * k * sizeof(__fp16));
        params->data = weight_data;
        csi_mem_free(pa_reorder);

        params->base.bc = csi_c906_cache_conv1d_fp16;
    }

    return CSINN_TRUE;
}

int csi_c906_cache_conv1d_fp16(struct csi_tensor *input, struct csi_tensor *output,
                               struct csi_tensor *weight, struct csi_tensor *bias,
                               struct cache_conv1d_params *params)
{
    __fp16 *input_data = input->data;
    __fp16 *output_data = output->data;
    __fp16 *weights_data = weight->data;
    __fp16 *bias_data = bias->data;
    const int weights_dims_count = weight->dim_count;
    const int output_depth = weight->dim[weights_dims_count - 3];
    const int accum_depth = weight->dim[weights_dims_count - 2];
    const int batches = input->dim[1];

    int packn = 16;
    int vl = 16;
    int b = 0;
    for (; b + 3 < batches; b += 4) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_output2 = init_output + output_depth;
        __fp16 *init_output3 = init_output2 + output_depth;
        __fp16 *init_output4 = init_output3 + output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_input2 = init_input + accum_depth;
        __fp16 *init_input3 = init_input2 + accum_depth;
        __fp16 *init_input4 = init_input3 + accum_depth;

        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;
        int n = output_depth;
        while (n > 0) {
            __fp16 *in_ptr = init_input;
            __fp16 *in_ptr2 = init_input2;
            __fp16 *in_ptr3 = init_input3;
            __fp16 *in_ptr4 = init_input4;

            vfloat16m2_t _acc = vle16_v_f16m2(init_bias, vl);
            vfloat16m2_t _acc2 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc3 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc4 = vmv_v_v_f16m2(_acc, vl);

            init_bias += vl;
            int k = accum_depth;
            while (k > 0) {
                vfloat16m2_t _weight = vle16_v_f16m2(init_weight, vl);
                _acc = vfmacc_vf_f16m2(_acc, *in_ptr, _weight, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, *in_ptr2, _weight, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, *in_ptr3, _weight, vl);
                _acc4 = vfmacc_vf_f16m2(_acc4, *in_ptr4, _weight, vl);
                init_weight += vl;
                in_ptr++;
                in_ptr2++;
                in_ptr3++;
                in_ptr4++;
                k--;
            }
            vse16_v_f16m2(init_output, _acc, vl);
            vse16_v_f16m2(init_output2, _acc2, vl);
            vse16_v_f16m2(init_output3, _acc3, vl);
            vse16_v_f16m2(init_output4, _acc4, vl);
            init_output += vl;
            init_output2 += vl;
            init_output3 += vl;
            init_output4 += vl;
            n -= vl;
        }
    }
    for (; b + 1 < batches; b += 2) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_output2 = init_output + output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_input2 = init_input + accum_depth;

        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;
        int n = output_depth;
        while (n > 0) {
            __fp16 *in_ptr = init_input;
            __fp16 *in_ptr2 = init_input2;
            vfloat16m2_t _acc = vle16_v_f16m2(init_bias, vl);
            vfloat16m2_t _acc2 = vmv_v_v_f16m2(_acc, vl);
            init_bias += vl;
            int k = accum_depth;
            while (k > 0) {
                vfloat16m2_t _weight = vle16_v_f16m2(init_weight, vl);
                _acc = vfmacc_vf_f16m2(_acc, *in_ptr, _weight, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, *in_ptr2, _weight, vl);
                init_weight += vl;
                in_ptr++;
                in_ptr2++;
                k--;
            }
            vse16_v_f16m2(init_output, _acc, vl);
            vse16_v_f16m2(init_output2, _acc2, vl);
            init_output += vl;
            init_output2 += vl;
            n -= vl;
        }
    }
    for (; b < batches; b++) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;

        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;
        int n = output_depth;
        while (n > 0) {
            __fp16 *in_ptr = init_input;
            vfloat16m2_t _acc = vle16_v_f16m2(init_bias, vl);
            init_bias += vl;
            int k = accum_depth;
            while (k > 0) {
                vfloat16m2_t _weight = vle16_v_f16m2(init_weight, vl);
                _acc = vfmacc_vf_f16m2(_acc, *in_ptr, _weight, vl);
                init_weight += vl;
                in_ptr++;
                k--;
            }
            vse16_v_f16m2(init_output, _acc, vl);
            init_output += vl;
            n -= vl;
        }
    }

    size_t insert_lenth = output->dim[1] * input->dim[1];  // 512*6
    __fp16 *output_from_buffer;
    output_from_buffer = asr_buffer_insert_c906_back(&params->asr_buffer, output_data,
                                                     insert_lenth * sizeof(__fp16));
    size_t output_lenth = output->dim[0] * output->dim[1] * output->dim[2];
    int *shape = output->dim;

    __fp16 *p_input = output_from_buffer;
    __fp16 *p_output = output->data;
    for (int i = 0; i < shape[2]; i++) {
        int j = 0;
        for (; j + 15 < shape[1]; j += 16) {
            int out_pos = j * shape[2] + i;
            vfloat16m2_t _output_from_buffer;
            _output_from_buffer = vle16_v_f16m2(p_input + i * shape[1] + j, 16);
            vsse16_v_f16m2(p_output + out_pos, 2 * shape[2], _output_from_buffer, 16);
        }
        if (j != shape[1]) {
            int vl = shape[1] - j;
            int out_pos = j * shape[2] + i;
            vfloat16m2_t _output_from_buffer;
            _output_from_buffer = vle16_v_f16m2(p_input + i * shape[1] + j, vl);
            vsse16_v_f16m2(p_output + out_pos, 2 * shape[2], _output_from_buffer, vl);
        }
    }
}