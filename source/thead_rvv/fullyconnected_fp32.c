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

#include "shl_thead_rvv.h"

/*************************************************************
    note: VLEN = 128/256
*************************************************************/
static void shl_rvv_reorder_weight_npackn_fp32(float *src, float *dst, int m, int k, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(float);  // VLEN128=4  VLEN256=8
    int vl = vsetvl_e32m1(packn);

    while (m > 0) {
        vl = vsetvl_e32m1(m);
        float *in_ptr = src;
        for (int j = 0; j < k; j++) {
            vfloat32m1_t _input = vlse32_v_f32m1(in_ptr, k * sizeof(float), vl);
            in_ptr++;
            vse32_v_f32m1(dst, _input, vl);
            dst += vl;
        }
        src += vl * k;
        m -= vl;
    }
}

void shl_rvv_fc_gemv_transform_weight_fp32(struct csinn_tensor *weights)
{
    float *weight_data = (float *)weights->data;

    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes

    float *pa_reorder = (float *)shl_mem_alloc(n * k * sizeof(float));
    shl_rvv_reorder_weight_npackn_fp32(weight_data, pa_reorder, n, k, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(float));
    shl_mem_free(pa_reorder);
}

int shl_rvv_fullyconnected_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *weights_data = (float *)weights->data;
    float *bias_data = (float *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    bool flag_bias = 1;  // default: fc layer include bias
    if (bias_data == NULL) {
        flag_bias = 0;
        bias_data = (float *)shl_mem_alloc(output_depth * 2);
    }
    const int packn = csrr_vlenb() / sizeof(float);  // VLEN128=4  VLEN256=8
    int vl = vsetvl_e32m1(packn);

    for (int b = 0; b < batches; b++) {
        float *init_output = output_data + b * output_depth;
        float *init_input = input_data + b * accum_depth;
        float *init_weight = weights_data;
        float *init_bias = bias_data;

        int n = output_depth;
        while (n > 0) {
            vl = vsetvl_e32m1(n);
            vfloat32m1_t _acc = vle32_v_f32m1(init_bias, vl);
            init_bias += vl;
            for (int k = 0; k < accum_depth; k++) {
                vfloat32m1_t _weight = vle32_v_f32m1(init_weight, vl);
                _acc = vfmacc_vf_f32m1(_acc, init_input[k], _weight, vl);
                init_weight += vl;
            }
            vse32_v_f32m1(init_output, _acc, vl);
            init_output += vl;
            n -= vl;
        }
    }
    if (!flag_bias) {
        shl_mem_free(bias_data);
        bias_data = NULL;
    }
    return CSINN_TRUE;
}
