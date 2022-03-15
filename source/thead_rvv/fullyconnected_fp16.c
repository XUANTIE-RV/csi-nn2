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

#include "csi_thead_rvv.h"

/*************************************************************
    note: VLEN = 128/256
*************************************************************/
static void csi_nn_rvv_reorder_weight_npackn_fp16(__fp16 *src, __fp16 *dst, int m, int k, int ldx)
{
    int packn = csrr_vlenb() / sizeof(__fp16);  // VLEN128=8  VLEN256=16
    int vl = vsetvl_e16m1(packn);
    int i = 0;
    for (; i + packn - 1 < m; i += packn) {
        __fp16 *in_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vfloat16m1_t _input = vlse16_v_f16m1(in_ptr, k * sizeof(__fp16), vl);
            in_ptr++;
            vse16_v_f16m1(dst, _input, vl);
            dst += packn;
        }
    }
    src += i * k;
    for (; i < m; i++) {
        memcpy(dst, src, sizeof(__fp16) * k);
        dst += k;
        src += k;
    }
}

void csi_nn_rvv_fc_gemv_transform_weight_fp16(struct csi_tensor *weights)
{
    __fp16 *weight_data = (__fp16 *)weights->data;

    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes

    __fp16 *pa_reorder = (__fp16 *)csi_mem_alloc(n * k * sizeof(__fp16));
    csi_nn_rvv_reorder_weight_npackn_fp16(weight_data, pa_reorder, n, k, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(__fp16));
    csi_mem_free(pa_reorder);
}

int csi_nn_rvv_fullyconnected_packn_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                         struct csi_tensor *weights, struct csi_tensor *bias,
                                         struct fc_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = (__fp16 *)weights->data;
    __fp16 *bias_data = (__fp16 *)bias->data;
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
        bias_data = (__fp16 *)csi_mem_alloc(output_depth * 2);
    }

    int packn = csrr_vlenb() / sizeof(__fp16);  // VLEN128=8  VLEN256=16
    int vl;

    for (int b = 0; b < batches; b++) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;

        vl = vsetvl_e16m1(packn);
        int n = 0;
        for (; n + packn - 1 < output_depth; n += packn) {
            __fp16 *in_ptr = init_input;
            vfloat16m1_t _acc = vle16_v_f16m1(init_bias, vl);
            init_bias += vl;

            for (int k = 0; k < accum_depth; k++) {
                vfloat16m1_t _weight = vle16_v_f16m1(init_weight, vl);
                _acc = vfmacc_vf_f16m1(_acc, in_ptr[k], _weight, vl);
                init_weight += vl;
            }
            vse16_v_f16m1(init_output, _acc, vl);
            init_output += vl;
        }
        for (; n < output_depth; n++) {
            __fp16 *in_ptr = init_input;
            __fp16 acc = init_bias[0];
            for (int k = 0; k < accum_depth; k++) {
                acc += in_ptr[k] * init_weight[k];
            }
            *init_output++ = acc;
            init_bias++;
            init_weight += accum_depth;
        }
    }
    if (!flag_bias) {
        csi_mem_free(bias_data);
        bias_data = NULL;
    }
    return CSINN_TRUE;
}
