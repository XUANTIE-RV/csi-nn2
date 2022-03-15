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
static void csi_nn_rvv_reorder_weight_npackn_fp32(float *src, float *dst, int m, int k, int ldx)
{
    int packn = csrr_vlenb() / sizeof(float);  // VLEN128=4  VLEN256=8
    int vl = vsetvl_e32m1(packn);
    int i = 0;
    for (; i + packn - 1 < m; i += packn) {
        float *in_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vfloat32m1_t _input = vlse32_v_f32m1(in_ptr, k * sizeof(float), vl);
            in_ptr++;
            vse32_v_f32m1(dst, _input, vl);
            dst += packn;
        }
    }
    src += i * k;
    for (; i < m; i++) {
        memcpy(dst, src, sizeof(float) * k);
        dst += k;
        src += k;
    }
}

void csi_nn_rvv_fc_gemv_transform_weight_fp32(struct csi_tensor *weights)
{
    float *weight_data = (float *)weights->data;

    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes

    float *pa_reorder = (float *)csi_mem_alloc(n * k * sizeof(float));
    csi_nn_rvv_reorder_weight_npackn_fp32(weight_data, pa_reorder, n, k, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(float));
    csi_mem_free(pa_reorder);
}

int csi_nn_rvv_fullyconnected_packn_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                         struct csi_tensor *weights, struct csi_tensor *bias,
                                         struct fc_params *params)
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
        bias_data = (float *)csi_mem_alloc(output_depth * 2);
    }
    int packn = csrr_vlenb() / sizeof(float);  // VLEN128=4  VLEN256=8
    int vl;

    for (int b = 0; b < batches; b++) {
        float *init_output = output_data + b * output_depth;
        float *init_input = input_data + b * accum_depth;
        float *init_weight = weights_data;
        float *init_bias = bias_data;

        vl = vsetvl_e32m1(packn);
        int n = 0;
        for (; n + packn - 1 < output_depth; n += packn) {
            float *in_ptr = init_input;
            vfloat32m1_t _acc = vle32_v_f32m1(init_bias, vl);
            init_bias += vl;

            for (int k = 0; k < accum_depth; k++) {
                vfloat32m1_t _weight = vle32_v_f32m1(init_weight, vl);
                _acc = vfmacc_vf_f32m1(_acc, in_ptr[k], _weight, vl);
                init_weight += vl;
            }
            vse32_v_f32m1(init_output, _acc, vl);
            init_output += vl;
        }
        for (; n < output_depth; n++) {
            float *in_ptr = init_input;
            float acc = init_bias[0];
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

int csi_nn_rvv_fullyconnected_init(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *weights, struct csi_tensor *bias,
                                   struct fc_params *params)
{
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        csi_nn_rvv_fc_gemv_transform_weight_fp32(weights);
        params->base.bc = csi_nn_rvv_fullyconnected_packn_fp32;
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        csi_nn_rvv_fc_gemv_transform_weight_fp16(weights);
        params->base.bc = csi_nn_rvv_fullyconnected_packn_fp16;
    } else if (input->dtype == CSINN_DTYPE_INT8) {
        csi_nn_rvv_fc_gemv_transform_weight_int8(weights);
        // support channel quantization
        for (int i = 0; i < weights->quant_channel; i++) {
            float real_scale = input->qinfo->scale * weights->qinfo[i].scale / output->qinfo->scale;
            csi_quantize_multiplier(real_scale, &(weights->qinfo[i].multiplier),
                                    &(weights->qinfo[i].shift));
        }
        params->base.bc = csi_nn_rvv_fullyconnected_packn_int8;
    }
    return CSINN_TRUE;
}
