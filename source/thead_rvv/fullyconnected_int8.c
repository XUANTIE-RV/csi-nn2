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
static void shl_rvv_reorder_weight_packn_int8(int8_t *src, int8_t *dst, int m, int k, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(int8_t);  // VLEN128=16  VLEN256=32
    int vl = vsetvl_e8m1(packn);

    while (m > 0) {
        vl = vsetvl_e8m1(m);
        int8_t *in_ptr = src;
        for (int j = 0; j < k; j++) {
            vint8m1_t _input = vlse8_v_i8m1(in_ptr, k * sizeof(int8_t), vl);
            in_ptr++;
            vse8_v_i8m1(dst, _input, vl);
            dst += vl;
        }
        src += vl * k;
        m -= vl;
    }
}

void shl_rvv_fc_gemv_transform_weight_int8(struct csinn_tensor *weights)
{
    int8_t *weight_data = (int8_t *)weights->data;

    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes

    int8_t *pa_reorder = (int8_t *)shl_mem_alloc(n * k * sizeof(int8_t));
    shl_rvv_reorder_weight_packn_int8(weight_data, pa_reorder, n, k, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(int8_t));
    shl_mem_free(pa_reorder);
}

static void shl_rvv_fullyconnectd_packn_int8_internel(const int8_t *input, int32_t *output,
                                                      int8_t *weight, const int32_t *bias,
                                                      int in_nodes, int out_nodes)
{
    const int packn = csrr_vlenb() / sizeof(int8_t);
    int vl = vsetvl_e8m1(packn);

    while (out_nodes > 0) {
        vl = vsetvl_e8m1(out_nodes);
        vint32m4_t _acc = vle32_v_i32m4(bias, vl);
        bias += vl;
        for (int j = 0; j < in_nodes; j++) {
            vint8m1_t _weight = vle8_v_i8m1(weight, vl);
            vint16m2_t _mul = vwmul_vx_i16m2(_weight, input[j], vl);
            _acc = vwmacc_vx_i32m4(_acc, 1, _mul, vl);
            weight += vl;
        }
        vse32_v_i32m4(output, _acc, vl);
        output += vl;
        out_nodes -= vl;
    }
}

int shl_rvv_fullyconnected_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *weights_data = (int8_t *)weights->data;
    int32_t *bias_data = (int32_t *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    const int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    const int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    int32_t *output_tmp = (int32_t *)shl_mem_alloc(output_depth * sizeof(int32_t));
    int vl;

    for (int b = 0; b < batches; b++) {
        int8_t *input_ptr = input_data + b * accum_depth;
        int8_t *weight_ptr = weights_data;
        int32_t *bias_ptr = bias_data;
        int32_t *output_ptr = output_tmp;

        shl_rvv_fullyconnectd_packn_int8_internel(input_ptr, output_ptr, weight_ptr, bias_ptr,
                                                  accum_depth, output_depth);

        if (weights->quant_channel == 1) {
            shl_rvv_requantize(output_ptr, weights->qinfo->multiplier, weights->qinfo->shift,
                               output_depth);
        } else if (weights->quant_channel == output_depth) {
            // support channel quantization
            for (int c = 0; c < weights->quant_channel; c++) {
                shl_rvv_requantize(output_ptr + c, weights->qinfo[c].multiplier,
                                   weights->qinfo[c].shift, 1);
            }
        }
        shl_rvv_saturated_int8(output_ptr, output_data + b * output_depth,
                               output->qinfo->zero_point, output_depth);
    }
    if (output_tmp) {
        shl_mem_free(output_tmp);
        output_tmp = NULL;
    }
    return CSINN_TRUE;
}

/************************************ dot **********************************************/
#ifdef XTHEADV
static void shl_rvv_reorder_weight_packn_int8_dot(int8_t *src, int8_t *dst, int m, int k, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(int8_t);
    int vl = vsetvl_e8m1(packn);

    while (m > 0) {
        vl = vsetvl_e8m1(m);
        int32_t *in_ptr0 = (int32_t *)src;
        int32_t *out_ptr0 = (int32_t *)dst;
        int j = 0;
        for (; j + 7 < k; j += 8) {
            vint32m4_t _nf0, _nf1;
            vlsseg2e32_v_i32m4(&_nf0, &_nf1, in_ptr0, k * sizeof(int8_t), vl);
            in_ptr0 += 2;
            vse32_v_i32m4(out_ptr0, _nf0, vl);
            out_ptr0 += vl;
            vse32_v_i32m4(out_ptr0, _nf1, vl);
            out_ptr0 += vl;
        }
        for (; j + 3 < k; j += 4) {
            vint32m4_t _input = vlse32_v_i32m4(in_ptr0, k * sizeof(int8_t), vl);
            in_ptr0++;
            vse32_v_i32m4(out_ptr0, _input, vl);
            out_ptr0 += vl;
        }
        src += vl * k;
        dst += vl * k;
        m -= vl;
    }
}

void shl_rvv_fc_gemv_transform_weight_int8_dot(struct csinn_tensor *weights)
{
    int8_t *weight_data = (int8_t *)weights->data;

    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes

    int8_t *pa_reorder = (int8_t *)shl_mem_alloc(n * k * sizeof(int8_t));
    shl_rvv_reorder_weight_packn_int8_dot(weight_data, pa_reorder, n, k, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(int8_t));
    shl_mem_free(pa_reorder);
}

static void shl_rvv_fullyconnectd_packn_int8_internel_dot(const int8_t *input, int32_t *output,
                                                          int8_t *weight, const int32_t *bias,
                                                          int in_nodes, int out_nodes)
{
    const int packn = csrr_vlenb() / sizeof(int8_t);
    int vl = vsetvl_e8m1(packn);

    while (out_nodes > 0) {
        vl = vsetvl_e8m1(out_nodes);
        int32_t *input_ptr = (int32_t *)input;
        vint32m4_t _acc0 = vle32_v_i32m4(bias, vl);
        bias += vl;
        for (int c = 0; c < in_nodes / 4; c++) {
            vint8m4_t _weight = vle8_v_i8m4(weight, vl * 4);
            _acc0 = vmaqa_vx_i32m4(_acc0, input_ptr[c], _weight, vl);
            weight += 4 * vl;
        }
        vse32_v_i32m4(output, _acc0, vl);
        output += vl;
        out_nodes -= vl;
    }
}

int shl_rvv_fullyconnected_packn_int8_dot(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *weights, struct csinn_tensor *bias,
                                          struct csinn_fc_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *weights_data = (int8_t *)weights->data;
    int32_t *bias_data = (int32_t *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    const int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    const int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    int32_t *output_tmp = (int32_t *)shl_mem_alloc(output_depth * sizeof(int32_t));
    int vl;

    for (int b = 0; b < batches; b++) {
        int8_t *input_ptr = input_data + b * accum_depth;
        int8_t *weight_ptr = weights_data;
        int32_t *bias_ptr = bias_data;
        int32_t *output_ptr = output_tmp;

        shl_rvv_fullyconnectd_packn_int8_internel_dot(input_ptr, output_ptr, weight_ptr, bias_ptr,
                                                      accum_depth, output_depth);

        if (weights->quant_channel == 1) {
            shl_rvv_requantize(output_ptr, weights->qinfo->multiplier, weights->qinfo->shift,
                               output_depth);
        } else if (weights->quant_channel == output_depth) {
            // support channel quantization
            for (int c = 0; c < weights->quant_channel; c++) {
                shl_rvv_requantize(output_ptr + c, weights->qinfo[c].multiplier,
                                   weights->qinfo[c].shift, 1);
            }
        }
        shl_rvv_saturated_int8(output_ptr, output_data + b * output_depth,
                               output->qinfo->zero_point, output_depth);
    }
    if (output_tmp) {
        shl_mem_free(output_tmp);
        output_tmp = NULL;
    }
    return CSINN_TRUE;
}
#endif
