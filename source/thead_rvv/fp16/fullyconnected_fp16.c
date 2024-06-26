/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "rvv/rvv.h"

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * n_blk: pack2n/packn/n_tail
 *
 * src: [n, k]
 * dst: [n/n_blk, k, n_blk]
 ************************************************************/
static void reorder_weight_npack2n_fp16(const __fp16 *src, __fp16 *dst, int n, int k)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int pack2n = packn * 2;

    int i = 0;
    int vl = vsetvl_e16m2(pack2n);
    for (; i + pack2n - 1 < n; i += pack2n) {
        const __fp16 *s_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vfloat16m2_t _src = vlse16_v_f16m2(s_ptr, k * sizeof(__fp16), vl);
            vse16_v_f16m2(dst, _src, vl);
            s_ptr += 1;
            dst += vl;
        }
    }
    while (i < n) {
        int vl = vsetvl_e16m1(n - i);
        const __fp16 *s_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vfloat16m1_t _src = vlse16_v_f16m1(s_ptr, k * sizeof(__fp16), vl);
            vse16_v_f16m1(dst, _src, vl);
            s_ptr += 1;
            dst += vl;
        }
        i += vl;
    }
}

void shl_rvv_fc_gemm_reorder_weight_fp16(struct csinn_tensor *weights)
{
    __fp16 *weight_data = (__fp16 *)weights->data;
    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes
    __fp16 *pa_reorder = (__fp16 *)shl_mem_alloc(n * k * sizeof(__fp16));
    reorder_weight_npack2n_fp16(weight_data, pa_reorder, n, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(__fp16));
    shl_mem_free(pa_reorder);
}

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * n_blk: pack2n/packn/n_tail
 *
 * src: [n, k]
 * dst: [n/n_blk, k, n_blk]
 ************************************************************/
static void reorder_weight_npack2n_fp16_w_int8(const int8_t *src, int8_t *dst, int n, int k)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int pack2n = packn * 2;

    int i = 0;
    int vl = vsetvl_e16m2(pack2n);
    for (; i + pack2n - 1 < n; i += pack2n) {
        const int8_t *s_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vint8m2_t _src = vlse8_v_i8m2(s_ptr, k * sizeof(int8_t), vl);
            vse8_v_i8m2(dst, _src, vl);
            s_ptr += 1;
            dst += vl;
        }
    }
    while (i < n) {
        int vl = vsetvl_e16m1(n - i);
        const int8_t *s_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vint8m1_t _src = vlse8_v_i8m1(s_ptr, k * sizeof(int8_t), vl);
            vse8_v_i8m1(dst, _src, vl);
            s_ptr += 1;
            dst += vl;
        }
        i += vl;
    }
}

void shl_rvv_fc_gemm_reorder_weight_fp16_w_int8(struct csinn_tensor *weights)
{
    int8_t *weight_data = (int8_t *)weights->data;
    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes
    int8_t *pa_reorder = (int8_t *)shl_mem_alloc(n * k * sizeof(int8_t));
    reorder_weight_npack2n_fp16_w_int8(weight_data, pa_reorder, n, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(int8_t));
    shl_mem_free(pa_reorder);
}

/*************************************************************************************
 * Per-channel dequantize int8 -> fp16
 ************************************************************************************/
void shl_rvv_fc_npack2n_dequantize_per_channel_i8_to_f16(struct csinn_tensor *weights,
                                                         struct csinn_fc_params *params,
                                                         __fp16 *weights_fp16)
{
    int8_t *weights_int8 = (int8_t *)weights->data;
    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes

    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int pack2n = packn * 2;

    int i = 0;
    int vl = vsetvl_e16m2(pack2n);
    for (; i + pack2n - 1 < n; i += pack2n) {
        int8_t *w_src = weights_int8 + i * k;
        __fp16 *w_dst = weights_fp16 + i * k;
        vint32m4_t _z32 =
            vlse32_v_i32m4(&(weights->qinfo[i].zero_point), sizeof(struct csinn_quant_info), vl);
        vint16m2_t _z16 = vnclip_wx_i16m2(_z32, 0, vl);
        vint8m1_t _z = vnclip_wx_i8m1(_z16, 0, vl);
        vfloat32m4_t _s32 =
            vlse32_v_f32m4(&(weights->qinfo[i].scale), sizeof(struct csinn_quant_info), vl);
        vfloat16m2_t _s = vfncvt_f_f_w_f16m2(_s32, vl);
        for (int j = 0; j < k; j++) {
            vint8m1_t _i8 = vle8_v_i8m1(w_src, vl);
            vfloat16m2_t _f16 = shl_rvv_vdeq_vv_f16m2(_i8, _z, _s, vl);
            vse16_v_f16m2(w_dst, _f16, vl);
            w_src += vl;
            w_dst += vl;
        }
    }
    while (i < n) {
        int vl = vsetvl_e16m1(n - i);
        int8_t *w_src = weights_int8 + i * k;
        __fp16 *w_dst = weights_fp16 + i * k;
        vint32m4_t _z32 =
            vlse32_v_i32m4(&(weights->qinfo[i].zero_point), sizeof(struct csinn_quant_info), vl);
        vint16m2_t _z16 = vnclip_wx_i16m2(_z32, 0, vl);
        vint8m1_t _z = vnclip_wx_i8m1(_z16, 0, vl);
        vfloat32m4_t _s32 =
            vlse32_v_f32m4(&(weights->qinfo[i].scale), sizeof(struct csinn_quant_info), vl);
        vfloat16m2_t _s = vfncvt_f_f_w_f16m2(_s32, vl);
        for (int j = 0; j < k; j++) {
            vint8m1_t _i8 = vle8_v_i8m1(w_src, vl);
            vfloat16m2_t _f16 = shl_rvv_vdeq_vv_f16m2(_i8, _z, _s, vl);
            vse16_v_f16m2(w_dst, _f16, vl);
            w_src += vl;
            w_dst += vl;
        }
        i += vl;
    }
}

int shl_rvv_fullyconnected_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    }

    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = NULL;
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

    int m = batches;
    int n = output_depth;
    int k = accum_depth;

    __fp16 *weights_fp16 = NULL;
    if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
        int size = csinn_tensor_size(weights);
        int8_t *weights_int8 = (int8_t *)weights->data;
        weights_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        if (weights->quant_channel == 1) {
            int32_t zp = weights->qinfo->zero_point;
            float scale = weights->qinfo->scale;
            shl_rvv_dequantize_i8_to_f16(weights_int8, weights_fp16, size, zp, scale);
        } else if (weights->quant_channel == output_depth) {
            shl_rvv_fc_npack2n_dequantize_per_channel_i8_to_f16(weights, params, weights_fp16);
        } else {
            shl_debug_error("%s unsupported quant_channel: %d\n", __func__, weights->quant_channel);
        }
        weights_data = weights_fp16;
    } else if (weights->dtype == CSINN_DTYPE_FLOAT16) {
        weights_data = (__fp16 *)weights->data;
    } else {
        shl_debug_error("weights unsupport dtype: %d\n", weights->dtype);
        return CSINN_FALSE;
    }

    __fp16 *input_reorder = (__fp16 *)shl_mem_alloc(m * k * sizeof(__fp16));
    shl_rvv_reorder_a_block_12xk_fp16(input_data, input_reorder, m, k, m, k);
    shl_rvv_gemm_a0b1_12xpack2n_fp16(output_data, input_reorder, weights_data, bias_data, m, k, n);

    shl_mem_free(input_reorder);
    if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(weights_fp16);
        return CSINN_TRUE;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, weights);
    return CSINN_TRUE;
}
