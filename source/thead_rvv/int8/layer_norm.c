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

#include "rvv/rvv.h"

/*************************************************************
    note: support flexible vlen
*************************************************************/

// FIXME: precision loss
int layer_norm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *gamma, struct csinn_tensor *beta,
                    struct csinn_layer_norm_params *params)
{
    if (params->center == false || params->scale == false) {
        shl_debug_error("Layer norm only support center & scale == true\n");
        return CSINN_FALSE;
    }
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
    }
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *gamma_data = (int8_t *)gamma->data;
    int8_t *beta_data = (int8_t *)beta->data;

    /* support negative axis */
    int axis = params->axis >= 0 ? params->axis : (params->axis + input->dim_count);

    int32_t batches = 1;
    for (int i = 0; i < axis; i++) {
        batches *= input->dim[i];
    }
    int32_t norm_size = 1;
    for (int i = axis; i < input->dim_count; i++) {
        norm_size *= input->dim[i];
    }
    int32_t z1 = input->qinfo->zero_point;
    int32_t z3 = output->qinfo->zero_point;
    float s1 = input->qinfo->scale;
    float s2 = gamma->qinfo->scale;
    float s3 = output->qinfo->scale;
    int32_t multiplier;
    int32_t shift;
    float real_scale = s1 * s2 / s3;
    shl_quantize_multiplier(real_scale, &multiplier, &shift);

    int32_t *qbeta = (int32_t *)shl_mem_alloc(norm_size * sizeof(int32_t));
    int32_t *qb = qbeta;
    int size = norm_size;
    while (size > 0) {
        int vl = vsetvl_e8m1(size);
        vint8m1_t _beta = vle8_v_i8m1(beta_data, vl);
        vint16m2_t _b_w = vwadd_vx_i16m2(_beta, 0, vl);
        vint32m4_t _b_ww = vwadd_vx_i32m4(_b_w, 0, vl);
        vint32m4_t _mulh = vmulh_vx_i32m4(_b_ww, multiplier, vl);
        if (shift < 0) {
            _mulh = vssra_vx_i32m4(_mulh, -shift - 1, vl);
        } else {
            _mulh = vsll_vx_i32m4(_mulh, shift + 1, vl);
        }
        _mulh = vadd_vx_i32m4(_mulh, z3, vl);
        vse32_v_i32m4(qb, _mulh, vl);
        beta_data += vl;
        qb += vl;
        size -= vl;
    }

    for (int b = 0; b < batches; b++) {
        int8_t *input_ptr = input_data + b * norm_size;
        int8_t *output_ptr = output_data + b * norm_size;
        int16_t *tmp = (int16_t *)shl_mem_alloc(norm_size * sizeof(int16_t));

        vint32m1_t _sum1 = vmv_v_x_i32m1(0, 1);
        int8_t *in0 = input_ptr;
        size = norm_size;
        while (size > 0) {
            int vl = vsetvl_e8m1(size);
            vint8m1_t _in = vle8_v_i8m1(in0, vl);
            in0 += vl;
            vint16m2_t _in_w = vwadd_vx_i16m2(_in, 0, vl);
            vint32m4_t _in_ww = vwadd_vx_i32m4(_in_w, 0, vl);
            _sum1 = vredsum_vs_i32m4_i32m1(vundefined_i32m1(), _in_ww, _sum1, vl);
            size -= vl;
        }
        int32_t sum1 = vmv_x_s_i32m1_i32(_sum1);
        int8_t mean = sum1 / norm_size;

        vint32m1_t _sum2 = vmv_v_x_i32m1(0, 1);
        int16_t *t0 = tmp;
        in0 = input_ptr;
        size = norm_size;
        while (size > 0) {
            int vl = vsetvl_e8m1(size);
            vint8m1_t _in = vle8_v_i8m1(in0, vl);
            in0 += vl;
            vint16m2_t _in_w = vwadd_vx_i16m2(_in, 0, vl);
            vint16m2_t _tmp = vsub_vx_i16m2(_in_w, mean, vl);
            vse16_v_i16m2(t0, _tmp, vl);
            t0 += vl;
            vint32m4_t _mul = vwmul_vv_i32m4(_tmp, _tmp, vl);
            _sum2 = vredsum_vs_i32m4_i32m1(vundefined_i32m1(), _mul, _sum2, vl);
            size -= vl;
        }
        int32_t sum2 = vmv_x_s_i32m1_i32(_sum2);
        __fp16 var = s1 * s1 * sum2 / norm_size;
        __fp16 std = sqrt(var + params->epsilon);

        int32_t multiplier2;
        int32_t shift2;
        float real_scale2 = real_scale / std;
        shl_quantize_multiplier(real_scale2, &multiplier2, &shift2);

        int8_t *g0 = gamma_data;
        t0 = tmp;
        qb = qbeta;
        size = norm_size;
        while (size > 0) {
            int vl = vsetvl_e32m1(size);
            vint16m2_t _tmp = vle16_v_i16m2(t0, vl);
            vint32m4_t _tmp_w = vwadd_vx_i32m4(_tmp, 0, vl);
            t0 += vl;
            vint8m1_t _gamma = vle8_v_i8m1(g0, vl);
            vint16m2_t _g_w = vwadd_vx_i16m2(_gamma, 0, vl);
            vint32m4_t _g_ww = vwadd_vx_i32m4(_g_w, 0, vl);
            g0 += vl;
            vint32m4_t _mul = vmul_vv_i32m4(_tmp_w, _g_ww, vl);

            if (shift2 < 0) {
                _mul = vssra_vx_i32m4(_mul, -shift2 - 1, vl);
            } else {
                _mul = vsll_vx_i32m4(_mul, shift2 + 1, vl);
            }
            vint32m4_t _mulh = vmulh_vx_i32m4(_mul, multiplier2, vl);

            vint32m4_t _beta = vle32_v_i32m4(qb, vl);
            qb += vl;
            vint32m4_t _res0 = vadd_vv_i32m4(_mulh, _beta, vl);
            vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);
            vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);
            vse8_v_i8m1(output_ptr, _res2, vl);
            output_ptr += vl;
            size -= vl;
        }
        shl_mem_free(tmp);
    }
    shl_mem_free(qbeta);

    return CSINN_TRUE;
}

int shl_rvv_layer_norm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *gamma, struct csinn_tensor *beta,
                            struct csinn_layer_norm_params *params)
{
    struct csinn_tensor *float_input = shl_rvv_tensor_transform_f32(input);
    struct csinn_tensor *float_output = shl_rvv_tensor_transform_f32(output);
    struct csinn_tensor *float_gamma = shl_rvv_tensor_transform_f32(gamma);
    struct csinn_tensor *float_beta = shl_rvv_tensor_transform_f32(beta);
    if (float_input == NULL) {
        shl_debug_warning(
            "shl_rvv_tensor_transform_f32 is not optimized to achieve under this condition on RVV, "
            "call reference func replaced.\n");
        float_input = shl_ref_tensor_transform_f32(input);
    }
    if (float_output == NULL) {
        shl_debug_warning(
            "shl_rvv_tensor_transform_f32 is not optimized to achieve under this condition on RVV, "
            "call reference func replaced.\n");
        float_output = shl_ref_tensor_transform_f32(output);
    }
    if (float_gamma == NULL) {
        shl_debug_warning(
            "shl_rvv_tensor_transform_f32 is not optimized to achieve under this condition on RVV, "
            "call reference func replaced.\n");
        float_gamma = shl_ref_tensor_transform_f32(gamma);
    }
    if (float_beta == NULL) {
        shl_debug_warning(
            "shl_rvv_tensor_transform_f32 is not optimized to achieve under this condition on RVV, "
            "call reference func replaced.\n");
        float_beta = shl_ref_tensor_transform_f32(beta);
    }

    int ret = shl_rvv_layer_norm_fp32(float_input, float_output, float_gamma, float_beta, params);

    if (shl_rvv_tensor_data_convert(float_output, output) != CSINN_TRUE) {
        shl_debug_warning(
            "shl_rvv_tensor_data_convert is not optimized to achieve under this condition on RVV, "
            "call reference func replaced.\n");
        csinn_tensor_data_convert(output, float_output);
    }

    shl_ref_tensor_transform_free_f32(float_input);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_gamma);
    shl_ref_tensor_transform_free_f32(float_beta);

    return ret;
}
