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

#include "shl_thead_rvv.h"

/*************************************************************
    note: support flexible vlen
*************************************************************/
int shl_rvv_layer_norm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *gamma, struct csinn_tensor *beta,
                            struct csinn_layer_norm_params *params)
{
    if (params->center == false || params->scale == false) {
        shl_debug_error("Layer norm only support center & scale == true\n");
        return CSINN_FALSE;
    }
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(input);
    }
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *gamma_data = (float *)gamma->data;
    float *beta_data = (float *)beta->data;

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

    float *tmp = (float *)shl_mem_alloc(norm_size * sizeof(float));
    for (int b = 0; b < batches; b++) {
        float *input_ptr = input_data + b * norm_size;
        float *output_ptr = output_data + b * norm_size;

        vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.0f, 1);
        float *in0 = input_ptr;
        int size = norm_size;
        while (size > 0) {
            int vl = vsetvl_e32m1(size);
            vfloat32m1_t _in = vle32_v_f32m1(in0, vl);
            in0 += vl;
            _sum1 = vfredosum_vs_f32m1_f32m1(vundefined_f32m1(), _in, _sum1, vl);
            size -= vl;
        }
        float sum1 = vfmv_f_s_f32m1_f32(_sum1);
        float mean = sum1 / norm_size;

        vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.0f, 1);
        float *t0 = tmp;
        in0 = input_ptr;
        size = norm_size;
        while (size > 0) {
            int vl = vsetvl_e32m1(size);
            vfloat32m1_t _in = vle32_v_f32m1(in0, vl);
            in0 += vl;
            vfloat32m1_t _tmp = vfsub_vf_f32m1(_in, mean, vl);
            vse32_v_f32m1(t0, _tmp, vl);
            t0 += vl;
            vfloat32m1_t _mul = vfmul_vv_f32m1(_tmp, _tmp, vl);
            _sum2 = vfredosum_vs_f32m1_f32m1(vundefined_f32m1(), _mul, _sum2, vl);
            size -= vl;
        }
        float sum2 = vfmv_f_s_f32m1_f32(_sum2);
        float var = sum2 / norm_size;
        float std = sqrt(var + params->epsilon);

        float *g0 = gamma_data;
        float *b0 = beta_data;
        t0 = tmp;
        size = norm_size;
        while (size > 0) {
            int vl = vsetvl_e32m1(size);
            vfloat32m1_t _tmp = vle32_v_f32m1(t0, vl);
            t0 += vl;
            vfloat32m1_t _gamma = vle32_v_f32m1(g0, vl);
            g0 += vl;
            vfloat32m1_t _beta = vle32_v_f32m1(b0, vl);
            b0 += vl;
            vfloat32m1_t _tmp2 = vfdiv_vf_f32m1(_tmp, std, vl);
            vfloat32m1_t _res = vfmacc_vv_f32m1(_beta, _tmp2, _gamma, vl);
            vse32_v_f32m1(output_ptr, _res, vl);
            output_ptr += vl;
            size -= vl;
        }
    }
    shl_mem_free(tmp);

    return CSINN_TRUE;
}
