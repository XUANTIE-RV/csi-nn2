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

static inline void div_vv_f32m4(float *in0, float *in1, float *out, int32_t size)
{
    while (size > 0) {
        int vl = vsetvl_e32m4(size);
        vfloat32m4_t _a = vle32_v_f32m4(in0, vl);
        vfloat32m4_t _b = vle32_v_f32m4(in1, vl);
        vfloat32m4_t _c = vfdiv_vv_f32m4(_a, _b, vl);
        vse32_v_f32m4(out, _c, vl);
        in0 += vl;
        in1 += vl;
        out += vl;
        size -= vl;
    }
}

static inline void div_vf_f32m4(float *in0, float *in1, float *out, int32_t size)
{
    while (size > 0) {
        int vl = vsetvl_e32m4(size);
        vfloat32m4_t _a = vle32_v_f32m4(in0, vl);
        vfloat32m4_t _c = vfdiv_vf_f32m4(_a, in1[0], vl);
        vse32_v_f32m4(out, _c, vl);
        in0 += vl;
        out += vl;
        size -= vl;
    }
}

static inline void div_fv_f32m4(float *in0, float *in1, float *out, int32_t size)
{
    while (size > 0) {
        int vl = vsetvl_e32m4(size);
        vfloat32m4_t _b = vle32_v_f32m4(in1, vl);
        vfloat32m4_t _c = vfrdiv_vf_f32m4(_b, in0[0], vl);
        vse32_v_f32m4(out, _c, vl);
        in1 += vl;
        out += vl;
        size -= vl;
    }
}

void *div_cb_fp32[] = {
    [CSINN_BROADCAST_VV] = div_vv_f32m4,
    [CSINN_BROADCAST_VS] = div_vf_f32m4,
    [CSINN_BROADCAST_SV] = div_fv_f32m4,
};

int shl_rvv_div_fp32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    return shl_rvv_binary_op_broadcast_fp32(input0, input1, output, div_cb_fp32);
}
