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

static inline void sub_vv_f16m4(__fp16 *in0, __fp16 *in1, __fp16 *out, int32_t size)
{
    while (size > 0) {
        int vl = vsetvl_e16m4(size);
        vfloat16m4_t _a = vle16_v_f16m4(in0, vl);
        vfloat16m4_t _b = vle16_v_f16m4(in1, vl);
        vfloat16m4_t _c = vfsub_vv_f16m4(_a, _b, vl);
        vse16_v_f16m4(out, _c, vl);
        in0 += vl;
        in1 += vl;
        out += vl;
        size -= vl;
    }
}

static inline void sub_vf_f16m4(__fp16 *in0, __fp16 *in1, __fp16 *out, int32_t size)
{
    while (size > 0) {
        int vl = vsetvl_e16m4(size);
        vfloat16m4_t _a = vle16_v_f16m4(in0, vl);
        vfloat16m4_t _c = vfsub_vf_f16m4(_a, in1[0], vl);
        vse16_v_f16m4(out, _c, vl);
        in0 += vl;
        out += vl;
        size -= vl;
    }
}

static inline void sub_fv_f16m4(__fp16 *in0, __fp16 *in1, __fp16 *out, int32_t size)
{
    while (size > 0) {
        int vl = vsetvl_e16m4(size);
        vfloat16m4_t _b = vle16_v_f16m4(in1, vl);
        vfloat16m4_t _c = vfrsub_vf_f16m4(_b, in0[0], vl);
        vse16_v_f16m4(out, _c, vl);
        in1 += vl;
        out += vl;
        size -= vl;
    }
}

void *sub_cb_fp16[] = {
    [CSINN_BROADCAST_VV] = sub_vv_f16m4,
    [CSINN_BROADCAST_VS] = sub_vf_f16m4,
    [CSINN_BROADCAST_SV] = sub_fv_f16m4,
};

int shl_rvv_sub_fp16(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    return shl_rvv_binary_op_broadcast_fp16(input0, input1, output, sub_cb_fp16);
}
