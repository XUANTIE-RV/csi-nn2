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

/* SHL version 2.1.x */

#include "shl_thead_rvv.h"
#ifdef SHL_USE_DOT_INT8
static vint8mf2_t requantize_m2_s(vint32m2_t _src, vint32m2_t _multiplier, vint32m2_t _shift,
                                  int32_t out_zp, int vl)
{
}

int shl_rvv_dwconv3x3s1_packn_int8_dot(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params)
{
    return CSINN_FALSE;
}

int shl_rvv_dwconv3x3s2_packn_int8_dot(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params)
{
    return CSINN_FALSE;
}

/****************************************************************************
 * packn = vlenb / sizeof(int8_t) / 2
 * maxk = ksize_h * ksize_w
 * constrain: out_c % packn = 0 and in_ch = 1
 * layout: [out_c, 1, ksize_h, ksize_w] ==> [out_c/packn, 1, maxk, packn]
 ***************************************************************************/
void shl_rvv_dwconv_reorder_kernel_packn_int8_dot(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    const int out_ch = kernel->dim[0];
    const int maxk = kernel->dim[2] * kernel->dim[3];
    int8_t *kernel_trans = (int8_t *)shl_mem_alloc(out_ch * maxk * sizeof(int8_t));

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8mf2(packn);

    for (int oc = 0; oc + packn - 1 < out_ch; oc += packn) {
        int8_t *ksrc = kernel_data + oc * maxk;
        int8_t *kdst = kernel_trans + oc * maxk;
        for (int ic = 0; ic < maxk; ic++) {
            vint8mf2_t _tmp = vlse8_v_i8mf2(ksrc + ic, maxk * sizeof(int8_t), vl);
            vse8_v_i8mf2(kdst, _tmp, vl);
            kdst += vl;
        }
    }
    memcpy(kernel_data, kernel_trans, out_ch * maxk * sizeof(int8_t));
    shl_mem_free(kernel_trans);
}
#endif
