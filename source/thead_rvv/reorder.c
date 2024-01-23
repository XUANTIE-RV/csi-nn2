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

/************************************************************************
 * pack1ton: change input(activation) layout from nchw to nc1hwc0
 *           当 inc 不是 packn 的倍数时, 末梢单独处理(用 vl 控制)
 * packnto1: change input(activation) layout from nc1hwc0 to nchw
 ***********************************************************************/
// constrains: inc % packn = 0
void shl_rvv_reorder_input_pack1ton_fp32(const float *src, float *dst, int inc, int inh, int inw)
{
    const int packn = csrr_vlenb() / sizeof(float);
    int vl = vsetvl_e32m1(packn);
    const int in_size = inh * inw;  // per-channel size

    while (inc > 0) {
        vl = vsetvl_e32m1(inc);
        float *in_ptr = (float *)src;
        for (int i = 0; i < inh; i++) {
            for (int j = 0; j < inw; j++) {
                vfloat32m1_t _tmp = vlse32_v_f32m1(in_ptr, in_size * sizeof(float), vl);
                in_ptr++;
                vse32_v_f32m1(dst, _tmp, vl);
                dst += vl;
            }
        }
        src += in_size * vl;
        inc -= vl;
    }
}

void shl_rvv_reorder_input_pack1ton_fp16(const __fp16 *src, __fp16 *dst, int inc, int inh, int inw)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    int vl = vsetvl_e16m1(packn);
    const int in_size = inh * inw;  // per-channel size

    while (inc > 0) {
        vl = vsetvl_e16m1(inc);
        __fp16 *in_ptr = (__fp16 *)src;
        for (int i = 0; i < inh; i++) {
            for (int j = 0; j < inw; j++) {
                vfloat16m1_t _tmp = vlse16_v_f16m1(in_ptr, in_size * sizeof(__fp16), vl);
                in_ptr++;
                vse16_v_f16m1(dst, _tmp, vl);
                dst += vl;
            }
        }
        src += in_size * vl;
        inc -= vl;
    }
}

void shl_rvv_reorder_input_pack1ton_int8(const int8_t *src, int8_t *dst, int inc, int inh, int inw)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int vl = vsetvl_e8m1(packn);
    const int in_size = inh * inw;  // per-channel size

    while (inc > 0) {
        vl = vsetvl_e8m1(inc > packn ? packn : inc);
        int8_t *in_ptr = (int8_t *)src;
        for (int i = 0; i < inh; i++) {
            for (int j = 0; j < inw; j++) {
                vint8m1_t _tmp = vlse8_v_i8m1(in_ptr, in_size * sizeof(int8_t), vl);
                in_ptr++;
                vse8_v_i8m1(dst, _tmp, vl);
                dst += vl;
            }
        }
        src += in_size * vl;
        inc -= vl;
    }
}

// constrains: inc % packn = 0 (tail)
void shl_rvv_reorder_input_packnto1_fp32(const float *src, float *dst, int inc, int inh, int inw)
{
    const int packn = csrr_vlenb() / sizeof(float);
    int vl = vsetvl_e32m1(packn);
    const int in_size = inh * inw;  // per-channel size

    while (inc > 0) {
        int vl = vsetvl_e32m1(inc);
        float *out_ptr = dst;
        for (int i = 0; i < inh; i++) {
            for (int j = 0; j < inw; j++) {
                vfloat32m1_t _tmp = vle32_v_f32m1(src, vl);
                src += vl;
                vsse32_v_f32m1(out_ptr, in_size * sizeof(float), _tmp, vl);
                out_ptr++;
            }
        }
        dst += in_size * vl;
        inc -= vl;
    }
}

void shl_rvv_reorder_input_packnto1_fp16(const __fp16 *src, __fp16 *dst, int inc, int inh, int inw)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    int vl = vsetvl_e16m1(packn);
    const int in_size = inh * inw;  // per-channel size

    while (inc > 0) {
        vl = vsetvl_e16m1(inc);
        __fp16 *out_ptr = dst;
        for (int i = 0; i < inh; i++) {
            for (int j = 0; j < inw; j++) {
                vfloat16m1_t _tmp = vle16_v_f16m1(src, vl);
                src += vl;
                vsse16_v_f16m1(out_ptr, in_size * sizeof(__fp16), _tmp, vl);
                out_ptr++;
            }
        }
        dst += in_size * vl;
        inc -= vl;
    }
}

void shl_rvv_reorder_input_packnto1_int8(const int8_t *src, int8_t *dst, int inc, int inh, int inw)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int vl = vsetvl_e8m1(packn);
    const int in_size = inh * inw;  // per-channel size

    while (inc > 0) {
        vl = vsetvl_e8m1(inc > packn ? packn : inc);
        int8_t *out_ptr = dst;
        for (int i = 0; i < inh; i++) {
            for (int j = 0; j < inw; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(src, vl);
                src += vl;
                vsse8_v_i8m1(out_ptr, in_size * sizeof(int8_t), _tmp, vl);
                out_ptr++;
            }
        }
        dst += in_size * vl;
        inc -= vl;
    }
}

/************************************************************************
 * reorder kernel matrix
 ***********************************************************************/
// vlen=128
void shl_rvv_reorder_kernel_n8_fp32(float *a, float *sa, int m, int k, int ldx)
{
    int i = 0;
    for (; i + 7 < m; i += 8) {
        for (int j = 0; j < k; j++) {
            float *in_ptr = a + j;
            vfloat32m2_t _input = vlse32_v_f32m2(in_ptr, k * sizeof(float), 8);
            vse32_v_f32m2(sa, _input, 8);
            sa += 8;
        }
        a += 8 * k;
    }
    for (; i + 3 < m; i += 4) {
        for (int j = 0; j < k; j++) {
            float *in_ptr = a + j;
            vfloat32m1_t _input = vlse32_v_f32m1(in_ptr, k * sizeof(float), 4);
            vse32_v_f32m1(sa, _input, 4);
            sa += 4;
        }
        a += 4 * k;
    }
    for (; i + 1 < m; i += 2) {
        for (int j = 0; j < k; j++) {
            float *in_ptr = a + j;
            vfloat32m1_t _input = vlse32_v_f32m1(in_ptr, k * sizeof(float), 2);
            vse32_v_f32m1(sa, _input, 2);
            sa += 2;
        }
        a += 2 * k;
    }
    for (; i < m; i++) {
        memcpy(sa, a, k * sizeof(float));
    }
}

void shl_rvv_reorder_kernel_n8_fp16(__fp16 *a, __fp16 *sa, int m, int k, int ldx)
{
    int i = 0;
    for (; i + 7 < m; i += 8) {
        for (int j = 0; j < k; j++) {
            __fp16 *in_ptr = a + j;
            vfloat16m1_t _input = vlse16_v_f16m1(in_ptr, k * sizeof(__fp16), 8);
            vse16_v_f16m1(sa, _input, 8);
            sa += 8;
        }
        a += 8 * k;
    }
    for (; i + 3 < m; i += 4) {
        for (int j = 0; j < k; j++) {
            __fp16 *in_ptr = a + j;
            vfloat16m1_t _input = vlse16_v_f16m1(in_ptr, k * sizeof(__fp16), 4);
            vse16_v_f16m1(sa, _input, 4);
            sa += 4;
        }
        a += 4 * k;
    }
    for (; i + 1 < m; i += 2) {
        for (int j = 0; j < k; j++) {
            __fp16 *in_ptr = a + j;
            vfloat16m1_t _input = vlse16_v_f16m1(in_ptr, k * sizeof(__fp16), 2);
            vse16_v_f16m1(sa, _input, 2);
            sa += 2;
        }
        a += 2 * k;
    }
    for (; i < m; i++) {
        memcpy(sa, a, k * sizeof(__fp16));
    }
}

void shl_rvv_reorder_kernel_n8_fp16_w_int8(int8_t *a, int8_t *sa, int m, int k, int ldx)
{
    int i = 0;
    for (; i + 7 < m; i += 8) {
        for (int j = 0; j < k; j++) {
            int8_t *in_ptr = a + j;
            vint8m1_t _input = vlse8_v_i8m1(in_ptr, k * sizeof(int8_t), 8);
            vse8_v_i8m1(sa, _input, 8);
            sa += 8;
        }
        a += 8 * k;
    }
    for (; i + 3 < m; i += 4) {
        for (int j = 0; j < k; j++) {
            int8_t *in_ptr = a + j;
            vint8m1_t _input = vlse8_v_i8m1(in_ptr, k * sizeof(int8_t), 4);
            vse8_v_i8m1(sa, _input, 4);
            sa += 4;
        }
        a += 4 * k;
    }
    for (; i + 1 < m; i += 2) {
        for (int j = 0; j < k; j++) {
            int8_t *in_ptr = a + j;
            vint8m1_t _input = vlse8_v_i8m1(in_ptr, k * sizeof(int8_t), 2);
            vse8_v_i8m1(sa, _input, 2);
            sa += 2;
        }
        a += 2 * k;
    }
    for (; i < m; i++) {
        memcpy(sa, a, k * sizeof(int8_t));
    }
}

void shl_rvv_reorder_kernel_n8_int8_dot(int8_t *a, int8_t *sa, int m, int k, int ldx)
{
    int i = 0;
    for (; i + 7 < m; i += 8) {
        int j = 0;
        for (; j + 3 < k; j += 4) {
            int8_t *in_ptr = a + j;
            for (int c = 0; c < 8; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 4);
                in_ptr += k;
                vse8_v_i8m1(sa, _input, 4);
                sa += 4;
            }
        }
        // k_tail
        if (j < k) {
            int8_t *in_ptr = a + j;
            for (int c = 0; c < 8; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, k & 3);
                in_ptr += k;
                vse8_v_i8m1(sa, _input, k & 3);
                sa += 4;
            }
        }
        a += 8 * k;
    }
    for (; i + 3 < m; i += 4) {
        int j = 0;
        for (; j + 3 < k; j += 4) {
            int8_t *in_ptr = a + j;
            for (int c = 0; c < 4; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 4);
                in_ptr += k;
                vse8_v_i8m1(sa, _input, 4);
                sa += 4;
            }
        }
        if (j < k) {
            int8_t *in_ptr = a + j;
            for (int c = 0; c < 4; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, k & 3);
                in_ptr += k;
                vse8_v_i8m1(sa, _input, k & 3);
                sa += 4;
            }
        }
        a += 4 * k;
    }
    for (; i + 1 < m; i += 2) {
        int j = 0;
        for (; j + 3 < k; j += 4) {
            int8_t *in_ptr = a + j;
            for (int c = 0; c < 2; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 4);
                in_ptr += k;
                vse8_v_i8m1(sa, _input, 4);
                sa += 4;
            }
        }
        if (j < k) {
            int8_t *in_ptr = a + j;
            for (int c = 0; c < 2; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, k & 3);
                in_ptr += k;
                vse8_v_i8m1(sa, _input, k & 3);
                sa += 4;
            }
        }
        a += 2 * k;
    }
    for (; i < m; i++) {
        memcpy(sa, a, k * sizeof(int8_t));
    }
}

// n4 n2 n1
void shl_rvv_reorder_kernel_n4_int8_v128(int8_t *a, int8_t *sa, int m, int k, int ldx)
{
    int i = 0;
    for (; i + 3 < m; i += 4) {
        for (int j = 0; j < k; j++) {
            int8_t *in_ptr = a + j;
            vint8m1_t _input = vlse8_v_i8m1(in_ptr, k * sizeof(int8_t), 4);
            vse8_v_i8m1(sa, _input, 4);
            sa += 4;
        }
        a += 4 * k;
    }
    for (; i + 1 < m; i += 2) {
        for (int j = 0; j < k; j++) {
            int8_t *in_ptr = a + j;
            vint8m1_t _input = vlse8_v_i8m1(in_ptr, k * sizeof(int8_t), 2);
            vse8_v_i8m1(sa, _input, 2);
            sa += 2;
        }
        a += 2 * k;
    }
    for (; i < m; i++) {
        memcpy(sa, a, k * sizeof(int8_t));
    }
}

// flexible vlen
/*************************************************************
 * constrain: m(out_channel) % packn = 0; k % packn = 0
 * e.g. vlen=128, n8 --> n4
 ************************************************************/
void shl_rvv_reorder_kernel_packn_fp32(float *a, float *sa, int m, int k, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int pack2n = packn * 2;
    int vl = vsetvl_e32m2(pack2n);
    int oc = 0;
    for (; oc + pack2n - 1 < m; oc += pack2n) {
        float *g0 = a + oc * k;
        for (int ic = 0; ic < k; ic++) {
            vfloat32m2_t _tmp = vlse32_v_f32m2(g0 + ic, k * sizeof(float), vl);
            vse32_v_f32m2(sa, _tmp, vl);
            sa += vl;
        }
    }
    vl = vsetvl_e32m1(packn);
    for (; oc + packn - 1 < m; oc += packn) {
        float *g0 = a + oc * k;
        for (int ic = 0; ic < k; ic++) {
            vfloat32m1_t _tmp = vlse32_v_f32m1(g0 + ic, k * sizeof(float), vl);
            vse32_v_f32m1(sa, _tmp, vl);
            sa += vl;
        }
    }
}

/*************************************************************
 * constrain: m(out_channel) % packn = 0; k % packn = 0
 * e.g. vlen=128, n16 --> n8
 ************************************************************/
void shl_rvv_reorder_kernel_packn_fp16(__fp16 *a, __fp16 *sa, int m, int k, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int pack2n = packn * 2;
    int vl = vsetvl_e16m2(pack2n);
    int oc = 0;
    for (; oc + pack2n - 1 < m; oc += pack2n) {
        __fp16 *g0 = a + oc * k;
        for (int ic = 0; ic < k; ic++) {
            vfloat16m2_t _tmp = vlse16_v_f16m2(g0 + ic, k * sizeof(__fp16), vl);
            vse16_v_f16m2(sa, _tmp, vl);
            sa += vl;
        }
    }
    vl = vsetvl_e16m1(packn);
    for (; oc + packn - 1 < m; oc += packn) {
        __fp16 *g0 = a + oc * k;
        for (int ic = 0; ic < k; ic++) {
            vfloat16m1_t _tmp = vlse16_v_f16m1(g0 + ic, k * sizeof(__fp16), vl);
            vse16_v_f16m1(sa, _tmp, vl);
            sa += vl;
        }
    }
}

/************************************************************************
 * reorder input matrix
 ***********************************************************************/

// vlen=128
/**************************************************************
 * Data arrangement: Z8 | | |
 **************************************************************/
void shl_rvv_reorder_input_z8_fp32(float *b, float *sb, int k, int n, int ldx)
{
    int32_t vl = vsetvl_e32m2(8);
    float *b0 = NULL;
    int i = 0;
    for (; i + 7 < n; i += 8) {
        b0 = b + i;
        for (int j = 0; j < k; j++) {
            vfloat32m2_t _tmp = vle32_v_f32m2(b0, vl);
            b0 += ldx;
            vse32_v_f32m2(sb, _tmp, vl);
            sb += 8;
        }
    }

    for (; i < n; i++) {
        vl = vsetvl_e32m2(8);
        b0 = b + i;
        int j = 0;
        for (; j + 7 < k; j += 8) {
            vfloat32m2_t _tmp = vlse32_v_f32m2(b0, ldx * sizeof(float), vl);
            b0 += 8 * ldx;
            vse32_v_f32m2(sb, _tmp, vl);
            sb += 8;
        }
        if (j < k) {
            vl = vsetvl_e32m2(k & 7);
            vfloat32m2_t _tmp = vlse32_v_f32m2(b0, ldx * sizeof(float), vl);
            vse32_v_f32m2(sb, _tmp, vl);
            sb += vl;
        }
    }
}

/**************************************************************
 * Data arrangement: Z16 Z8 | | |
 **************************************************************/
void shl_rvv_reorder_input_z16_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx)
{
    int vl = vsetvl_e16m2(16);
    __fp16 *b0 = NULL;
    int i = 0;
    for (; i + 15 < n; i += 16) {
        b0 = b + i;
        for (int j = 0; j < k; j++) {
            vfloat16m2_t _tmp = vle16_v_f16m2(b0, vl);
            b0 += ldx;
            vse16_v_f16m2(sb, _tmp, vl);
            sb += 16;
        }
    }

    for (; i + 7 < n; i += 8) {
        vl = vsetvl_e16m1(8);
        b0 = b + i;
        for (int j = 0; j < k; j++) {
            vfloat16m1_t _tmp = vle16_v_f16m1(b0, vl);
            b0 += ldx;
            vse16_v_f16m1(sb, _tmp, vl);
            sb += 8;
        }
    }

    for (; i < n; i++) {
        vl = vsetvl_e16m2(16);
        b0 = b + i;
        int j = 0;
        for (; j + 15 < k; j += 16) {
            vfloat16m2_t _tmp = vlse16_v_f16m2(b0, ldx * sizeof(__fp16), vl);
            b0 += 16 * ldx;
            vse16_v_f16m2(sb, _tmp, vl);
            sb += 16;
        }
        if (j < k) {
            vl = vsetvl_e16m2(k & 15);
            vfloat16m2_t _tmp = vlse16_v_f16m2(b0, ldx * sizeof(__fp16), vl);
            vse16_v_f16m2(sb, _tmp, vl);
            sb += vl;
        }
    }
}

/**************************************************************
 * Data arrangement: Z8 Z4 | | |
 **************************************************************/
void shl_rvv_reorder_input_z8_int8_dot(int8_t *b, int8_t *sb, int k, int n, int ldx)
{
    int vl = vsetvl_e8m1(8);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        int8_t *b0 = b + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(sb, 4 * sizeof(int8_t), _tmp, vl);
            sb++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(sb, 4 * sizeof(int8_t), _tmp, vl);
            sb++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(sb, 4 * sizeof(int8_t), _tmp, vl);
            sb++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(sb, 4 * sizeof(int8_t), _tmp, vl);
            sb += 32 - 3;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = sb;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
            sb += 32;
        }
    }
    for (; i + 3 < n; i += 4) {
        vl = vsetvl_e8m1(4);
        int8_t *b0 = b + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(sb, 4 * sizeof(int8_t), _tmp, vl);
            sb++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(sb, 4 * sizeof(int8_t), _tmp, vl);
            sb++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(sb, 4 * sizeof(int8_t), _tmp, vl);
            sb++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(sb, 4 * sizeof(int8_t), _tmp, vl);
            sb += 13;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = sb;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
            sb += 16;
        }
    }
    // n_tail
    for (; i < n; i++) {
        vl = vsetvl_e8m1(16);
        int8_t *b0 = b + i;
        int j = 0;
        for (; j + 15 < k; j += 16) {
            vint8m1_t _tmp = vlse8_v_i8m1(b0, ldx * sizeof(int8_t), vl);
            b0 += 16 * ldx;
            vse8_v_i8m1(sb, _tmp, vl);
            sb += 16;
        }
        if (j < k) {
            vl = vsetvl_e8m1(k & 15);
            vint8m1_t _tmp = vlse8_v_i8m1(b0, ldx * sizeof(int8_t), vl);
            vse8_v_i8m1(sb, _tmp, vl);
            sb += ((k - j - 1) / 4 + 1) * 4;
        }
    }
}

/**************************************************************
 * Data arrangement: Z16 Z8 Z8_tail
 **************************************************************/
void shl_rvv_reorder_input_z16_int8_v128(int8_t *b, int8_t *sb, int k, int n, int ldx)
{
    int vl = vsetvl_e8m1(16);
    int8_t *b0 = NULL;
    int i = 0;
    for (; i + 15 < n; i += 16) {
        b0 = b + i;
        for (int j = 0; j < k; j++) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += ldx;
            vse8_v_i8m1(sb, _tmp, vl);
            sb += 16;
        }
    }

    for (; i + 7 < n; i += 8) {
        vl = vsetvl_e8m1(8);
        b0 = b + i;
        for (int j = 0; j < k; j++) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += ldx;
            vse8_v_i8m1(sb, _tmp, vl);
            sb += 8;
        }
    }
    // tail
    if (i < n) {
        vl = vsetvl_e8m1(n - i);
        b0 = b + i;
        for (int j = 0; j < k; j++) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += ldx;
            vse8_v_i8m1(sb, _tmp, vl);
            sb += vl;
        }
    }
}

// flexible vlen
/**************************************************************
 * src: b   [inc/packn, maxk, n, packn] + [maxk, n, inc%packn]
 * dst: sb  [n/12, inc/packn * maxk * packn + maxk * inc%packn, 12]
 * Data arrangement: Z12 Z8 Z4 Z2 Z1
 * 注意 inc 在 packn 倍数和非 packn 的倍数时边界点
 **************************************************************/
void shl_rvv_reorder_input_z12_pack1ton_fp32(float *b, float *sb, int inc, int maxk, int n, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(float);
    int vl = vsetvl_e32m1(inc);

    int t = 0;
    for (; t + 11 < n; t += 12) {
        const float *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e32m1(loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, avl);
                vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + avl * 1, avl);
                vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + avl * 2, avl);
                vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + avl * 3, avl);
                vfloat32m1_t _tmp4 = vle32_v_f32m1(tm1 + avl * 4, avl);
                vfloat32m1_t _tmp5 = vle32_v_f32m1(tm1 + avl * 5, avl);
                vfloat32m1_t _tmp6 = vle32_v_f32m1(tm1 + avl * 6, avl);
                vfloat32m1_t _tmp7 = vle32_v_f32m1(tm1 + avl * 7, avl);
                vfloat32m1_t _tmp8 = vle32_v_f32m1(tm1 + avl * 8, avl);
                vfloat32m1_t _tmp9 = vle32_v_f32m1(tm1 + avl * 9, avl);
                vfloat32m1_t _tmp10 = vle32_v_f32m1(tm1 + avl * 10, avl);
                vfloat32m1_t _tmp11 = vle32_v_f32m1(tm1 + avl * 11, avl);

                vsse32_v_f32m1(sb, 12 * sizeof(float), _tmp0, avl);
                vsse32_v_f32m1(sb + 1, 12 * sizeof(float), _tmp1, avl);
                vsse32_v_f32m1(sb + 2, 12 * sizeof(float), _tmp2, avl);
                vsse32_v_f32m1(sb + 3, 12 * sizeof(float), _tmp3, avl);
                vsse32_v_f32m1(sb + 4, 12 * sizeof(float), _tmp4, avl);
                vsse32_v_f32m1(sb + 5, 12 * sizeof(float), _tmp5, avl);
                vsse32_v_f32m1(sb + 6, 12 * sizeof(float), _tmp6, avl);
                vsse32_v_f32m1(sb + 7, 12 * sizeof(float), _tmp7, avl);
                vsse32_v_f32m1(sb + 8, 12 * sizeof(float), _tmp8, avl);
                vsse32_v_f32m1(sb + 9, 12 * sizeof(float), _tmp9, avl);
                vsse32_v_f32m1(sb + 10, 12 * sizeof(float), _tmp10, avl);
                vsse32_v_f32m1(sb + 11, 12 * sizeof(float), _tmp11, avl);

                tm1 += n * avl;
                sb += 12 * avl;
            }
            loop_c -= avl;
        }
    }
    for (; t + 7 < n; t += 8) {
        const float *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e32m1(loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, avl);
                vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + avl * 1, avl);
                vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + avl * 2, avl);
                vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + avl * 3, avl);
                vfloat32m1_t _tmp4 = vle32_v_f32m1(tm1 + avl * 4, avl);
                vfloat32m1_t _tmp5 = vle32_v_f32m1(tm1 + avl * 5, avl);
                vfloat32m1_t _tmp6 = vle32_v_f32m1(tm1 + avl * 6, avl);
                vfloat32m1_t _tmp7 = vle32_v_f32m1(tm1 + avl * 7, avl);
                vsseg8e32_v_f32m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7, avl);
                tm1 += n * avl;
                sb += 8 * avl;
            }
            loop_c -= avl;
        }
    }
    for (; t + 3 < n; t += 4) {
        const float *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e32m1(loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, avl);
                vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + avl * 1, avl);
                vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + avl * 2, avl);
                vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + avl * 3, avl);
                vsseg4e32_v_f32m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, avl);
                tm1 += n * avl;
                sb += 4 * avl;
            }
            loop_c -= avl;
        }
    }
    for (; t + 1 < n; t += 2) {
        const float *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e32m1(loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, avl);
                vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + avl * 1, avl);
                vsseg2e32_v_f32m1(sb, _tmp0, _tmp1, avl);
                tm1 += n * avl;
                sb += 2 * avl;
            }
            loop_c -= avl;
        }
    }
    for (; t < n; t++) {
        const float *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e32m1(loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, avl);
                vse32_v_f32m1(sb, _tmp0, avl);
                tm1 += n * avl;
                sb += 1 * avl;
            }
            loop_c -= avl;
        }
    }
}

void shl_rvv_reorder_input_z12_pack1ton_fp16(__fp16 *b, __fp16 *sb, int inc, int maxk, int n,
                                             int ldx)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    int vl = vsetvl_e16m1(inc);

    int t = 0;
    for (; t + 11 < n; t += 12) {
        const __fp16 *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e16m1(loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, avl);
                vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + avl * 1, avl);
                vfloat16m1_t _tmp2 = vle16_v_f16m1(tm1 + avl * 2, avl);
                vfloat16m1_t _tmp3 = vle16_v_f16m1(tm1 + avl * 3, avl);
                vfloat16m1_t _tmp4 = vle16_v_f16m1(tm1 + avl * 4, avl);
                vfloat16m1_t _tmp5 = vle16_v_f16m1(tm1 + avl * 5, avl);
                vfloat16m1_t _tmp6 = vle16_v_f16m1(tm1 + avl * 6, avl);
                vfloat16m1_t _tmp7 = vle16_v_f16m1(tm1 + avl * 7, avl);
                vfloat16m1_t _tmp8 = vle16_v_f16m1(tm1 + avl * 8, avl);
                vfloat16m1_t _tmp9 = vle16_v_f16m1(tm1 + avl * 9, avl);
                vfloat16m1_t _tmp10 = vle16_v_f16m1(tm1 + avl * 10, avl);
                vfloat16m1_t _tmp11 = vle16_v_f16m1(tm1 + avl * 11, avl);

                vsse16_v_f16m1(sb, 12 * sizeof(__fp16), _tmp0, avl);
                vsse16_v_f16m1(sb + 1, 12 * sizeof(__fp16), _tmp1, avl);
                vsse16_v_f16m1(sb + 2, 12 * sizeof(__fp16), _tmp2, avl);
                vsse16_v_f16m1(sb + 3, 12 * sizeof(__fp16), _tmp3, avl);
                vsse16_v_f16m1(sb + 4, 12 * sizeof(__fp16), _tmp4, avl);
                vsse16_v_f16m1(sb + 5, 12 * sizeof(__fp16), _tmp5, avl);
                vsse16_v_f16m1(sb + 6, 12 * sizeof(__fp16), _tmp6, avl);
                vsse16_v_f16m1(sb + 7, 12 * sizeof(__fp16), _tmp7, avl);
                vsse16_v_f16m1(sb + 8, 12 * sizeof(__fp16), _tmp8, avl);
                vsse16_v_f16m1(sb + 9, 12 * sizeof(__fp16), _tmp9, avl);
                vsse16_v_f16m1(sb + 10, 12 * sizeof(__fp16), _tmp10, avl);
                vsse16_v_f16m1(sb + 11, 12 * sizeof(__fp16), _tmp11, avl);

                tm1 += n * avl;
                sb += 12 * avl;
            }
            loop_c -= avl;
        }
    }
    for (; t + 7 < n; t += 8) {
        const __fp16 *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e16m1(loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, avl);
                vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + avl * 1, avl);
                vfloat16m1_t _tmp2 = vle16_v_f16m1(tm1 + avl * 2, avl);
                vfloat16m1_t _tmp3 = vle16_v_f16m1(tm1 + avl * 3, avl);
                vfloat16m1_t _tmp4 = vle16_v_f16m1(tm1 + avl * 4, avl);
                vfloat16m1_t _tmp5 = vle16_v_f16m1(tm1 + avl * 5, avl);
                vfloat16m1_t _tmp6 = vle16_v_f16m1(tm1 + avl * 6, avl);
                vfloat16m1_t _tmp7 = vle16_v_f16m1(tm1 + avl * 7, avl);
                vsseg8e16_v_f16m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7, avl);
                tm1 += n * avl;
                sb += 8 * avl;
            }
            loop_c -= avl;
        }
    }
    for (; t + 3 < n; t += 4) {
        const __fp16 *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e16m1(loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, avl);
                vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + avl * 1, avl);
                vfloat16m1_t _tmp2 = vle16_v_f16m1(tm1 + avl * 2, avl);
                vfloat16m1_t _tmp3 = vle16_v_f16m1(tm1 + avl * 3, avl);
                vsseg4e16_v_f16m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, avl);
                tm1 += n * avl;
                sb += 4 * avl;
            }
            loop_c -= avl;
        }
    }
    for (; t + 1 < n; t += 2) {
        const __fp16 *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e16m1(loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, avl);
                vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + avl * 1, avl);
                vsseg2e16_v_f16m1(sb, _tmp0, _tmp1, avl);
                tm1 += n * avl;
                sb += 2 * avl;
            }
            loop_c -= avl;
        }
    }
    for (; t < n; t++) {
        const __fp16 *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e16m1(loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, avl);
                vse16_v_f16m1(sb, _tmp0, avl);
                tm1 += n * avl;
                sb += 1 * avl;
            }
            loop_c -= avl;
        }
    }
}

void shl_rvv_reorder_input_z4_pack1ton_int8(int8_t *b, int8_t *sb, int inc, int maxk, int n,
                                            int ldx)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int vl = vsetvl_e8m1(packn);

    int t = 0;
    for (; t + 3 < n; t += 4) {
        const int8_t *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e8m1(loop_c > packn ? packn : loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vint8m1_t _tmp0 = vle8_v_i8m1(tm1, avl);
                vint8m1_t _tmp1 = vle8_v_i8m1(tm1 + avl * 1, avl);
                vint8m1_t _tmp2 = vle8_v_i8m1(tm1 + avl * 2, avl);
                vint8m1_t _tmp3 = vle8_v_i8m1(tm1 + avl * 3, avl);
                vsseg4e8_v_i8m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, avl);
                tm1 += n * avl;
                sb += 4 * avl;
            }
            loop_c -= avl;
        }
    }
    for (; t + 1 < n; t += 2) {
        const int8_t *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e8m1(loop_c > packn ? packn : loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vint8m1_t _tmp0 = vle8_v_i8m1(tm1, avl);
                vint8m1_t _tmp1 = vle8_v_i8m1(tm1 + avl * 1, avl);
                vsseg2e8_v_i8m1(sb, _tmp0, _tmp1, avl);
                tm1 += n * avl;
                sb += 2 * avl;
            }
            loop_c -= avl;
        }
    }
    for (; t < n; t++) {
        const int8_t *tm1 = b + t * vl;
        int loop_c = inc;
        while (loop_c > 0) {
            int avl = vsetvl_e8m1(loop_c > packn ? packn : loop_c);
            tm1 += t * (avl - vl);
            for (int i = 0; i < maxk; i++) {
                vint8m1_t _tmp0 = vle8_v_i8m1(tm1, avl);
                vse8_v_i8m1(sb, _tmp0, avl);
                tm1 += n * avl;
                sb += 1 * avl;
            }
            loop_c -= avl;
        }
    }
}

/**************************************************************
 * inc % 4 = 0
 **************************************************************/
void shl_rvv_reorder_input_z12_pack1ton_int8_dot(int8_t *b, int8_t *sb, int inc, int maxk, int n,
                                                 int ldx)
{
#ifdef RVV_1_0_0
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int vl = vsetvl_e8mf2(inc);
    int avl = vl / 4;
    int avl_tail = (inc % packn) / 4;
    int32_t *dst = (int32_t *)sb;

    int t = 0;
    for (; t + 11 < n; t += 12) {
        const int32_t *tm1 = (const int32_t *)(b + t * vl);
        int ic = 0;
        for (; ic + packn - 1 < inc; ic += packn) {
            for (int i = 0; i < maxk; i++) {
                vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
                vsse32_v_i32mf2(dst, 12 * sizeof(int32_t), _col0, avl);
                vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
                vsse32_v_i32mf2(dst + 1, 12 * sizeof(int32_t), _col1, avl);
                vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
                vsse32_v_i32mf2(dst + 2, 12 * sizeof(int32_t), _col2, avl);
                vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
                vsse32_v_i32mf2(dst + 3, 12 * sizeof(int32_t), _col3, avl);
                vint32mf2_t _col4 = vle32_v_i32mf2(tm1 + avl * 4, avl);
                vsse32_v_i32mf2(dst + 4, 12 * sizeof(int32_t), _col4, avl);
                vint32mf2_t _col5 = vle32_v_i32mf2(tm1 + avl * 5, avl);
                vsse32_v_i32mf2(dst + 5, 12 * sizeof(int32_t), _col5, avl);
                vint32mf2_t _col6 = vle32_v_i32mf2(tm1 + avl * 6, avl);
                vsse32_v_i32mf2(dst + 6, 12 * sizeof(int32_t), _col6, avl);
                vint32mf2_t _col7 = vle32_v_i32mf2(tm1 + avl * 7, avl);
                vsse32_v_i32mf2(dst + 7, 12 * sizeof(int32_t), _col7, avl);
                vint32mf2_t _col8 = vle32_v_i32mf2(tm1 + avl * 8, avl);
                vsse32_v_i32mf2(dst + 8, 12 * sizeof(int32_t), _col8, avl);
                vint32mf2_t _col9 = vle32_v_i32mf2(tm1 + avl * 9, avl);
                vsse32_v_i32mf2(dst + 9, 12 * sizeof(int32_t), _col9, avl);
                vint32mf2_t _cola = vle32_v_i32mf2(tm1 + avl * 10, avl);
                vsse32_v_i32mf2(dst + 10, 12 * sizeof(int32_t), _cola, avl);
                vint32mf2_t _colb = vle32_v_i32mf2(tm1 + avl * 11, avl);
                vsse32_v_i32mf2(dst + 11, 12 * sizeof(int32_t), _colb, avl);

                dst += 12 * avl;
                tm1 += n * avl;
            }
        }
        if (ic < inc) {
            tm1 += t * (avl_tail - avl);
            for (int i = 0; i < maxk; i++) {
                vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl_tail);
                vsse32_v_i32mf2(dst, 12 * sizeof(int32_t), _col0, avl_tail);
                vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl_tail * 1, avl_tail);
                vsse32_v_i32mf2(dst + 1, 12 * sizeof(int32_t), _col1, avl_tail);
                vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl_tail * 2, avl_tail);
                vsse32_v_i32mf2(dst + 2, 12 * sizeof(int32_t), _col2, avl_tail);
                vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl_tail * 3, avl_tail);
                vsse32_v_i32mf2(dst + 3, 12 * sizeof(int32_t), _col3, avl_tail);
                vint32mf2_t _col4 = vle32_v_i32mf2(tm1 + avl_tail * 4, avl_tail);
                vsse32_v_i32mf2(dst + 4, 12 * sizeof(int32_t), _col4, avl_tail);
                vint32mf2_t _col5 = vle32_v_i32mf2(tm1 + avl_tail * 5, avl_tail);
                vsse32_v_i32mf2(dst + 5, 12 * sizeof(int32_t), _col5, avl_tail);
                vint32mf2_t _col6 = vle32_v_i32mf2(tm1 + avl_tail * 6, avl_tail);
                vsse32_v_i32mf2(dst + 6, 12 * sizeof(int32_t), _col6, avl_tail);
                vint32mf2_t _col7 = vle32_v_i32mf2(tm1 + avl_tail * 7, avl_tail);
                vsse32_v_i32mf2(dst + 7, 12 * sizeof(int32_t), _col7, avl_tail);
                vint32mf2_t _col8 = vle32_v_i32mf2(tm1 + avl_tail * 8, avl_tail);
                vsse32_v_i32mf2(dst + 8, 12 * sizeof(int32_t), _col8, avl_tail);
                vint32mf2_t _col9 = vle32_v_i32mf2(tm1 + avl_tail * 9, avl_tail);
                vsse32_v_i32mf2(dst + 9, 12 * sizeof(int32_t), _col9, avl_tail);
                vint32mf2_t _cola = vle32_v_i32mf2(tm1 + avl_tail * 10, avl_tail);
                vsse32_v_i32mf2(dst + 10, 12 * sizeof(int32_t), _cola, avl_tail);
                vint32mf2_t _colb = vle32_v_i32mf2(tm1 + avl_tail * 11, avl_tail);
                vsse32_v_i32mf2(dst + 11, 12 * sizeof(int32_t), _colb, avl_tail);

                dst += 12 * avl_tail;
                tm1 += n * avl_tail;
            }
        }
    }
    for (; t + 7 < n; t += 8) {
        const int32_t *tm1 = (const int32_t *)(b + t * vl);
        int ic = 0;
        for (; ic + packn - 1 < inc; ic += packn) {
            for (int i = 0; i < maxk; i++) {
                vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
                vsse32_v_i32mf2(dst, 8 * sizeof(int32_t), _col0, avl);
                vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
                vsse32_v_i32mf2(dst + 1, 8 * sizeof(int32_t), _col1, avl);
                vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
                vsse32_v_i32mf2(dst + 2, 8 * sizeof(int32_t), _col2, avl);
                vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
                vsse32_v_i32mf2(dst + 3, 8 * sizeof(int32_t), _col3, avl);
                vint32mf2_t _col4 = vle32_v_i32mf2(tm1 + avl * 4, avl);
                vsse32_v_i32mf2(dst + 4, 8 * sizeof(int32_t), _col4, avl);
                vint32mf2_t _col5 = vle32_v_i32mf2(tm1 + avl * 5, avl);
                vsse32_v_i32mf2(dst + 5, 8 * sizeof(int32_t), _col5, avl);
                vint32mf2_t _col6 = vle32_v_i32mf2(tm1 + avl * 6, avl);
                vsse32_v_i32mf2(dst + 6, 8 * sizeof(int32_t), _col6, avl);
                vint32mf2_t _col7 = vle32_v_i32mf2(tm1 + avl * 7, avl);
                vsse32_v_i32mf2(dst + 7, 8 * sizeof(int32_t), _col7, avl);

                dst += 8 * avl;
                tm1 += n * avl;
            }
        }
        if (ic < inc) {
            tm1 += t * (avl_tail - avl);
            for (int i = 0; i < maxk; i++) {
                vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl_tail);
                vsse32_v_i32mf2(dst, 8 * sizeof(int32_t), _col0, avl_tail);
                vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl_tail * 1, avl_tail);
                vsse32_v_i32mf2(dst + 1, 8 * sizeof(int32_t), _col1, avl_tail);
                vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl_tail * 2, avl_tail);
                vsse32_v_i32mf2(dst + 2, 8 * sizeof(int32_t), _col2, avl_tail);
                vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl_tail * 3, avl_tail);
                vsse32_v_i32mf2(dst + 3, 8 * sizeof(int32_t), _col3, avl_tail);
                vint32mf2_t _col4 = vle32_v_i32mf2(tm1 + avl_tail * 4, avl_tail);
                vsse32_v_i32mf2(dst + 4, 8 * sizeof(int32_t), _col4, avl_tail);
                vint32mf2_t _col5 = vle32_v_i32mf2(tm1 + avl_tail * 5, avl_tail);
                vsse32_v_i32mf2(dst + 5, 8 * sizeof(int32_t), _col5, avl_tail);
                vint32mf2_t _col6 = vle32_v_i32mf2(tm1 + avl_tail * 6, avl_tail);
                vsse32_v_i32mf2(dst + 6, 8 * sizeof(int32_t), _col6, avl_tail);
                vint32mf2_t _col7 = vle32_v_i32mf2(tm1 + avl_tail * 7, avl_tail);
                vsse32_v_i32mf2(dst + 7, 8 * sizeof(int32_t), _col7, avl_tail);

                dst += 8 * avl_tail;
                tm1 += n * avl_tail;
            }
        }
    }
    for (; t + 3 < n; t += 4) {
        const int32_t *tm1 = (const int32_t *)(b + t * vl);
        int ic = 0;
        for (; ic + packn - 1 < inc; ic += packn) {
            for (int i = 0; i < maxk; i++) {
                vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
                vsse32_v_i32mf2(dst, 4 * sizeof(int32_t), _col0, avl);
                vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
                vsse32_v_i32mf2(dst + 1, 4 * sizeof(int32_t), _col1, avl);
                vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
                vsse32_v_i32mf2(dst + 2, 4 * sizeof(int32_t), _col2, avl);
                vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
                vsse32_v_i32mf2(dst + 3, 4 * sizeof(int32_t), _col3, avl);

                dst += 4 * avl;
                tm1 += n * avl;
            }
        }
        if (ic < inc) {
            tm1 += t * (avl_tail - avl);
            for (int i = 0; i < maxk; i++) {
                vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl_tail);
                vsse32_v_i32mf2(dst, 4 * sizeof(int32_t), _col0, avl_tail);
                vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl_tail * 1, avl_tail);
                vsse32_v_i32mf2(dst + 1, 4 * sizeof(int32_t), _col1, avl_tail);
                vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl_tail * 2, avl_tail);
                vsse32_v_i32mf2(dst + 2, 4 * sizeof(int32_t), _col2, avl_tail);
                vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl_tail * 3, avl_tail);
                vsse32_v_i32mf2(dst + 3, 4 * sizeof(int32_t), _col3, avl_tail);

                dst += 4 * avl_tail;
                tm1 += n * avl_tail;
            }
        }
    }
    for (; t + 1 < n; t += 2) {
        const int32_t *tm1 = (const int32_t *)(b + t * vl);
        int ic = 0;
        for (; ic + packn - 1 < inc; ic += packn) {
            for (int i = 0; i < maxk; i++) {
                vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
                vsse32_v_i32mf2(dst, 2 * sizeof(int32_t), _col0, avl);
                vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
                vsse32_v_i32mf2(dst + 1, 2 * sizeof(int32_t), _col1, avl);
                dst += 2 * avl;
                tm1 += n * avl;
            }
        }
        if (ic < inc) {
            tm1 += t * (avl_tail - avl);
            for (int i = 0; i < maxk; i++) {
                vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl_tail);
                vsse32_v_i32mf2(dst, 2 * sizeof(int32_t), _col0, avl_tail);
                vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl_tail * 1, avl_tail);
                vsse32_v_i32mf2(dst + 1, 2 * sizeof(int32_t), _col1, avl_tail);
                dst += 2 * avl_tail;
                tm1 += n * avl_tail;
            }
        }
    }
    for (; t < n; t += 1) {
        const int32_t *tm1 = (const int32_t *)(b + t * vl);
        int ic = 0;
        for (; ic + packn - 1 < inc; ic += packn) {
            for (int i = 0; i < maxk; i++) {
                vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
                vse32_v_i32mf2(dst, _col0, avl);
                dst += 1 * avl;
                tm1 += n * avl;
            }
        }
        if (ic < inc) {
            tm1 += t * (avl_tail - avl);
            for (int i = 0; i < maxk; i++) {
                vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl_tail);
                vse32_v_i32mf2(dst, _col0, avl_tail);
                dst += 1 * avl_tail;
                tm1 += n * avl_tail;
            }
        }
    }
#endif
}

/**************************************************************
 * input—matrix: [k, n]
 * src: b   [k/packn, n, packn]
 * dst: sb  [n/8, k, 8]
 * Data arrangement: Z8 Z4 Z2 Z1
 **************************************************************/
void shl_rvv_reorder_input_z8_packn_fp32(float *b, float *sb, int k, int n, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    int t = 0;
    for (; t + 7 < n; t += 8) {
        const float *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
            vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
            vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + packn * 2, vl);
            vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + packn * 3, vl);
            vfloat32m1_t _tmp4 = vle32_v_f32m1(tm1 + packn * 4, vl);
            vfloat32m1_t _tmp5 = vle32_v_f32m1(tm1 + packn * 5, vl);
            vfloat32m1_t _tmp6 = vle32_v_f32m1(tm1 + packn * 6, vl);
            vfloat32m1_t _tmp7 = vle32_v_f32m1(tm1 + packn * 7, vl);
            vsseg8e32_v_f32m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7, vl);
            tm1 += n * packn;
            sb += 8 * packn;
        }
    }
    for (; t + 3 < n; t += 4) {
        const float *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
            vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
            vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + packn * 2, vl);
            vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + packn * 3, vl);
            vsseg4e32_v_f32m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, vl);
            tm1 += n * packn;
            sb += 4 * packn;
        }
    }
    for (; t + 1 < n; t += 2) {
        const float *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
            vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
            vsseg2e32_v_f32m1(sb, _tmp0, _tmp1, vl);
            tm1 += n * packn;
            sb += 2 * packn;
        }
    }
    for (; t < n; t++) {
        const float *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
            vse32_v_f32m1(sb, _tmp0, vl);
            tm1 += n * packn;
            sb += 1 * packn;
        }
    }
}

void shl_rvv_reorder_input_z8_packn_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    int t = 0;
    for (; t + 7 < n; t += 8) {
        const __fp16 *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
            vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + packn * 1, vl);
            vfloat16m1_t _tmp2 = vle16_v_f16m1(tm1 + packn * 2, vl);
            vfloat16m1_t _tmp3 = vle16_v_f16m1(tm1 + packn * 3, vl);
            vfloat16m1_t _tmp4 = vle16_v_f16m1(tm1 + packn * 4, vl);
            vfloat16m1_t _tmp5 = vle16_v_f16m1(tm1 + packn * 5, vl);
            vfloat16m1_t _tmp6 = vle16_v_f16m1(tm1 + packn * 6, vl);
            vfloat16m1_t _tmp7 = vle16_v_f16m1(tm1 + packn * 7, vl);
            vsseg8e16_v_f16m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7, vl);
            tm1 += n * packn;
            sb += 8 * packn;
        }
    }
    for (; t + 3 < n; t += 4) {
        const __fp16 *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
            vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + packn * 1, vl);
            vfloat16m1_t _tmp2 = vle16_v_f16m1(tm1 + packn * 2, vl);
            vfloat16m1_t _tmp3 = vle16_v_f16m1(tm1 + packn * 3, vl);
            vsseg4e16_v_f16m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, vl);
            tm1 += n * packn;
            sb += 4 * packn;
        }
    }
    for (; t + 1 < n; t += 2) {
        const __fp16 *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
            vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + packn * 1, vl);
            vsseg2e16_v_f16m1(sb, _tmp0, _tmp1, vl);
            tm1 += n * packn;
            sb += 2 * packn;
        }
    }
    for (; t < n; t++) {
        const __fp16 *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
            vse16_v_f16m1(sb, _tmp0, vl);
            tm1 += n * packn;
            sb += 1 * packn;
        }
    }
}

void shl_rvv_reorder_input_z8_packn_int8_dot(int8_t *b, int8_t *sb, int k, int n, int ldx)
{
#ifdef RVV_1_0_0
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8mf2(packn);
    int32_t *dst = (int32_t *)sb;

    int t = 0;
    /* 只适合 vlen=128，需要兼容 vlen
    for (; t + 7 < n; t += 8) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32m2_t _line0, _line1;
            vlseg2e32_v_i32m2(&_line0, &_line1, tm1, 8);
            vse32_v_i32m2(dst, _line0, 8);
            dst += 8;
            vse32_v_i32m2(dst, _line1, 8);
            dst += 8;
            tm1 += n * packn / 4;
        }
    }
    for (; t + 3 < n; t += 4) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32m1_t _line0, _line1;
            vlseg2e32_v_i32m1(&_line0, &_line1, tm1, 4);
            vse32_v_i32m1(dst, _line0, 4);
            dst += 4;
            vse32_v_i32m1(dst, _line1, 4);
            dst += 4;
            tm1 += n * packn / 4;
        }
    }
    for (; t + 1 < n; t += 2) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32m1_t _line0, _line1;
            vlseg2e32_v_i32m1(&_line0, &_line1, tm1, 2);
            vse32_v_i32m1(dst, _line0, 2);
            dst += 2;
            vse32_v_i32m1(dst, _line1, 2);
            dst += 2;
            tm1 += n * packn / 4;
        }
    }
    for (; t < n; t++) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32m1_t _line0, _line1;
            vlseg2e32_v_i32m1(&_line0, &_line1, tm1, 1);
            vse32_v_i32m1(dst, _line0, 1);
            dst += 1;
            vse32_v_i32m1(dst, _line1, 1);
            dst += 1;
            tm1 += n * packn / 4;
        }
    }
    */

    int avl = packn / 4;
    for (; t + 7 < n; t += 8) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 8 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 8 * sizeof(int32_t), _col1, avl);
            vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
            vsse32_v_i32mf2(dst + 2, 8 * sizeof(int32_t), _col2, avl);
            vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
            vsse32_v_i32mf2(dst + 3, 8 * sizeof(int32_t), _col3, avl);
            vint32mf2_t _col4 = vle32_v_i32mf2(tm1 + avl * 4, avl);
            vsse32_v_i32mf2(dst + 4, 8 * sizeof(int32_t), _col4, avl);
            vint32mf2_t _col5 = vle32_v_i32mf2(tm1 + avl * 5, avl);
            vsse32_v_i32mf2(dst + 5, 8 * sizeof(int32_t), _col5, avl);
            vint32mf2_t _col6 = vle32_v_i32mf2(tm1 + avl * 6, avl);
            vsse32_v_i32mf2(dst + 6, 8 * sizeof(int32_t), _col6, avl);
            vint32mf2_t _col7 = vle32_v_i32mf2(tm1 + avl * 7, avl);
            vsse32_v_i32mf2(dst + 7, 8 * sizeof(int32_t), _col7, avl);

            dst += 8 * avl;
            tm1 += n * avl;
        }
    }
    for (; t + 3 < n; t += 4) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 4 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 4 * sizeof(int32_t), _col1, avl);
            vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
            vsse32_v_i32mf2(dst + 2, 4 * sizeof(int32_t), _col2, avl);
            vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
            vsse32_v_i32mf2(dst + 3, 4 * sizeof(int32_t), _col3, avl);

            dst += 4 * avl;
            tm1 += n * avl;
        }
    }
    for (; t + 1 < n; t += 2) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 2 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 2 * sizeof(int32_t), _col1, avl);

            dst += 2 * avl;
            tm1 += n * avl;
        }
    }
    for (; t < n; t++) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vse32_v_i32mf2(dst, _col0, avl);

            dst += 1 * avl;
            tm1 += n * avl;
        }
    }
#endif
}

/**************************************************************
 * input—matrix: [k, n]
 * src: b   [k/packn/2, n, packn/2]
 * dst: sb  [n/8, k, 8]
 * Data arrangement: Z8 Z4 Z2 Z1
 **************************************************************/
void shl_rvv_reorder_input_z8_packn_int4(int8_t *b, int8_t *sb, int k, int n, int ldx)
{
#ifdef RVV_1_0_0
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2 / 2;
    const int vl = vsetvl_e8mf4(packn);
    int32_t *dst = (int32_t *)sb;

    int t = 0;
    int avl = packn / 4;
    for (; t + 7 < n; t += 8) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 8 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 8 * sizeof(int32_t), _col1, avl);
            vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
            vsse32_v_i32mf2(dst + 2, 8 * sizeof(int32_t), _col2, avl);
            vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
            vsse32_v_i32mf2(dst + 3, 8 * sizeof(int32_t), _col3, avl);
            vint32mf2_t _col4 = vle32_v_i32mf2(tm1 + avl * 4, avl);
            vsse32_v_i32mf2(dst + 4, 8 * sizeof(int32_t), _col4, avl);
            vint32mf2_t _col5 = vle32_v_i32mf2(tm1 + avl * 5, avl);
            vsse32_v_i32mf2(dst + 5, 8 * sizeof(int32_t), _col5, avl);
            vint32mf2_t _col6 = vle32_v_i32mf2(tm1 + avl * 6, avl);
            vsse32_v_i32mf2(dst + 6, 8 * sizeof(int32_t), _col6, avl);
            vint32mf2_t _col7 = vle32_v_i32mf2(tm1 + avl * 7, avl);
            vsse32_v_i32mf2(dst + 7, 8 * sizeof(int32_t), _col7, avl);

            dst += 8 * avl;
            tm1 += n * avl;
        }
    }
    for (; t + 3 < n; t += 4) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 4 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 4 * sizeof(int32_t), _col1, avl);
            vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
            vsse32_v_i32mf2(dst + 2, 4 * sizeof(int32_t), _col2, avl);
            vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
            vsse32_v_i32mf2(dst + 3, 4 * sizeof(int32_t), _col3, avl);

            dst += 4 * avl;
            tm1 += n * avl;
        }
    }
    for (; t + 1 < n; t += 2) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 2 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 2 * sizeof(int32_t), _col1, avl);

            dst += 2 * avl;
            tm1 += n * avl;
        }
    }
    for (; t < n; t++) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vse32_v_i32mf2(dst, _col0, avl);

            dst += 1 * avl;
            tm1 += n * avl;
        }
    }
#endif
}

/**************************************************************
 * input—matrix: [k, n]
 * src: b   [k/packn, n, packn]
 * dst: sb  [n/12, k, 12]
 * Data arrangement: Z12 Z8 Z4 Z2 Z1
 **************************************************************/
void shl_rvv_reorder_input_z12_packn_fp32(float *b, float *sb, int k, int n, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    int t = 0;
    for (; t + 11 < n; t += 12) {
        const float *tm1 = b + t * packn;  // start addr
        for (int q = 0; q < k / packn; q++) {
            vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
            vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
            vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + packn * 2, vl);
            vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + packn * 3, vl);
            vfloat32m1_t _tmp4 = vle32_v_f32m1(tm1 + packn * 4, vl);
            vfloat32m1_t _tmp5 = vle32_v_f32m1(tm1 + packn * 5, vl);
            vfloat32m1_t _tmp6 = vle32_v_f32m1(tm1 + packn * 6, vl);
            vfloat32m1_t _tmp7 = vle32_v_f32m1(tm1 + packn * 7, vl);
            vfloat32m1_t _tmp8 = vle32_v_f32m1(tm1 + packn * 8, vl);
            vfloat32m1_t _tmp9 = vle32_v_f32m1(tm1 + packn * 9, vl);
            vfloat32m1_t _tmp10 = vle32_v_f32m1(tm1 + packn * 10, vl);
            vfloat32m1_t _tmp11 = vle32_v_f32m1(tm1 + packn * 11, vl);

            vsse32_v_f32m1(sb, 12 * sizeof(float), _tmp0, vl);
            vsse32_v_f32m1(sb + 1, 12 * sizeof(float), _tmp1, vl);
            vsse32_v_f32m1(sb + 2, 12 * sizeof(float), _tmp2, vl);
            vsse32_v_f32m1(sb + 3, 12 * sizeof(float), _tmp3, vl);
            vsse32_v_f32m1(sb + 4, 12 * sizeof(float), _tmp4, vl);
            vsse32_v_f32m1(sb + 5, 12 * sizeof(float), _tmp5, vl);
            vsse32_v_f32m1(sb + 6, 12 * sizeof(float), _tmp6, vl);
            vsse32_v_f32m1(sb + 7, 12 * sizeof(float), _tmp7, vl);
            vsse32_v_f32m1(sb + 8, 12 * sizeof(float), _tmp8, vl);
            vsse32_v_f32m1(sb + 9, 12 * sizeof(float), _tmp9, vl);
            vsse32_v_f32m1(sb + 10, 12 * sizeof(float), _tmp10, vl);
            vsse32_v_f32m1(sb + 11, 12 * sizeof(float), _tmp11, vl);
            tm1 += n * packn;
            sb += 12 * packn;
        }
    }
    for (; t + 7 < n; t += 8) {
        const float *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
            vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
            vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + packn * 2, vl);
            vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + packn * 3, vl);
            vfloat32m1_t _tmp4 = vle32_v_f32m1(tm1 + packn * 4, vl);
            vfloat32m1_t _tmp5 = vle32_v_f32m1(tm1 + packn * 5, vl);
            vfloat32m1_t _tmp6 = vle32_v_f32m1(tm1 + packn * 6, vl);
            vfloat32m1_t _tmp7 = vle32_v_f32m1(tm1 + packn * 7, vl);
            vsseg8e32_v_f32m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7, vl);
            tm1 += n * packn;
            sb += 8 * packn;
        }
    }
    for (; t + 3 < n; t += 4) {
        const float *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
            vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
            vfloat32m1_t _tmp2 = vle32_v_f32m1(tm1 + packn * 2, vl);
            vfloat32m1_t _tmp3 = vle32_v_f32m1(tm1 + packn * 3, vl);
            vsseg4e32_v_f32m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, vl);
            tm1 += n * packn;
            sb += 4 * packn;
        }
    }
    for (; t + 1 < n; t += 2) {
        const float *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
            vfloat32m1_t _tmp1 = vle32_v_f32m1(tm1 + packn * 1, vl);
            vsseg2e32_v_f32m1(sb, _tmp0, _tmp1, vl);
            tm1 += n * packn;
            sb += 2 * packn;
        }
    }
    for (; t < n; t++) {
        const float *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat32m1_t _tmp0 = vle32_v_f32m1(tm1, vl);
            vse32_v_f32m1(sb, _tmp0, vl);
            tm1 += n * packn;
            sb += 1 * packn;
        }
    }
}

void shl_rvv_reorder_input_z12_packn_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    int t = 0;
    for (; t + 11 < n; t += 12) {
        const __fp16 *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
            vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + packn * 1, vl);
            vfloat16m1_t _tmp2 = vle16_v_f16m1(tm1 + packn * 2, vl);
            vfloat16m1_t _tmp3 = vle16_v_f16m1(tm1 + packn * 3, vl);
            vfloat16m1_t _tmp4 = vle16_v_f16m1(tm1 + packn * 4, vl);
            vfloat16m1_t _tmp5 = vle16_v_f16m1(tm1 + packn * 5, vl);
            vfloat16m1_t _tmp6 = vle16_v_f16m1(tm1 + packn * 6, vl);
            vfloat16m1_t _tmp7 = vle16_v_f16m1(tm1 + packn * 7, vl);
            vfloat16m1_t _tmp8 = vle16_v_f16m1(tm1 + packn * 8, vl);
            vfloat16m1_t _tmp9 = vle16_v_f16m1(tm1 + packn * 9, vl);
            vfloat16m1_t _tmp10 = vle16_v_f16m1(tm1 + packn * 10, vl);
            vfloat16m1_t _tmp11 = vle16_v_f16m1(tm1 + packn * 11, vl);

            vsse16_v_f16m1(sb, 12 * sizeof(__fp16), _tmp0, vl);
            vsse16_v_f16m1(sb + 1, 12 * sizeof(__fp16), _tmp1, vl);
            vsse16_v_f16m1(sb + 2, 12 * sizeof(__fp16), _tmp2, vl);
            vsse16_v_f16m1(sb + 3, 12 * sizeof(__fp16), _tmp3, vl);
            vsse16_v_f16m1(sb + 4, 12 * sizeof(__fp16), _tmp4, vl);
            vsse16_v_f16m1(sb + 5, 12 * sizeof(__fp16), _tmp5, vl);
            vsse16_v_f16m1(sb + 6, 12 * sizeof(__fp16), _tmp6, vl);
            vsse16_v_f16m1(sb + 7, 12 * sizeof(__fp16), _tmp7, vl);
            vsse16_v_f16m1(sb + 8, 12 * sizeof(__fp16), _tmp8, vl);
            vsse16_v_f16m1(sb + 9, 12 * sizeof(__fp16), _tmp9, vl);
            vsse16_v_f16m1(sb + 10, 12 * sizeof(__fp16), _tmp10, vl);
            vsse16_v_f16m1(sb + 11, 12 * sizeof(__fp16), _tmp11, vl);
            tm1 += n * packn;
            sb += 12 * packn;
        }
    }
    for (; t + 7 < n; t += 8) {
        const __fp16 *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
            vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + packn * 1, vl);
            vfloat16m1_t _tmp2 = vle16_v_f16m1(tm1 + packn * 2, vl);
            vfloat16m1_t _tmp3 = vle16_v_f16m1(tm1 + packn * 3, vl);
            vfloat16m1_t _tmp4 = vle16_v_f16m1(tm1 + packn * 4, vl);
            vfloat16m1_t _tmp5 = vle16_v_f16m1(tm1 + packn * 5, vl);
            vfloat16m1_t _tmp6 = vle16_v_f16m1(tm1 + packn * 6, vl);
            vfloat16m1_t _tmp7 = vle16_v_f16m1(tm1 + packn * 7, vl);
            vsseg8e16_v_f16m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7, vl);
            tm1 += n * packn;
            sb += 8 * packn;
        }
    }
    for (; t + 3 < n; t += 4) {
        const __fp16 *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
            vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + packn * 1, vl);
            vfloat16m1_t _tmp2 = vle16_v_f16m1(tm1 + packn * 2, vl);
            vfloat16m1_t _tmp3 = vle16_v_f16m1(tm1 + packn * 3, vl);
            vsseg4e16_v_f16m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, vl);
            tm1 += n * packn;
            sb += 4 * packn;
        }
    }
    for (; t + 1 < n; t += 2) {
        const __fp16 *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
            vfloat16m1_t _tmp1 = vle16_v_f16m1(tm1 + packn * 1, vl);
            vsseg2e16_v_f16m1(sb, _tmp0, _tmp1, vl);
            tm1 += n * packn;
            sb += 2 * packn;
        }
    }
    for (; t < n; t++) {
        const __fp16 *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vfloat16m1_t _tmp0 = vle16_v_f16m1(tm1, vl);
            vse16_v_f16m1(sb, _tmp0, vl);
            tm1 += n * packn;
            sb += 1 * packn;
        }
    }
}

void shl_rvv_reorder_input_z4_packn_int8(int8_t *b, int8_t *sb, int k, int n, int ldx)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int t = 0;
    for (; t + 3 < n; t += 4) {
        const int8_t *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vint8m1_t _tmp0 = vle8_v_i8m1(tm1, vl);
            vint8m1_t _tmp1 = vle8_v_i8m1(tm1 + packn * 1, vl);
            vint8m1_t _tmp2 = vle8_v_i8m1(tm1 + packn * 2, vl);
            vint8m1_t _tmp3 = vle8_v_i8m1(tm1 + packn * 3, vl);
            vsseg4e8_v_i8m1(sb, _tmp0, _tmp1, _tmp2, _tmp3, vl);
            tm1 += n * packn;
            sb += 4 * packn;
        }
    }
    for (; t + 1 < n; t += 2) {
        const int8_t *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vint8m1_t _tmp0 = vle8_v_i8m1(tm1, vl);
            vint8m1_t _tmp1 = vle8_v_i8m1(tm1 + packn * 1, vl);
            vsseg2e8_v_i8m1(sb, _tmp0, _tmp1, vl);
            tm1 += n * packn;
            sb += 2 * packn;
        }
    }
    for (; t < n; t++) {
        const int8_t *tm1 = b + t * packn;
        for (int q = 0; q < k / packn; q++) {
            vint8m1_t _tmp0 = vle8_v_i8m1(tm1, vl);
            vse8_v_i8m1(sb, _tmp0, vl);
            tm1 += n * packn;
            sb += 1 * packn;
        }
    }
}

void shl_rvv_reorder_input_z12_packn_int8_dot(int8_t *b, int8_t *sb, int k, int n, int ldx)
{
#ifdef RVV_1_0_0
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8mf2(packn);
    int32_t *dst = (int32_t *)sb;

    int t = 0;
    int avl = packn / 4;
    for (; t + 11 < n; t += 12) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 12 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 12 * sizeof(int32_t), _col1, avl);
            vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
            vsse32_v_i32mf2(dst + 2, 12 * sizeof(int32_t), _col2, avl);
            vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
            vsse32_v_i32mf2(dst + 3, 12 * sizeof(int32_t), _col3, avl);
            vint32mf2_t _col4 = vle32_v_i32mf2(tm1 + avl * 4, avl);
            vsse32_v_i32mf2(dst + 4, 12 * sizeof(int32_t), _col4, avl);
            vint32mf2_t _col5 = vle32_v_i32mf2(tm1 + avl * 5, avl);
            vsse32_v_i32mf2(dst + 5, 12 * sizeof(int32_t), _col5, avl);
            vint32mf2_t _col6 = vle32_v_i32mf2(tm1 + avl * 6, avl);
            vsse32_v_i32mf2(dst + 6, 12 * sizeof(int32_t), _col6, avl);
            vint32mf2_t _col7 = vle32_v_i32mf2(tm1 + avl * 7, avl);
            vsse32_v_i32mf2(dst + 7, 12 * sizeof(int32_t), _col7, avl);
            vint32mf2_t _col8 = vle32_v_i32mf2(tm1 + avl * 8, avl);
            vsse32_v_i32mf2(dst + 8, 12 * sizeof(int32_t), _col8, avl);
            vint32mf2_t _col9 = vle32_v_i32mf2(tm1 + avl * 9, avl);
            vsse32_v_i32mf2(dst + 9, 12 * sizeof(int32_t), _col9, avl);
            vint32mf2_t _cola = vle32_v_i32mf2(tm1 + avl * 10, avl);
            vsse32_v_i32mf2(dst + 10, 12 * sizeof(int32_t), _cola, avl);
            vint32mf2_t _colb = vle32_v_i32mf2(tm1 + avl * 11, avl);
            vsse32_v_i32mf2(dst + 11, 12 * sizeof(int32_t), _colb, avl);

            dst += 12 * avl;
            tm1 += n * avl;
        }
    }
    for (; t + 7 < n; t += 8) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 8 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 8 * sizeof(int32_t), _col1, avl);
            vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
            vsse32_v_i32mf2(dst + 2, 8 * sizeof(int32_t), _col2, avl);
            vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
            vsse32_v_i32mf2(dst + 3, 8 * sizeof(int32_t), _col3, avl);
            vint32mf2_t _col4 = vle32_v_i32mf2(tm1 + avl * 4, avl);
            vsse32_v_i32mf2(dst + 4, 8 * sizeof(int32_t), _col4, avl);
            vint32mf2_t _col5 = vle32_v_i32mf2(tm1 + avl * 5, avl);
            vsse32_v_i32mf2(dst + 5, 8 * sizeof(int32_t), _col5, avl);
            vint32mf2_t _col6 = vle32_v_i32mf2(tm1 + avl * 6, avl);
            vsse32_v_i32mf2(dst + 6, 8 * sizeof(int32_t), _col6, avl);
            vint32mf2_t _col7 = vle32_v_i32mf2(tm1 + avl * 7, avl);
            vsse32_v_i32mf2(dst + 7, 8 * sizeof(int32_t), _col7, avl);

            dst += 8 * avl;
            tm1 += n * avl;
        }
    }
    for (; t + 3 < n; t += 4) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 4 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 4 * sizeof(int32_t), _col1, avl);
            vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
            vsse32_v_i32mf2(dst + 2, 4 * sizeof(int32_t), _col2, avl);
            vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
            vsse32_v_i32mf2(dst + 3, 4 * sizeof(int32_t), _col3, avl);

            dst += 4 * avl;
            tm1 += n * avl;
        }
    }
    for (; t + 1 < n; t += 2) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 2 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 2 * sizeof(int32_t), _col1, avl);

            dst += 2 * avl;
            tm1 += n * avl;
        }
    }
    for (; t < n; t++) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vse32_v_i32mf2(dst, _col0, avl);

            dst += 1 * avl;
            tm1 += n * avl;
        }
    }
#endif
}

/**************************************************************
 * input—matrix: [k, n]
 * src: b   [k/packn/2, n, packn/2]
 * dst: sb  [n/12, k, 12]
 * Data arrangement: Z12 Z8 Z4 Z2 Z1
 **************************************************************/
void shl_rvv_reorder_input_z12_packn_int4(int8_t *b, int8_t *sb, int k, int n, int ldx)
{
#ifdef RVV_1_0_0
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2 / 2;
    const int vl = vsetvl_e8mf4(packn);
    int32_t *dst = (int32_t *)sb;

    int t = 0;
    int avl = packn / 4;
    for (; t + 11 < n; t += 12) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 12 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 12 * sizeof(int32_t), _col1, avl);
            vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
            vsse32_v_i32mf2(dst + 2, 12 * sizeof(int32_t), _col2, avl);
            vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
            vsse32_v_i32mf2(dst + 3, 12 * sizeof(int32_t), _col3, avl);
            vint32mf2_t _col4 = vle32_v_i32mf2(tm1 + avl * 4, avl);
            vsse32_v_i32mf2(dst + 4, 12 * sizeof(int32_t), _col4, avl);
            vint32mf2_t _col5 = vle32_v_i32mf2(tm1 + avl * 5, avl);
            vsse32_v_i32mf2(dst + 5, 12 * sizeof(int32_t), _col5, avl);
            vint32mf2_t _col6 = vle32_v_i32mf2(tm1 + avl * 6, avl);
            vsse32_v_i32mf2(dst + 6, 12 * sizeof(int32_t), _col6, avl);
            vint32mf2_t _col7 = vle32_v_i32mf2(tm1 + avl * 7, avl);
            vsse32_v_i32mf2(dst + 7, 12 * sizeof(int32_t), _col7, avl);
            vint32mf2_t _col8 = vle32_v_i32mf2(tm1 + avl * 8, avl);
            vsse32_v_i32mf2(dst + 8, 12 * sizeof(int32_t), _col8, avl);
            vint32mf2_t _col9 = vle32_v_i32mf2(tm1 + avl * 9, avl);
            vsse32_v_i32mf2(dst + 9, 12 * sizeof(int32_t), _col9, avl);
            vint32mf2_t _cola = vle32_v_i32mf2(tm1 + avl * 10, avl);
            vsse32_v_i32mf2(dst + 10, 12 * sizeof(int32_t), _cola, avl);
            vint32mf2_t _colb = vle32_v_i32mf2(tm1 + avl * 11, avl);
            vsse32_v_i32mf2(dst + 11, 12 * sizeof(int32_t), _colb, avl);

            dst += 12 * avl;
            tm1 += n * avl;
        }
    }
    for (; t + 7 < n; t += 8) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 8 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 8 * sizeof(int32_t), _col1, avl);
            vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
            vsse32_v_i32mf2(dst + 2, 8 * sizeof(int32_t), _col2, avl);
            vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
            vsse32_v_i32mf2(dst + 3, 8 * sizeof(int32_t), _col3, avl);
            vint32mf2_t _col4 = vle32_v_i32mf2(tm1 + avl * 4, avl);
            vsse32_v_i32mf2(dst + 4, 8 * sizeof(int32_t), _col4, avl);
            vint32mf2_t _col5 = vle32_v_i32mf2(tm1 + avl * 5, avl);
            vsse32_v_i32mf2(dst + 5, 8 * sizeof(int32_t), _col5, avl);
            vint32mf2_t _col6 = vle32_v_i32mf2(tm1 + avl * 6, avl);
            vsse32_v_i32mf2(dst + 6, 8 * sizeof(int32_t), _col6, avl);
            vint32mf2_t _col7 = vle32_v_i32mf2(tm1 + avl * 7, avl);
            vsse32_v_i32mf2(dst + 7, 8 * sizeof(int32_t), _col7, avl);

            dst += 8 * avl;
            tm1 += n * avl;
        }
    }
    for (; t + 3 < n; t += 4) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 4 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 4 * sizeof(int32_t), _col1, avl);
            vint32mf2_t _col2 = vle32_v_i32mf2(tm1 + avl * 2, avl);
            vsse32_v_i32mf2(dst + 2, 4 * sizeof(int32_t), _col2, avl);
            vint32mf2_t _col3 = vle32_v_i32mf2(tm1 + avl * 3, avl);
            vsse32_v_i32mf2(dst + 3, 4 * sizeof(int32_t), _col3, avl);

            dst += 4 * avl;
            tm1 += n * avl;
        }
    }
    for (; t + 1 < n; t += 2) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vsse32_v_i32mf2(dst, 2 * sizeof(int32_t), _col0, avl);
            vint32mf2_t _col1 = vle32_v_i32mf2(tm1 + avl * 1, avl);
            vsse32_v_i32mf2(dst + 1, 2 * sizeof(int32_t), _col1, avl);

            dst += 2 * avl;
            tm1 += n * avl;
        }
    }
    for (; t < n; t++) {
        const int32_t *tm1 = (const int32_t *)(b + t * packn);

        for (int q = 0; q < k / packn; q++) {
            vint32mf2_t _col0 = vle32_v_i32mf2(tm1, avl);
            vse32_v_i32mf2(dst, _col0, avl);

            dst += 1 * avl;
            tm1 += n * avl;
        }
    }
#endif
}

/*************************************************************
 * src: [M_BLOCK, K_BLOCK]
 * dst: [M_BLOCK/m_blk, K_BLOCK, m_blk]
 * m_blk: 12/8/4/2/1
 ************************************************************/
static inline void reorder_a_12xk_fp32(float *src, float *dst, int M_BLOCK, int K_BLOCK, int lda)
{
    int i = 0;
    for (; i + 11 < M_BLOCK; i += 12) {
        float *s_ptr = src + i * lda;
        float *d_ptr = dst + i * K_BLOCK;
        int stride = 12 * sizeof(float);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e32m2(K_BLOCK - c);
            vfloat32m2_t _s0 = vle32_v_f32m2(s_ptr, vl);
            vfloat32m2_t _s1 = vle32_v_f32m2(s_ptr + lda, vl);
            vfloat32m2_t _s2 = vle32_v_f32m2(s_ptr + lda * 2, vl);
            vfloat32m2_t _s3 = vle32_v_f32m2(s_ptr + lda * 3, vl);
            vfloat32m2_t _s4 = vle32_v_f32m2(s_ptr + lda * 4, vl);
            vfloat32m2_t _s5 = vle32_v_f32m2(s_ptr + lda * 5, vl);
            vfloat32m2_t _s6 = vle32_v_f32m2(s_ptr + lda * 6, vl);
            vfloat32m2_t _s7 = vle32_v_f32m2(s_ptr + lda * 7, vl);
            vfloat32m2_t _s8 = vle32_v_f32m2(s_ptr + lda * 8, vl);
            vfloat32m2_t _s9 = vle32_v_f32m2(s_ptr + lda * 9, vl);
            vfloat32m2_t _s10 = vle32_v_f32m2(s_ptr + lda * 10, vl);
            vfloat32m2_t _s11 = vle32_v_f32m2(s_ptr + lda * 11, vl);
            vsse32_v_f32m2(d_ptr, stride, _s0, vl);
            vsse32_v_f32m2(d_ptr + 1, stride, _s1, vl);
            vsse32_v_f32m2(d_ptr + 2, stride, _s2, vl);
            vsse32_v_f32m2(d_ptr + 3, stride, _s3, vl);
            vsse32_v_f32m2(d_ptr + 4, stride, _s4, vl);
            vsse32_v_f32m2(d_ptr + 5, stride, _s5, vl);
            vsse32_v_f32m2(d_ptr + 6, stride, _s6, vl);
            vsse32_v_f32m2(d_ptr + 7, stride, _s7, vl);
            vsse32_v_f32m2(d_ptr + 8, stride, _s8, vl);
            vsse32_v_f32m2(d_ptr + 9, stride, _s9, vl);
            vsse32_v_f32m2(d_ptr + 10, stride, _s10, vl);
            vsse32_v_f32m2(d_ptr + 11, stride, _s11, vl);
            s_ptr += vl;
            d_ptr += vl * 12;
            c += vl;
        }
    }
    for (; i + 7 < M_BLOCK; i += 8) {
        float *s_ptr = src + i * lda;
        float *d_ptr = dst + i * K_BLOCK;
        int stride = 8 * sizeof(float);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e32m4(K_BLOCK - c);
            vfloat32m4_t _s0 = vle32_v_f32m4(s_ptr, vl);
            vfloat32m4_t _s1 = vle32_v_f32m4(s_ptr + lda, vl);
            vfloat32m4_t _s2 = vle32_v_f32m4(s_ptr + lda * 2, vl);
            vfloat32m4_t _s3 = vle32_v_f32m4(s_ptr + lda * 3, vl);
            vfloat32m4_t _s4 = vle32_v_f32m4(s_ptr + lda * 4, vl);
            vfloat32m4_t _s5 = vle32_v_f32m4(s_ptr + lda * 5, vl);
            vfloat32m4_t _s6 = vle32_v_f32m4(s_ptr + lda * 6, vl);
            vfloat32m4_t _s7 = vle32_v_f32m4(s_ptr + lda * 7, vl);
            vsse32_v_f32m4(d_ptr, stride, _s0, vl);
            vsse32_v_f32m4(d_ptr + 1, stride, _s1, vl);
            vsse32_v_f32m4(d_ptr + 2, stride, _s2, vl);
            vsse32_v_f32m4(d_ptr + 3, stride, _s3, vl);
            vsse32_v_f32m4(d_ptr + 4, stride, _s4, vl);
            vsse32_v_f32m4(d_ptr + 5, stride, _s5, vl);
            vsse32_v_f32m4(d_ptr + 6, stride, _s6, vl);
            vsse32_v_f32m4(d_ptr + 7, stride, _s7, vl);
            s_ptr += vl;
            d_ptr += vl * 8;
            c += vl;
        }
    }
    for (; i + 3 < M_BLOCK; i += 4) {
        float *s_ptr = src + i * lda;
        float *d_ptr = dst + i * K_BLOCK;
        int stride = 4 * sizeof(float);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e32m4(K_BLOCK - c);
            vfloat32m4_t _s0 = vle32_v_f32m4(s_ptr, vl);
            vfloat32m4_t _s1 = vle32_v_f32m4(s_ptr + lda, vl);
            vfloat32m4_t _s2 = vle32_v_f32m4(s_ptr + lda * 2, vl);
            vfloat32m4_t _s3 = vle32_v_f32m4(s_ptr + lda * 3, vl);
            vsse32_v_f32m4(d_ptr, stride, _s0, vl);
            vsse32_v_f32m4(d_ptr + 1, stride, _s1, vl);
            vsse32_v_f32m4(d_ptr + 2, stride, _s2, vl);
            vsse32_v_f32m4(d_ptr + 3, stride, _s3, vl);
            s_ptr += vl;
            d_ptr += vl * 4;
            c += vl;
        }
    }
    for (; i + 1 < M_BLOCK; i += 2) {
        float *s_ptr = src + i * lda;
        float *d_ptr = dst + i * K_BLOCK;
        int stride = 2 * sizeof(float);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e32m4(K_BLOCK - c);
            vfloat32m4_t _s0 = vle32_v_f32m4(s_ptr, vl);
            vfloat32m4_t _s1 = vle32_v_f32m4(s_ptr + lda, vl);
            vsse32_v_f32m4(d_ptr, stride, _s0, vl);
            vsse32_v_f32m4(d_ptr + 1, stride, _s1, vl);
            s_ptr += vl;
            d_ptr += vl * 2;
            c += vl;
        }
    }
    for (; i < M_BLOCK; i++) {
        float *s_ptr = src + i * lda;
        float *d_ptr = dst + i * K_BLOCK;
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e32m4(K_BLOCK - c);
            vfloat32m4_t _src = vle32_v_f32m4(s_ptr, vl);
            vse32_v_f32m4(d_ptr, _src, vl);
            s_ptr += vl;
            d_ptr += vl;
            c += vl;
        }
    }
}

/*************************************************************
 * src: [m, k]
 * dst: [m/m_blk, k/k_blk, m_blk/12, 12, k_blk]
 * m_blk: M_BLK, M_tail
 * k_blk: K_BLK, K_tail
 ************************************************************/
void shl_rvv_reorder_a_block_12xk_fp32(float *src, float *dst, int m, int k, const int M_BLK,
                                       const int K_BLK)
{
    int m_block = M_BLK;
    int m_idx = 0;
    while (m_idx < m) {
        if (m - m_idx < m_block) {
            m_block = m - m_idx;
        }
        int k_block = K_BLK;
        int k_idx = 0;
        while (k_idx < k) {
            if (k - k_idx < k_block) {
                k_block = k - k_idx;
            }
            float *s_ptr = src + m_idx * k + k_idx;
            float *d_ptr = dst + m_idx * k + k_idx * m_block;
            reorder_a_12xk_fp32(s_ptr, d_ptr, m_block, k_block, k);
            k_idx += k_block;
        }
        m_idx += m_block;
    }
}

/*************************************************************
 * src: [M_BLOCK, K_BLOCK]
 * dst: [M_BLOCK/m_blk, K_BLOCK, m_blk]
 * m_blk: 12/8/4/2/1
 ************************************************************/
static inline void reorder_a_12xk_fp16(__fp16 *src, __fp16 *dst, int M_BLOCK, int K_BLOCK, int lda)
{
    int i = 0;
    for (; i + 11 < M_BLOCK; i += 12) {
        __fp16 *s_ptr = src + i * lda;
        __fp16 *d_ptr = dst + i * K_BLOCK;
        int stride = 12 * sizeof(__fp16);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e16m2(K_BLOCK - c);
            vfloat16m2_t _s0 = vle16_v_f16m2(s_ptr, vl);
            vfloat16m2_t _s1 = vle16_v_f16m2(s_ptr + lda, vl);
            vfloat16m2_t _s2 = vle16_v_f16m2(s_ptr + lda * 2, vl);
            vfloat16m2_t _s3 = vle16_v_f16m2(s_ptr + lda * 3, vl);
            vfloat16m2_t _s4 = vle16_v_f16m2(s_ptr + lda * 4, vl);
            vfloat16m2_t _s5 = vle16_v_f16m2(s_ptr + lda * 5, vl);
            vfloat16m2_t _s6 = vle16_v_f16m2(s_ptr + lda * 6, vl);
            vfloat16m2_t _s7 = vle16_v_f16m2(s_ptr + lda * 7, vl);
            vfloat16m2_t _s8 = vle16_v_f16m2(s_ptr + lda * 8, vl);
            vfloat16m2_t _s9 = vle16_v_f16m2(s_ptr + lda * 9, vl);
            vfloat16m2_t _s10 = vle16_v_f16m2(s_ptr + lda * 10, vl);
            vfloat16m2_t _s11 = vle16_v_f16m2(s_ptr + lda * 11, vl);
            vsse16_v_f16m2(d_ptr, stride, _s0, vl);
            vsse16_v_f16m2(d_ptr + 1, stride, _s1, vl);
            vsse16_v_f16m2(d_ptr + 2, stride, _s2, vl);
            vsse16_v_f16m2(d_ptr + 3, stride, _s3, vl);
            vsse16_v_f16m2(d_ptr + 4, stride, _s4, vl);
            vsse16_v_f16m2(d_ptr + 5, stride, _s5, vl);
            vsse16_v_f16m2(d_ptr + 6, stride, _s6, vl);
            vsse16_v_f16m2(d_ptr + 7, stride, _s7, vl);
            vsse16_v_f16m2(d_ptr + 8, stride, _s8, vl);
            vsse16_v_f16m2(d_ptr + 9, stride, _s9, vl);
            vsse16_v_f16m2(d_ptr + 10, stride, _s10, vl);
            vsse16_v_f16m2(d_ptr + 11, stride, _s11, vl);
            s_ptr += vl;
            d_ptr += vl * 12;
            c += vl;
        }
    }
    for (; i + 7 < M_BLOCK; i += 8) {
        __fp16 *s_ptr = src + i * lda;
        __fp16 *d_ptr = dst + i * K_BLOCK;
        int stride = 8 * sizeof(__fp16);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e16m4(K_BLOCK - c);
            vfloat16m4_t _s0 = vle16_v_f16m4(s_ptr, vl);
            vfloat16m4_t _s1 = vle16_v_f16m4(s_ptr + lda, vl);
            vfloat16m4_t _s2 = vle16_v_f16m4(s_ptr + lda * 2, vl);
            vfloat16m4_t _s3 = vle16_v_f16m4(s_ptr + lda * 3, vl);
            vfloat16m4_t _s4 = vle16_v_f16m4(s_ptr + lda * 4, vl);
            vfloat16m4_t _s5 = vle16_v_f16m4(s_ptr + lda * 5, vl);
            vfloat16m4_t _s6 = vle16_v_f16m4(s_ptr + lda * 6, vl);
            vfloat16m4_t _s7 = vle16_v_f16m4(s_ptr + lda * 7, vl);
            vsse16_v_f16m4(d_ptr, stride, _s0, vl);
            vsse16_v_f16m4(d_ptr + 1, stride, _s1, vl);
            vsse16_v_f16m4(d_ptr + 2, stride, _s2, vl);
            vsse16_v_f16m4(d_ptr + 3, stride, _s3, vl);
            vsse16_v_f16m4(d_ptr + 4, stride, _s4, vl);
            vsse16_v_f16m4(d_ptr + 5, stride, _s5, vl);
            vsse16_v_f16m4(d_ptr + 6, stride, _s6, vl);
            vsse16_v_f16m4(d_ptr + 7, stride, _s7, vl);
            s_ptr += vl;
            d_ptr += vl * 8;
            c += vl;
        }
    }
    for (; i + 3 < M_BLOCK; i += 4) {
        __fp16 *s_ptr = src + i * lda;
        __fp16 *d_ptr = dst + i * K_BLOCK;
        int stride = 4 * sizeof(__fp16);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e16m4(K_BLOCK - c);
            vfloat16m4_t _s0 = vle16_v_f16m4(s_ptr, vl);
            vfloat16m4_t _s1 = vle16_v_f16m4(s_ptr + lda, vl);
            vfloat16m4_t _s2 = vle16_v_f16m4(s_ptr + lda * 2, vl);
            vfloat16m4_t _s3 = vle16_v_f16m4(s_ptr + lda * 3, vl);
            vsse16_v_f16m4(d_ptr, stride, _s0, vl);
            vsse16_v_f16m4(d_ptr + 1, stride, _s1, vl);
            vsse16_v_f16m4(d_ptr + 2, stride, _s2, vl);
            vsse16_v_f16m4(d_ptr + 3, stride, _s3, vl);
            s_ptr += vl;
            d_ptr += vl * 4;
            c += vl;
        }
    }
    for (; i + 1 < M_BLOCK; i += 2) {
        __fp16 *s_ptr = src + i * lda;
        __fp16 *d_ptr = dst + i * K_BLOCK;
        int stride = 2 * sizeof(__fp16);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e16m4(K_BLOCK - c);
            vfloat16m4_t _s0 = vle16_v_f16m4(s_ptr, vl);
            vfloat16m4_t _s1 = vle16_v_f16m4(s_ptr + lda, vl);
            vsse16_v_f16m4(d_ptr, stride, _s0, vl);
            vsse16_v_f16m4(d_ptr + 1, stride, _s1, vl);
            s_ptr += vl;
            d_ptr += vl * 2;
            c += vl;
        }
    }
    for (; i < M_BLOCK; i++) {
        __fp16 *s_ptr = src + i * lda;
        __fp16 *d_ptr = dst + i * K_BLOCK;
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e16m4(K_BLOCK - c);
            vfloat16m4_t _src = vle16_v_f16m4(s_ptr, vl);
            vse16_v_f16m4(d_ptr, _src, vl);
            s_ptr += vl;
            d_ptr += vl;
            c += vl;
        }
    }
}

/*************************************************************
 * src: [m, k]
 * dst: [m/m_blk, k/k_blk, m_blk/12, 12, k_blk]
 * m_blk: M_BLK, M_tail
 * k_blk: K_BLK, K_tail
 ************************************************************/
void shl_rvv_reorder_a_block_12xk_fp16(__fp16 *src, __fp16 *dst, int m, int k, const int M_BLK,
                                       const int K_BLK)
{
    int m_block = M_BLK;
    int m_idx = 0;
    while (m_idx < m) {
        if (m - m_idx < m_block) {
            m_block = m - m_idx;
        }
        int k_block = K_BLK;
        int k_idx = 0;
        while (k_idx < k) {
            if (k - k_idx < k_block) {
                k_block = k - k_idx;
            }
            __fp16 *s_ptr = src + m_idx * k + k_idx;
            __fp16 *d_ptr = dst + m_idx * k + k_idx * m_block;
            reorder_a_12xk_fp16(s_ptr, d_ptr, m_block, k_block, k);
            k_idx += k_block;
        }
        m_idx += m_block;
    }
}

/*************************************************************
 * packn = vlenb / sizeof(float)
 * src: [K_BLOCK, N_BLOCK]
 * dst: [N_BLOCK/n_blk, K_BLOCK, n_blk]
 * n_blk: pack2n/packn/n_tail
 ************************************************************/
static inline void reorder_b_pack2nxk_fp32(float *src, float *dst, int N_BLOCK, int K_BLOCK,
                                           int ldb)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int pack2n = packn * 2;
    int vl = vsetvl_e32m2(pack2n);

    int j = 0;
    for (; j + pack2n - 1 < N_BLOCK; j += pack2n) {
        float *s_ptr = src + j;
        for (int c = 0; c < K_BLOCK; c++) {
            vfloat32m2_t _src = vle32_v_f32m2(s_ptr, vl);
            vse32_v_f32m2(dst, _src, vl);
            s_ptr += ldb;
            dst += vl;
        }
    }
    while (j < N_BLOCK) {
        vl = vsetvl_e32m1(N_BLOCK - j);
        float *s_ptr = src + j;
        for (int c = 0; c < K_BLOCK; c++) {
            vfloat32m1_t _src = vle32_v_f32m1(s_ptr, vl);
            vse32_v_f32m1(dst, _src, vl);
            s_ptr += ldb;
            dst += vl;
        }
        j += vl;
    }
}

/*************************************************************
 * packn = vlenb / sizeof(float)
 * src: [k, n]
 * dst: [n/n_blk, k/k_blk, n_blk/pack2n, k_blk, pack2n]
 * n_blk: N_BLK, N_tail
 * k_blk: K_BLK, K_tail
 ************************************************************/
void shl_rvv_reorder_b_block_pack2nxk_fp32(float *src, float *dst, int k, int n, const int K_BLK,
                                           const int N_BLK)
{
    int k_block = K_BLK;
    int k_idx = 0;
    while (k_idx < k) {
        if (k - k_idx < k_block) {
            k_block = k - k_idx;
        }
        int n_block = N_BLK;
        int n_idx = 0;
        while (n_idx < n) {
            if (n - n_idx < n_block) {
                n_block = n - n_idx;
            }
            float *s_ptr = src + k_idx * n + n_idx;
            float *d_ptr = dst + n_idx * k + k_idx * n_block;
            reorder_b_pack2nxk_fp32(s_ptr, d_ptr, n_block, k_block, n);
            n_idx += n_block;
        }
        k_idx += k_block;
    }
}

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * src: [K_BLOCK, N_BLOCK]
 * dst: [N_BLOCK/n_blk, K_BLOCK, n_blk]
 * n_blk: pack2n/packn/n_tail
 ************************************************************/
static inline void reorder_b_pack2nxk_fp16(__fp16 *src, __fp16 *dst, int N_BLOCK, int K_BLOCK,
                                           int ldb)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int pack2n = packn * 2;
    int vl = vsetvl_e16m2(pack2n);

    int j = 0;
    for (; j + pack2n - 1 < N_BLOCK; j += pack2n) {
        __fp16 *s_ptr = src + j;
        for (int c = 0; c < K_BLOCK; c++) {
            vfloat16m2_t _src = vle16_v_f16m2(s_ptr, vl);
            vse16_v_f16m2(dst, _src, vl);
            s_ptr += ldb;
            dst += vl;
        }
    }
    while (j < N_BLOCK) {
        vl = vsetvl_e16m1(N_BLOCK - j);
        __fp16 *s_ptr = src + j;
        for (int c = 0; c < K_BLOCK; c++) {
            vfloat16m1_t _src = vle16_v_f16m1(s_ptr, vl);
            vse16_v_f16m1(dst, _src, vl);
            s_ptr += ldb;
            dst += vl;
        }
        j += vl;
    }
}

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * src: [k, n]
 * dst: [n/n_blk, k/k_blk, n_blk/pack2n, k_blk, pack2n]
 * n_blk: N_BLK, N_tail
 * k_blk: K_BLK, K_tail
 ************************************************************/
void shl_rvv_reorder_b_block_pack2nxk_fp16(__fp16 *src, __fp16 *dst, int k, int n, const int K_BLK,
                                           const int N_BLK)
{
    int k_block = K_BLK;
    int k_idx = 0;
    while (k_idx < k) {
        if (k - k_idx < k_block) {
            k_block = k - k_idx;
        }
        int n_block = N_BLK;
        int n_idx = 0;
        while (n_idx < n) {
            if (n - n_idx < n_block) {
                n_block = n - n_idx;
            }
            __fp16 *s_ptr = src + k_idx * n + n_idx;
            __fp16 *d_ptr = dst + n_idx * k + k_idx * n_block;
            reorder_b_pack2nxk_fp16(s_ptr, d_ptr, n_block, k_block, n);
            n_idx += n_block;
        }
        k_idx += k_block;
    }
}
