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
 * params:
 *  input:          origin input data
 *  input_padded:   input data after pad
 *  inc:            origin input channel
 *  inh:            origin input height
 *  inw:            origin input width
 *  padded_h:       input height after pad
 *  padded_w:       input width after pad
 *  pad_top:        origin pad top
 *  pad_left:       origin pad left
 *************************************************************/
void shl_rvv_pad_input_fp32(const float *input, float *input_padded, int inc, int inh, int inw,
                            int padded_h, int padded_w, int pad_top, int pad_left)
{
    int padded_hw = padded_h * padded_w;

    float *pad_ptr = input_padded;
    float *inp_ptr = (float *)input;
    int resi_h = padded_h - pad_top - inh;   // remain to pad on h (pad_down)
    int resi_w = padded_w - pad_left - inw;  // remain to pad on w (pad_right)
    int size;
    int vl = vsetvl_e32m1(csrr_vlenb() / sizeof(float));
    vfloat32m1_t _zero = vfmv_v_f_f32m1(0.0f, vl);

    for (int c = 0; c < inc; c++) {
        pad_ptr = input_padded + c * padded_hw;
        // pad h_top
        size = padded_w * pad_top;
        while (size > 0) {
            vl = vsetvl_e32m1(size);
            vse32_v_f32m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
            size -= vl;
        }
        // pad h_mid
        for (int h = 0; h < inh; h++) {
            // pad w_left
            memset(pad_ptr, 0, pad_left * sizeof(float));
            pad_ptr += pad_left;
            // pad w_mid
            size = inw;
            while (size > 0) {
                vl = vsetvl_e32m1(size);
                vfloat32m1_t _input = vle32_v_f32m1(inp_ptr, vl);
                inp_ptr += vl;
                vse32_v_f32m1(pad_ptr, _input, vl);
                pad_ptr += vl;
                size -= vl;
            }
            // pad w_end
            memset(pad_ptr, 0, resi_w * sizeof(float));
            pad_ptr += resi_w;
        }
        // pad h_bottom
        size = padded_w * resi_h;
        while (size > 0) {
            vl = vsetvl_e32m1(size);
            vse32_v_f32m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
            size -= vl;
        }
    }
}

void shl_rvv_pad_input_fp16(const __fp16 *input, __fp16 *input_padded, int inc, int inh, int inw,
                            int padded_h, int padded_w, int pad_top, int pad_left)
{
    int padded_hw = padded_h * padded_w;

    __fp16 *pad_ptr = input_padded;
    __fp16 *inp_ptr = (__fp16 *)input;
    int resi_h = padded_h - pad_top - inh;   // remain to pad on h (pad_down)
    int resi_w = padded_w - pad_left - inw;  // remain to pad on w (pad_right)
    int size;
    int vl = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
    vfloat16m1_t _zero = vfmv_v_f_f16m1(0.0f, vl);

    for (int c = 0; c < inc; c++) {
        pad_ptr = input_padded + c * padded_hw;
        // pad h_top
        size = padded_w * pad_top;
        while (size > 0) {
            vl = vsetvl_e16m1(size);
            vse16_v_f16m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
            size -= vl;
        }
        // pad h_mid
        for (int h = 0; h < inh; h++) {
            // pad w_left
            memset(pad_ptr, 0, pad_left * sizeof(__fp16));
            pad_ptr += pad_left;
            // pad w_mid
            size = inw;
            while (size > 0) {
                vl = vsetvl_e16m1(size);
                vfloat16m1_t _input = vle16_v_f16m1(inp_ptr, vl);
                inp_ptr += vl;
                vse16_v_f16m1(pad_ptr, _input, vl);
                pad_ptr += vl;
                size -= vl;
            }
            // pad w_end
            memset(pad_ptr, 0, resi_w * sizeof(__fp16));
            pad_ptr += resi_w;
        }
        // pad h_bottom
        size = padded_w * resi_h;
        while (size > 0) {
            vl = vsetvl_e16m1(size);
            vse16_v_f16m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
            size -= vl;
        }
    }
}

void shl_rvv_pad_input_int8(const int8_t *input, int8_t *input_padded, int inc, int inh, int inw,
                            int padded_h, int padded_w, int pad_top, int pad_left, int8_t pad_value)
{
    int padded_hw = padded_h * padded_w;

    int8_t *pad_ptr = input_padded;
    int8_t *inp_ptr = (int8_t *)input;
    int resi_h = padded_h - pad_top - inh;   // remain to pad on h (pad_down)
    int resi_w = padded_w - pad_left - inw;  // remain to pad on w (pad_right)
    int size;
    int vl = vsetvl_e8m1(csrr_vlenb() / sizeof(int8_t));
    vint8m1_t _pad_zero = vmv_v_x_i8m1(pad_value, vl);  // float 0.0 -> input->zero_point

    for (int c = 0; c < inc; c++) {
        pad_ptr = input_padded + c * padded_hw;
        // pad h_top
        size = padded_w * pad_top;
        while (size > 0) {
            vl = vsetvl_e8m1(size);
            vse8_v_i8m1(pad_ptr, _pad_zero, vl);
            pad_ptr += vl;
            size -= vl;
        }
        // pad h_mid
        for (int h = 0; h < inh; h++) {
            // pad w_left
            memset(pad_ptr, pad_value, pad_left * sizeof(int8_t));
            pad_ptr += pad_left;
            // pad w_mid
            size = inw;
            while (size > 0) {
                vl = vsetvl_e8m1(size);
                vint8m1_t _input = vle8_v_i8m1(inp_ptr, vl);
                inp_ptr += vl;
                vse8_v_i8m1(pad_ptr, _input, vl);
                pad_ptr += vl;
                size -= vl;
            }
            // pad w_end
            memset(pad_ptr, pad_value, resi_w * sizeof(int8_t));
            pad_ptr += resi_w;
        }
        // pad h_bottom
        size = padded_w * resi_h;
        while (size > 0) {
            vl = vsetvl_e8m1(size);
            vse8_v_i8m1(pad_ptr, _pad_zero, vl);
            pad_ptr += vl;
            size -= vl;
        }
    }
}

// constrains: in_c % packn = 0
void shl_rvv_pad_input_packn_fp32(const float *input, float *input_padded, int inc, int inh,
                                  int inw, int padded_h, int padded_w, int pad_top, int pad_left)
{
    const int packn = csrr_vlenb() / sizeof(float);
    int vl = vsetvl_e32m1(packn);

    float *pad_ptr = input_padded;
    float *inp_ptr = (float *)input;
    int pad_down = padded_h - pad_top - inh;    // remain to pad on h (pad_down)
    int pad_right = padded_w - pad_left - inw;  // remain to pad on w (pad_right)

    while (inc > 0) {
        vl = vsetvl_e32m1(inc);
        vfloat32m1_t _zero = vfmv_v_f_f32m1(0.0f, vl);
        // pad h_top
        for (int i = 0; i < pad_top * padded_w; i++) {
            vse32_v_f32m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
        }
        // pad h_mid
        for (int i = 0; i < inh; i++) {
            // pad w_left
            for (int j = 0; j < pad_left; j++) {
                vse32_v_f32m1(pad_ptr, _zero, vl);
                pad_ptr += vl;
            }
            // pad w_mid
            for (int j = 0; j < inw; j++) {
                vfloat32m1_t _tmp = vle32_v_f32m1(inp_ptr, vl);
                inp_ptr += vl;
                vse32_v_f32m1(pad_ptr, _tmp, vl);
                pad_ptr += vl;
            }
            // pad w_end
            for (int j = 0; j < pad_right; j++) {
                vse32_v_f32m1(pad_ptr, _zero, vl);
                pad_ptr += vl;
            }
        }
        // pad h_bottom
        for (int i = 0; i < pad_down * padded_w; i++) {
            vse32_v_f32m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
        }
        inc -= vl;
    }
}

void shl_rvv_pad_input_packn_fp16(const __fp16 *input, __fp16 *input_padded, int inc, int inh,
                                  int inw, int padded_h, int padded_w, int pad_top, int pad_left)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    __fp16 *pad_ptr = input_padded;
    __fp16 *inp_ptr = (__fp16 *)input;
    int pad_down = padded_h - pad_top - inh;    // remain to pad on h (pad_down)
    int pad_right = padded_w - pad_left - inw;  // remain to pad on w (pad_right)

    vfloat16m1_t _zero = vfmv_v_f_f16m1(0.0f, vl);

    int c = 0;
    for (; c + packn - 1 < inc; c += packn) {
        // pad h_top
        for (int i = 0; i < pad_top * padded_w; i++) {
            vse16_v_f16m1(pad_ptr, _zero, vl);
            pad_ptr += packn;
        }
        // pad h_mid
        for (int i = 0; i < inh; i++) {
            // pad w_left
            for (int j = 0; j < pad_left; j++) {
                vse16_v_f16m1(pad_ptr, _zero, vl);
                pad_ptr += packn;
            }
            // pad w_mid
            for (int j = 0; j < inw; j++) {
                vfloat16m1_t _tmp = vle16_v_f16m1(inp_ptr, vl);
                inp_ptr += packn;
                vse16_v_f16m1(pad_ptr, _tmp, vl);
                pad_ptr += packn;
            }
            // pad w_end
            for (int j = 0; j < pad_right; j++) {
                vse16_v_f16m1(pad_ptr, _zero, vl);
                pad_ptr += packn;
            }
        }
        // pad h_bottom
        for (int i = 0; i < pad_down * padded_w; i++) {
            vse16_v_f16m1(pad_ptr, _zero, vl);
            pad_ptr += packn;
        }
    }
}

// XXX: 需要适配 vector 0.7.1, mf2 不支持
// packn = 8 for vlen128
void shl_rvv_pad_input_packn_int8(const int8_t *input, int8_t *input_padded, int inc, int inh,
                                  int inw, int padded_h, int padded_w, int pad_top, int pad_left,
                                  int8_t pad_value)
{
#ifdef RVV_1_0_0
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8mf2(packn);

    int8_t *pad_ptr = input_padded;
    int8_t *inp_ptr = (int8_t *)input;
    int pad_down = padded_h - pad_top - inh;    // remain to pad on h (pad_down)
    int pad_right = padded_w - pad_left - inw;  // remain to pad on w (pad_right)

    vint8mf2_t _zero = vmv_v_x_i8mf2(pad_value, vl);

    int c = 0;
    for (; c + packn - 1 < inc; c += packn) {
        // pad h_top
        for (int i = 0; i < pad_top * padded_w; i++) {
            vse8_v_i8mf2(pad_ptr, _zero, vl);
            pad_ptr += packn;
        }
        // pad h_mid
        for (int i = 0; i < inh; i++) {
            // pad w_left
            for (int j = 0; j < pad_left; j++) {
                vse8_v_i8mf2(pad_ptr, _zero, vl);
                pad_ptr += packn;
            }
            // pad w_mid
            for (int j = 0; j < inw; j++) {
                vint8mf2_t _tmp = vle8_v_i8mf2(inp_ptr, vl);
                inp_ptr += packn;
                vse8_v_i8mf2(pad_ptr, _tmp, vl);
                pad_ptr += packn;
            }
            // pad w_end
            for (int j = 0; j < pad_right; j++) {
                vse8_v_i8mf2(pad_ptr, _zero, vl);
                pad_ptr += packn;
            }
        }
        // pad h_bottom
        for (int i = 0; i < pad_down * padded_w; i++) {
            vse8_v_i8mf2(pad_ptr, _zero, vl);
            pad_ptr += packn;
        }
    }
#endif
}

// constrains: inc % packn = 0
void shl_rvv_pad_input_pack1ton_fp32(const float *input, float *input_padded, int inc, int inh,
                                     int inw, int padded_h, int padded_w, int pad_top, int pad_left)
{
    const int packn = csrr_vlenb() / sizeof(float);
    int vl = vsetvl_e32m1(packn);
    const int in_size = inh * inw;  // per-channel size

    float *pad_ptr = input_padded;
    int pad_down = padded_h - pad_top - inh;    // remain to pad on h (pad_down)
    int pad_right = padded_w - pad_left - inw;  // remain to pad on w (pad_right)

    vfloat32m1_t _zero = vfmv_v_f_f32m1(0.0f, vl);

    while (inc > 0) {
        vl = vsetvl_e32m1(inc);
        float *inp_ptr = (float *)input;
        // pad h_top
        for (int i = 0; i < pad_top * padded_w; i++) {
            vse32_v_f32m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
        }
        // pad h_mid
        for (int i = 0; i < inh; i++) {
            // pad w_left
            for (int j = 0; j < pad_left; j++) {
                vse32_v_f32m1(pad_ptr, _zero, vl);
                pad_ptr += vl;
            }
            // pad w_mid
            for (int j = 0; j < inw; j++) {
                vfloat32m1_t _tmp = vlse32_v_f32m1(inp_ptr, in_size * sizeof(float), vl);
                inp_ptr++;
                vse32_v_f32m1(pad_ptr, _tmp, vl);
                pad_ptr += vl;
            }
            // pad w_end
            for (int j = 0; j < pad_right; j++) {
                vse32_v_f32m1(pad_ptr, _zero, vl);
                pad_ptr += vl;
            }
        }
        // pad h_bottom
        for (int i = 0; i < pad_down * padded_w; i++) {
            vse32_v_f32m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
        }
        input += in_size * vl;
        inc -= vl;
    }
}

void shl_rvv_pad_input_pack1ton_fp16(const __fp16 *input, __fp16 *input_padded, int inc, int inh,
                                     int inw, int padded_h, int padded_w, int pad_top, int pad_left)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    int vl = vsetvl_e16m1(packn);
    const int in_size = inh * inw;  // per-channel size

    __fp16 *pad_ptr = input_padded;
    int pad_down = padded_h - pad_top - inh;    // remain to pad on h (pad_down)
    int pad_right = padded_w - pad_left - inw;  // remain to pad on w (pad_right)

    vfloat16m1_t _zero = vfmv_v_f_f16m1(0.0f, vl);

    int c = 0;
    while (inc > 0) {
        vl = vsetvl_e16m1(inc);
        __fp16 *inp_ptr = (__fp16 *)input;
        // pad h_top
        for (int i = 0; i < pad_top * padded_w; i++) {
            vse16_v_f16m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
        }
        // pad h_mid
        for (int i = 0; i < inh; i++) {
            // pad w_left
            for (int j = 0; j < pad_left; j++) {
                vse16_v_f16m1(pad_ptr, _zero, vl);
                pad_ptr += vl;
            }
            // pad w_mid
            for (int j = 0; j < inw; j++) {
                vfloat16m1_t _tmp = vlse16_v_f16m1(inp_ptr, in_size * sizeof(__fp16), vl);
                inp_ptr++;
                vse16_v_f16m1(pad_ptr, _tmp, vl);
                pad_ptr += vl;
            }
            // pad w_end
            for (int j = 0; j < pad_right; j++) {
                vse16_v_f16m1(pad_ptr, _zero, vl);
                pad_ptr += vl;
            }
        }
        // pad h_bottom
        for (int i = 0; i < pad_down * padded_w; i++) {
            vse16_v_f16m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
        }
        input += in_size * vl;
        inc -= vl;
    }
}

void shl_rvv_pad_input_pack1ton_int8(const int8_t *input, int8_t *input_padded, int inc, int inh,
                                     int inw, int padded_h, int padded_w, int pad_top, int pad_left,
                                     int8_t pad_value)
{
#ifdef RVV_1_0_0
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int vl = vsetvl_e8mf2(packn);
    const int in_size = inh * inw;  // per-channel size

    int8_t *pad_ptr = input_padded;
    int pad_down = padded_h - pad_top - inh;    // remain to pad on h (pad_down)
    int pad_right = padded_w - pad_left - inw;  // remain to pad on w (pad_right)

    vint8mf2_t _zero = vmv_v_x_i8mf2(pad_value, vl);

    int c = 0;
    while (inc > 0) {
        vl = vsetvl_e8mf2(inc);
        int8_t *inp_ptr = (int8_t *)input;
        // pad h_top
        for (int i = 0; i < pad_top * padded_w; i++) {
            vse8_v_i8mf2(pad_ptr, _zero, vl);
            pad_ptr += vl;
        }
        // pad h_mid
        for (int i = 0; i < inh; i++) {
            // pad w_left
            for (int j = 0; j < pad_left; j++) {
                vse8_v_i8mf2(pad_ptr, _zero, vl);
                pad_ptr += vl;
            }
            // pad w_mid
            for (int j = 0; j < inw; j++) {
                vint8mf2_t _tmp = vlse8_v_i8mf2(inp_ptr, in_size * sizeof(int8_t), vl);
                inp_ptr++;
                vse8_v_i8mf2(pad_ptr, _tmp, vl);
                pad_ptr += vl;
            }
            // pad w_end
            for (int j = 0; j < pad_right; j++) {
                vse8_v_i8mf2(pad_ptr, _zero, vl);
                pad_ptr += vl;
            }
        }
        // pad h_bottom
        for (int i = 0; i < pad_down * padded_w; i++) {
            vse8_v_i8mf2(pad_ptr, _zero, vl);
            pad_ptr += vl;
        }
        input += in_size * vl;
        inc -= vl;
    }
#endif
}
