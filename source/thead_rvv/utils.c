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

int csrr_vl()
{
    int a = 0;
    asm volatile("csrr %0, vl" : "=r"(a) : : "memory");
    return a;
}

int csrr_vlenb()
{
    int a = 0;
    asm volatile("csrr %0, vlenb" : "=r"(a) : : "memory");
    return a;
}

/*  params:
    input:          origin input data
    input_padded:   input data after pad
    inc:            origin input channel
    inh:            origin input height
    inw:            origin input width
    padded_h:       input height after pad
    padded_w:       input width after pad
    pad_top:        origin pad top
    pad_left:       origin pad left
*/
void csi_nn_rvv_pad_input_fp32(const float *input, float *input_padded, int inc, int inh, int inw,
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

void csi_nn_rvv_pad_input_fp16(const __fp16 *input, __fp16 *input_padded, int inc, int inh, int inw,
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

void csi_nn_rvv_pad_input_int8(const int8_t *input, int8_t *input_padded, int inc, int inh, int inw,
                               int padded_h, int padded_w, int pad_top, int pad_left,
                               int8_t pad_value)
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

/********************* for int8 quantization *********************/
// add output_zeropint
void csi_nn_rvv_saturated_int8(int32_t *src, int8_t *dst, int32_t out_zp, int size)
{
    while (size > 0) {
        int vl = vsetvl_e32m2(size);
        vint32m4_t _tmp = vle32_v_i32m4(src, vl);
        _tmp = vadd_vx_i32m4(_tmp, out_zp, vl);

        vint16m2_t _tmp1 = vnclip_wx_i16m2(_tmp, 0, vl);  // narrow 32->16
        vint8m1_t _tmp2 = vnclip_wx_i8m1(_tmp1, 0, vl);   // narrow 16->8

        vse8_v_i8m1(dst, _tmp2, vl);
        src += vl;
        dst += vl;
        size -= vl;
    }
}

// 再量化 int32 -> int8
// (val * multiplier)/(2 ^ shift)
void csi_nn_rvv_requantize(int32_t *src, int32_t multiplier, int32_t shift, int channel_size)
{
    while (channel_size > 0) {
        int vl = vsetvl_e32m4(channel_size);
        vint32m4_t _val = vle32_v_i32m4(src, vl);
        vint32m4_t _mulh = vmulh_vx_i32m4(_val, multiplier, vl);
        vint32m4_t _res;
        // FIXME: precision error
        if (shift < 0) {
            _res = vssra_vx_i32m4(_mulh, -shift - 1, vl);
        } else {
            _res = vsll_vx_i32m4(_mulh, shift + 1, vl);
        }
        vse32_v_i32m4(src, _res, vl);
        src += vl;
        channel_size -= vl;
    }
}

// 反量化 int32 -> float32  int8 -> float32
void csi_nn_rvv_dequantize()
{
    ;
}

/********************* int4 easter eggs *********************/
void csi_nn_rvv_pad_input_int4_trans_int8(const int8_t *input, int8_t *input_padded, int inc,
                                          int inh, int inw, int padded_h, int padded_w, int pad_top,
                                          int pad_left, int8_t pad_value)
{
    int padded_hw = padded_h * padded_w;

    int8_t *pad_ptr = input_padded;
    int8_t *inp_ptr = (int8_t *)input;
    int resi_h = padded_h - pad_top - inh;   // remain to pad on h (pad_down)
    int resi_w = padded_w - pad_left - inw;  // remain to pad on w (pad_right)
    int size;
    int vl = vsetvl_e8m1(csrr_vlenb() / sizeof(int8_t));
    vint8m1_t _pad_zero = vmv_v_x_i8m1(pad_value, vl);  // float 0.0 -> input->zero_point

    // pad h_top
    size = padded_w * pad_top * inc;
    while (size > 0) {
        vl = vsetvl_e8m1(size);
        vse8_v_i8m1(pad_ptr, _pad_zero, vl);
        pad_ptr += vl;
        size -= vl;
    }
    // pad h_mid
    for (int h = 0; h < inh; h++) {
        // pad w_left
        size = pad_left * inc;
        memset(pad_ptr, pad_value, size * sizeof(int8_t));
        pad_ptr += size;
        // pad w_mid
        csi_nn_rvv_int4_trans_int8(inp_ptr, pad_ptr, inw * inc);
        inp_ptr += inw * inc / 2;
        pad_ptr += inw * inc;
        // pad w_right
        size = resi_w * inc;
        memset(pad_ptr, pad_value, size * sizeof(int8_t));
        pad_ptr += size;
    }
    // pad h_bottom
    size = padded_w * resi_h * inc;
    while (size > 0) {
        vl = vsetvl_e8m1(size);
        vse8_v_i8m1(pad_ptr, _pad_zero, vl);
        pad_ptr += vl;
        size -= vl;
    }
}

// size: int4 number
// TODO: 这里是不是需要增加一条指令
void csi_nn_rvv_int4_to_int8(int8_t *src, int8_t *dst, int size)
{
    int j = size / 2;
    while (j > 0) {
        int vl = vsetvl_e8m1(j);
        vint8m1_t _input = vle8_v_i8m1(src, vl);
        vint8m1_t _low = vand_vx_i8m1(_input, 0x0f, vl);
        vint8m1_t _high_input = vsra_vx_i8m1(_input, 4, vl);
        vint8m1_t _high = vand_vx_i8m1(_high_input, 0x0f, vl);
        vsse8_v_i8m1(dst, 2 * sizeof(int8_t), _low, vl);
        vsse8_v_i8m1(dst + 1, 2 * sizeof(int8_t), _high, vl);

        src += vl;
        dst += 2 * vl;
        j -= vl;
    }
    // tail, odd size
    if (size & 1) {
        *dst = *src;
    }
}

// size: int4 number
// todo: replace with vpnclip_wx inst
void csi_nn_rvv_int8_to_int4(int8_t *src, int8_t *dst, int size)
{
    int j = size / 2;
    while (j > 0) {
        int vl = vsetvl_e8m1(j);
        vint8m1_t _low_tmp = vlse8_v_i8m1(src, 2 * sizeof(int8_t), vl);
        vint8m1_t _high_tmp = vlse8_v_i8m1(src + 1, 2 * sizeof(int8_t), vl);
        vint8m1_t _low = vand_vx_i8m1(_low_tmp, 0x0f, vl);
        vint8m1_t _high = vsll_vx_i8m1(_high_tmp, 4, vl);
        vint8m1_t _output = vor_vv_i8m1(_low, _high, vl);
        vse8_v_i8m1(dst, _output, vl);

        src += 2 * vl;
        dst += vl;
        j -= vl;
    }
    // tail, odd size
    if (size & 1) {
        *dst = *src;
    }
}

// size: int4 number
// TODO: replace with vpwadd.vx inst
void csi_nn_rvv_int4_trans_int8(int8_t *src, int8_t *dst, int size)
{
    int j = size / 2;
    while (j > 0) {
        int vl = vsetvl_e8m1(j);
        vint8m1_t _input = vle8_v_i8m1(src, vl);
        vint8m1_t _low = vand_vx_i8m1(_input, 0x0f, vl);
        vbool8_t _mask = vmsgt_vx_i8m1_b8(_low, 7, vl);
        vint8m1_t _low_int8 = vsub_vx_i8m1_m(_mask, _low, _low, 16, vl);
        vint8m1_t _high_int8 = vsra_vx_i8m1(_input, 4, vl);
        vsse8_v_i8m1(dst, 2 * sizeof(int8_t), _low_int8, vl);
        vsse8_v_i8m1(dst + 1, 2 * sizeof(int8_t), _high_int8, vl);

        src += vl;
        dst += 2 * vl;
        j -= vl;
    }
    // tail, odd size
    if (size & 1) {
        *dst = *src > 7 ? (*src - 16) : (*src);
    }
}
