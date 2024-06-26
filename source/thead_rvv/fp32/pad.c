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

void shl_rvv_pad_input_nhwc_fp32(const float *input, float *input_padded, int inh, int inw, int inc,
                                 int padded_h, int padded_w, int pad_top, int pad_left)
{
    float *pad_ptr = input_padded;
    float *inp_ptr = (float *)input;
    int resi_h = padded_h - pad_top - inh;   // remain to pad on h (pad_down)
    int resi_w = padded_w - pad_left - inw;  // remain to pad on w (pad_right)
    int padded_wc = padded_w * inc;
    int padded_wtop = pad_left * inc;
    int padded_wbottom = resi_w * inc;
    int copy_wc = inw * inc;
    int size;
    int vl = vsetvl_e32m1(csrr_vlenb() / sizeof(float));
    vfloat32m1_t _zero = vfmv_v_f_f32m1(0.0f, vl);

    // pad h_top
    size = padded_wc * pad_top;
    while (size > 0) {
        vl = vsetvl_e32m1(size);
        vse32_v_f32m1(pad_ptr, _zero, vl);
        pad_ptr += vl;
        size -= vl;
    }
    // pad h_mid
    for (int h = 0; h < inh; h++) {
        // pad w_top
        size = padded_wtop;
        while (size > 0) {
            vl = vsetvl_e32m1(size);
            vse32_v_f32m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
            size -= vl;
        }
        // copy w_mid
        size = copy_wc;
        while (size > 0) {
            vl = vsetvl_e32m1(size);
            vfloat32m1_t _input = vle32_v_f32m1(inp_ptr, vl);
            inp_ptr += vl;
            vse32_v_f32m1(pad_ptr, _input, vl);
            pad_ptr += vl;
            size -= vl;
        }
        // pad w_bottom
        size = padded_wbottom;
        while (size > 0) {
            vl = vsetvl_e32m1(size);
            vse32_v_f32m1(pad_ptr, _zero, vl);
            pad_ptr += vl;
            size -= vl;
        }
    }
    // pad h_bottom
    size = padded_wc * resi_h;
    while (size > 0) {
        vl = vsetvl_e32m1(size);
        vse32_v_f32m1(pad_ptr, _zero, vl);
        pad_ptr += vl;
        size -= vl;
    }
}
