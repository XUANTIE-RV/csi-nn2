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

static float *rvv_tensor_ndarray_to_nc1xc0_fp32(struct csinn_tensor *t)
{
    int batch = t->dim[0];
    int in_c = t->dim[1];
    int inner_size = 1;
    for (int i = 2; i < t->dim_count; i++) {
        inner_size *= t->dim[i];
    }

    float *src = t->data;
    float *dst = (float *)shl_mem_alloc(csinn_tensor_byte_size(t));

    const int packn = csrr_vlenb() / sizeof(float);
    int vl = vsetvl_e32m1(packn);
    int batch_size = in_c * inner_size;

    float *out_ptr = dst;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            float *in_ptr = src + b * batch_size + c * inner_size;
            for (int i = 0; i < inner_size; i++) {
                vfloat32m1_t _tmp = vlse32_v_f32m1(in_ptr, inner_size * sizeof(float), vl);
                in_ptr++;
                vse32_v_f32m1(out_ptr, _tmp, vl);
                out_ptr += vl;
            }
        }
    }
    /* update tensor info to nc1hwc0 */
    t->dim[1] = in_c / packn;
    t->dim_count = t->dim_count + 1;
    t->dim[t->dim_count - 1] = packn;
    if (t->layout == CSINN_LAYOUT_NCDHW) {
        t->layout = CSINN_LAYOUT_NC1DHWC0;
    } else if (t->layout == CSINN_LAYOUT_NCHW) {
        t->layout = CSINN_LAYOUT_NC1HWC0;
    } else if (t->layout == CSINN_LAYOUT_NCW) {
        t->layout = CSINN_LAYOUT_NC1WC0;
    } else if (t->layout == CSINN_LAYOUT_NC) {
        t->layout = CSINN_LAYOUT_NC1C0;
    }
    return dst;
}

static __fp16 *rvv_tensor_ndarray_to_nc1xc0_fp16(struct csinn_tensor *t)
{
    int batch = t->dim[0];
    int in_c = t->dim[1];
    int inner_size = 1;
    for (int i = 2; i < t->dim_count; i++) {
        inner_size *= t->dim[i];
    }

    __fp16 *src = t->data;
    __fp16 *dst = (__fp16 *)shl_mem_alloc(csinn_tensor_byte_size(t));

    const int packn = csrr_vlenb() / sizeof(__fp16);
    int vl = vsetvl_e16m1(packn);
    int batch_size = in_c * inner_size;

    __fp16 *out_ptr = dst;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            __fp16 *in_ptr = src + b * batch_size + c * inner_size;
            for (int i = 0; i < inner_size; i++) {
                vfloat16m1_t _tmp = vlse16_v_f16m1(in_ptr, inner_size * sizeof(__fp16), vl);
                in_ptr++;
                vse16_v_f16m1(out_ptr, _tmp, vl);
                out_ptr += vl;
            }
        }
    }
    /* update tensor info to nc1hwc0 */
    t->dim[1] = in_c / packn;
    t->dim_count = t->dim_count + 1;
    t->dim[t->dim_count - 1] = packn;
    if (t->layout == CSINN_LAYOUT_NCDHW) {
        t->layout = CSINN_LAYOUT_NC1DHWC0;
    } else if (t->layout == CSINN_LAYOUT_NCHW) {
        t->layout = CSINN_LAYOUT_NC1HWC0;
    } else if (t->layout == CSINN_LAYOUT_NCW) {
        t->layout = CSINN_LAYOUT_NC1WC0;
    } else if (t->layout == CSINN_LAYOUT_NC) {
        t->layout = CSINN_LAYOUT_NC1C0;
    }
    return dst;
}

static int8_t *rvv_tensor_ndarray_to_nc1xc0_int8(struct csinn_tensor *t)
{
    int batch = t->dim[0];
    int in_c = t->dim[1];
    int inner_size = 1;
    for (int i = 2; i < t->dim_count; i++) {
        inner_size *= t->dim[i];
    }

    int8_t *src = t->data;
    int8_t *dst = (int8_t *)shl_mem_alloc(csinn_tensor_byte_size(t));

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int vl = vsetvl_e8m1(packn);
    int batch_size = in_c * inner_size;

    int8_t *out_ptr = dst;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            int8_t *in_ptr = src + b * batch_size + c * inner_size;
            for (int i = 0; i < inner_size; i++) {
                vint8m1_t _tmp = vlse8_v_i8m1(in_ptr, inner_size * sizeof(int8_t), vl);
                in_ptr++;
                vse8_v_i8m1(out_ptr, _tmp, vl);
                out_ptr += vl;
            }
        }
    }
    /* update tensor info to nc1hwc0 */
    t->dim[1] = in_c / packn;
    t->dim_count = t->dim_count + 1;
    t->dim[t->dim_count - 1] = packn;
    if (t->layout == CSINN_LAYOUT_NCDHW) {
        t->layout = CSINN_LAYOUT_NC1DHWC0;
    } else if (t->layout == CSINN_LAYOUT_NCHW) {
        t->layout = CSINN_LAYOUT_NC1HWC0;
    } else if (t->layout == CSINN_LAYOUT_NCW) {
        t->layout = CSINN_LAYOUT_NC1WC0;
    } else if (t->layout == CSINN_LAYOUT_NC) {
        t->layout = CSINN_LAYOUT_NC1C0;
    }
    return dst;
}

static float *rvv_tensor_nc1xc0_to_ndarray_fp32(struct csinn_tensor *t)
{
    int batch = t->dim[0];
    int in_c1 = t->dim[1];
    int inner_size = 1;
    for (int i = 2; i < t->dim_count - 1; i++) {
        inner_size *= t->dim[i];
    }
    int in_elempack = t->dim[t->dim_count - 1];

    float *src = t->data;
    float *dst = (float *)shl_mem_alloc(csinn_tensor_byte_size(t));

    const int packn = csrr_vlenb() / sizeof(float);
    int vl = vsetvl_e32m1(packn);
    int batch_size = in_c1 * inner_size * in_elempack;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c1; c++) {
            float *out_ptr = dst + b * batch_size + c * inner_size * in_elempack;
            for (int i = 0; i < inner_size; i++) {
                vfloat32m1_t _tmp = vle32_v_f32m1(src, vl);
                src += vl;
                vsse32_v_f32m1(out_ptr, inner_size * sizeof(float), _tmp, vl);
                out_ptr++;
            }
        }
    }
    /* update tensor info to nchw */
    t->dim[1] = in_c1 * packn;
    t->dim[t->dim_count - 1] = 0;
    t->dim_count = t->dim_count - 1;
    if (t->layout == CSINN_LAYOUT_NC1DHWC0) {
        t->layout = CSINN_LAYOUT_NCDHW;
    } else if (t->layout == CSINN_LAYOUT_NC1HWC0) {
        t->layout = CSINN_LAYOUT_NCHW;
    } else if (t->layout == CSINN_LAYOUT_NC1WC0) {
        t->layout = CSINN_LAYOUT_NCW;
    } else if (t->layout == CSINN_LAYOUT_NC1C0) {
        t->layout = CSINN_LAYOUT_NC;
    }
    return dst;
}

static __fp16 *rvv_tensor_nc1xc0_to_ndarray_fp16(struct csinn_tensor *t)
{
    int batch = t->dim[0];
    int in_c1 = t->dim[1];
    int inner_size = 1;
    for (int i = 2; i < t->dim_count - 1; i++) {
        inner_size *= t->dim[i];
    }
    int in_elempack = t->dim[t->dim_count - 1];

    __fp16 *src = t->data;
    __fp16 *dst = (__fp16 *)shl_mem_alloc(csinn_tensor_byte_size(t));

    const int packn = csrr_vlenb() / sizeof(__fp16);
    int vl = vsetvl_e16m1(packn);
    int batch_size = in_c1 * inner_size * in_elempack;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c1; c++) {
            __fp16 *out_ptr = dst + b * batch_size + c * inner_size * in_elempack;
            for (int i = 0; i < inner_size; i++) {
                vfloat16m1_t _tmp = vle16_v_f16m1(src, vl);
                src += vl;
                vsse16_v_f16m1(out_ptr, inner_size * sizeof(__fp16), _tmp, vl);
                out_ptr++;
            }
        }
    }
    /* update tensor info to nchw */
    t->dim[1] = in_c1 * packn;
    t->dim[t->dim_count - 1] = 0;
    t->dim_count = t->dim_count - 1;
    if (t->layout == CSINN_LAYOUT_NC1DHWC0) {
        t->layout = CSINN_LAYOUT_NCDHW;
    } else if (t->layout == CSINN_LAYOUT_NC1HWC0) {
        t->layout = CSINN_LAYOUT_NCHW;
    } else if (t->layout == CSINN_LAYOUT_NC1WC0) {
        t->layout = CSINN_LAYOUT_NCW;
    } else if (t->layout == CSINN_LAYOUT_NC1C0) {
        t->layout = CSINN_LAYOUT_NC;
    }
    return dst;
}

static int8_t *rvv_tensor_nc1xc0_to_ndarray_int8(struct csinn_tensor *t)
{
    int batch = t->dim[0];
    int in_c1 = t->dim[1];
    int inner_size = 1;
    for (int i = 2; i < t->dim_count - 1; i++) {
        inner_size *= t->dim[i];
    }
    int in_elempack = t->dim[t->dim_count - 1];

    int8_t *src = t->data;
    int8_t *dst = (int8_t *)shl_mem_alloc(csinn_tensor_byte_size(t));

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int vl = vsetvl_e8m1(packn);
    int batch_size = in_c1 * inner_size * in_elempack;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c1; c++) {
            int8_t *out_ptr = dst + b * batch_size + c * inner_size * in_elempack;
            for (int i = 0; i < inner_size; i++) {
                vint8m1_t _tmp = vle8_v_i8m1(src, vl);
                src += vl;
                vsse8_v_i8m1(out_ptr, inner_size * sizeof(int8_t), _tmp, vl);
                out_ptr++;
            }
        }
    }
    /* update tensor info to nchw */
    t->dim[1] = in_c1 * packn;
    t->dim[t->dim_count - 1] = 0;
    t->dim_count = t->dim_count - 1;
    if (t->layout == CSINN_LAYOUT_NC1DHWC0) {
        t->layout = CSINN_LAYOUT_NCDHW;
    } else if (t->layout == CSINN_LAYOUT_NC1HWC0) {
        t->layout = CSINN_LAYOUT_NCHW;
    } else if (t->layout == CSINN_LAYOUT_NC1WC0) {
        t->layout = CSINN_LAYOUT_NCW;
    } else if (t->layout == CSINN_LAYOUT_NC1C0) {
        t->layout = CSINN_LAYOUT_NC;
    }
    return dst;
}

/*********************************************************************************
 * tensor_replace: tensor->data point new alloc memory, free old
 ********************************************************************************/
void shl_rvv_tensor_ndarray_to_nc1xc0_replace_fp32(struct csinn_tensor *t)
{
    float *ret = rvv_tensor_ndarray_to_nc1xc0_fp32(t);
    shl_mem_free(t->data);
    t->data = ret;
}

void shl_rvv_tensor_ndarray_to_nc1xc0_replace_fp16(struct csinn_tensor *t)
{
    __fp16 *ret = rvv_tensor_ndarray_to_nc1xc0_fp16(t);
    shl_mem_free(t->data);
    t->data = ret;
}

void shl_rvv_tensor_ndarray_to_nc1xc0_replace_int8(struct csinn_tensor *t)
{
    int8_t *ret = rvv_tensor_ndarray_to_nc1xc0_int8(t);
    shl_mem_free(t->data);
    t->data = ret;
}

void shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(struct csinn_tensor *t)
{
    float *ret = rvv_tensor_nc1xc0_to_ndarray_fp32(t);
    shl_mem_free(t->data);
    t->data = ret;
}

void shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(struct csinn_tensor *t)
{
    __fp16 *ret = rvv_tensor_nc1xc0_to_ndarray_fp16(t);
    shl_mem_free(t->data);
    t->data = ret;
}

void shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(struct csinn_tensor *t)
{
    int8_t *ret = rvv_tensor_nc1xc0_to_ndarray_int8(t);
    shl_mem_free(t->data);
    t->data = ret;
}

/*********************************************************************************
 * tensor_inplace: tensor->data point origin address, memcpy from new layout data
 ********************************************************************************/
void shl_rvv_tensor_ndarray_to_nc1xc0_inplace_fp32(struct csinn_tensor *t)
{
    float *ret = rvv_tensor_ndarray_to_nc1xc0_fp32(t);
    memcpy(t->data, ret, csinn_tensor_byte_size(t));
    shl_mem_free(ret);
}

void shl_rvv_tensor_ndarray_to_nc1xc0_inplace_fp16(struct csinn_tensor *t)
{
    __fp16 *ret = rvv_tensor_ndarray_to_nc1xc0_fp16(t);
    memcpy(t->data, ret, csinn_tensor_byte_size(t));
    shl_mem_free(ret);
}

void shl_rvv_tensor_ndarray_to_nc1xc0_inplace_int8(struct csinn_tensor *t)
{
    int8_t *ret = rvv_tensor_ndarray_to_nc1xc0_int8(t);
    memcpy(t->data, ret, csinn_tensor_byte_size(t));
    shl_mem_free(ret);
}

void shl_rvv_tensor_nc1xc0_to_ndarray_inplace_fp32(struct csinn_tensor *t)
{
    float *ret = rvv_tensor_nc1xc0_to_ndarray_fp32(t);
    memcpy(t->data, ret, csinn_tensor_byte_size(t));
    shl_mem_free(ret);
}

void shl_rvv_tensor_nc1xc0_to_ndarray_inplace_fp16(struct csinn_tensor *t)
{
    __fp16 *ret = rvv_tensor_nc1xc0_to_ndarray_fp16(t);
    memcpy(t->data, ret, csinn_tensor_byte_size(t));
    shl_mem_free(ret);
}

void shl_rvv_tensor_nc1xc0_to_ndarray_inplace_int8(struct csinn_tensor *t)
{
    int8_t *ret = rvv_tensor_nc1xc0_to_ndarray_int8(t);
    memcpy(t->data, ret, csinn_tensor_byte_size(t));
    shl_mem_free(ret);
}

/********************* for fp16 quantization *********************/

void shl_rvv_sidcso_op_requantize_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel)
{
}

/* linear calculations ops, such as relu, leaky_relu, prelu, etc. */
void shl_rvv_siso_op_requantize_fp16(struct csinn_tensor *input, struct csinn_tensor *output)
{
}

void shl_rvv_diso_op_requantize_fp16(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                     struct csinn_tensor *output)
{
}

/********************* for int8 quantization *********************/
// add output_zeropoint
void shl_rvv_saturated_int8(int32_t *src, int8_t *dst, int32_t out_zp, int size)
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
void shl_rvv_requantize(int32_t *src, int32_t multiplier, int32_t shift, int channel_size)
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
void shl_rvv_dequantize() { ; }

// 反量化 int8 -> float16
void shl_rvv_dequantize_i8_to_f16(int8_t *src, __fp16 *dst, int size, int32_t zp, float scale)
{
    while (size > 0) {
        int vl = vsetvl_e8m4(size);
        vint8m4_t _in = vle8_v_i8m4(src, vl);
        vint16m8_t _in_w = vwsub_vx_i16m8(_in, zp, vl);
        vfloat16m8_t _in_f = vfcvt_f_x_v_f16m8(_in_w, vl);
        _in_f = vfmul_vf_f16m8(_in_f, scale, vl);
        vse16_v_f16m8(dst, _in_f, vl);
        src += vl;
        dst += vl;
        size -= vl;
    }
}

/********************* int4 easter eggs *********************/
void shl_rvv_pad_input_int4_trans_int8(const int8_t *input, int8_t *input_padded, int inc, int inh,
                                       int inw, int padded_h, int padded_w, int pad_top,
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
        shl_rvv_int4_trans_int8(inp_ptr, pad_ptr, inw * inc);
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
void shl_rvv_int4_to_int8(int8_t *src, int8_t *dst, int size)
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
void shl_rvv_int8_to_int4(int8_t *src, int8_t *dst, int size)
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
void shl_rvv_int4_trans_int8(int8_t *src, int8_t *dst, int size)
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

#ifdef XTHEADVDOT
void shl_rvv_saturated_int4(int32_t *src, int8_t *dst, int32_t out_zp, int size)
{
    while (size > 0) {
        int vl = vsetvl_e32m8(size);
        vint32m8_t _tmp = vle32_v_i32m8(src, vl);
        _tmp = vadd_vx_i32m8(_tmp, out_zp, vl);

        vint16m4_t _tmp1 = vnclip_wx_i16m4(_tmp, 0, vl);  // narrow 32->16
        vint8m2_t _tmp2 = vnclip_wx_i8m2(_tmp1, 0, vl);   // narrow 16->8
        vint8m1_t _res = vpnclip_wx_i8m1(vreinterpret_v_i8m2_i16m2(_tmp2), 0, vl / 2);

        vse8_v_i8m1(dst, _res, vl / 2);
        src += vl;
        dst += vl / 2;
        size -= vl;
    }
}
#endif

static void shl_rvv_avgpool_get_pad(enum avgpool_loc_enum loc, int *pad_h, int *pad_w, int pad_top,
                                    int pad_down, int pad_left, int pad_right)
{
    switch (loc) {
        case AVGPOOL_LEFT_TOP:
            *pad_h = pad_top;
            *pad_w = pad_left;
            break;
        case AVGPOOL_RIGHT_TOP:
            *pad_h = pad_top;
            *pad_w = pad_right;
            break;
        case AVGPOOL_LEFT_BOTTOM:
            *pad_h = pad_down;
            *pad_w = pad_left;
            break;
        case AVGPOOL_RIGHT_BOTTOM:
            *pad_h = pad_down;
            *pad_w = pad_right;
            break;
        case AVGPOOL_LEFT:
            *pad_h = 0;
            *pad_w = pad_left;
            break;
        case AVGPOOL_RIGHT:
            *pad_h = 0;
            *pad_w = pad_right;
            break;
        case AVGPOOL_TOP:
            *pad_h = pad_top;
            *pad_w = 0;
            break;
        case AVGPOOL_BOTTOM:
            *pad_h = pad_down;
            *pad_w = 0;
            break;
        case AVGPOOL_CENTER:
            *pad_h = 0;
            *pad_w = 0;
            break;
        default:
            *pad_h = 0;
            *pad_w = 0;
            break;
    }
}

int shl_rvv_avgpool_get_window_size(struct csinn_pool_params *params, int idx_h_start,
                                    int idx_h_end, int idx_w_start, int idx_w_end,
                                    enum avgpool_loc_enum loc)
{
    int kernel_h = params->filter_height;
    int kernel_w = params->filter_width;
    int pad_left = params->pad_left;
    int pad_right = params->pad_right;
    int pad_top = params->pad_top;
    int pad_down = params->pad_down;

    int valid_h = idx_h_end - idx_h_start;
    int valid_w = idx_w_end - idx_w_start;
    int valid_size = valid_h * valid_w;

    int pad_h, pad_w;
    shl_rvv_avgpool_get_pad(loc, &pad_h, &pad_w, pad_top, pad_down, pad_left, pad_right);
    int real_kernel_h = (valid_h + pad_h < kernel_h) ? valid_h + pad_h : kernel_h;
    int real_kernel_w = (valid_w + pad_w < kernel_w) ? valid_w + pad_w : kernel_w;
    int window_size =
        (params->count_include_pad == 1) ? (real_kernel_h * real_kernel_w) : valid_size;
    return window_size;
}

void shl_rvv_i16_to_f32(const int16_t *input, float *output, int32_t offset, float *scale,
                        uint32_t length)
{
    int vl = vsetvl_e32m4(length);
    vint16m2_t _z = vmv_v_x_i16m2(offset, vl);
    vfloat32m4_t _s = vfmv_v_f_f32m4(*scale, vl);
    while (length > 0) {
        vl = vsetvl_e16m2(length);
        vint16m2_t _in = vle16_v_i16m2(input, vl);
        input += vl;
        vint32m4_t _i32 = vwsub_vv_i32m4(_in, _z, vl);
        vfloat32m4_t _f32 = vfcvt_f_x_v_f32m4(_i32, vl);
        _f32 = vfmul_vv_f32m4(_f32, _s, vl);
        vse32_v_f32m4(output, _f32, vl);
        output += vl;
        length -= vl;
    }
}

void shl_rvv_f32_to_i16(const float *input, int16_t *output, int32_t offset, float *scale,
                        uint32_t length)
{
    int vl = vsetvl_e32m4(length);
    vint32m4_t _z = vmv_v_x_i32m4(offset, vl);
    vfloat32m4_t _1_s = vfmv_v_f_f32m4(1 / *scale, vl);
    while (length > 0) {
        vl = vsetvl_e16m2(length);
        vfloat32m4_t _in = vle32_v_f32m4(input, vl);
        input += vl;
        vfloat32m4_t _f32 = vfmul_vv_f32m4(_in, _1_s, vl);
        vint32m4_t _i32 = vfcvt_x_f_v_i32m4(_f32, vl);
        _i32 = vadd_vv_i32m4(_i32, _z, vl);
        vint16m2_t _i16 = vnclip_wx_i16m2(_i32, 0, vl);
        vse16_v_i16m2(output, _i16, vl);
        output += vl;
        length -= vl;
    }
}

void shl_rvv_i32_to_f32(const int32_t *input, float *output, int32_t offset, float *scale,
                        uint32_t length)
{
    while (length > 0) {
        int vl = vsetvl_e32m4(length);
        vint32m4_t _i32 = vle32_v_i32m4(input, vl);
        input += vl;
        _i32 = vsub_vx_i32m4(_i32, offset, vl);
        vfloat32m4_t _f32 = vfcvt_f_x_v_f32m4(_i32, vl);
        _f32 = vfmul_vf_f32m4(_f32, *scale, vl);
        vse32_v_f32m4(output, _f32, vl);
        output += vl;
        length -= vl;
    }
}

void shl_rvv_f32_to_i32(const float *input, int32_t *output, int32_t offset, float *scale,
                        uint32_t length)
{
    float _1_s = 1 / *scale;
    while (length > 0) {
        int vl = vsetvl_e32m4(length);
        vfloat32m4_t _f32 = vle32_v_f32m4(input, vl);
        input += vl;
        _f32 = vfmul_vf_f32m4(_f32, _1_s, vl);
        vint32m4_t _i32 = vfcvt_x_f_v_i32m4(_f32, vl);
        _i32 = vadd_vx_i32m4(_i32, offset, vl);
        vse32_v_i32m4(output, _i32, vl);
        output += vl;
        length -= vl;
    }
}

void shl_rvv_i64_to_f32(const int64_t *input, float *output, uint32_t length)
{
    while (length > 0) {
        int vl = vsetvl_e32m4(length);
        vint64m8_t _i64 = vle64_v_i64m8(input, vl);
        input += vl;
        vfloat32m4_t _f32 = vfncvt_f_x_w_f32m4(_i64, vl);
        vse32_v_f32m4(output, _f32, vl);
        output += vl;
        length -= vl;
    }
}

void shl_rvv_f32_to_i64(const float *input, int64_t *output, uint32_t length)
{
    while (length > 0) {
        int vl = vsetvl_e32m4(length);
        vfloat32m4_t _f32 = vle32_v_f32m4(input, vl);
        input += vl;
        vint64m8_t _i64 = vfwcvt_x_f_v_i64m8(_f32, vl);
        vse64_v_i64m8(output, _i64, vl);
        output += vl;
        length -= vl;
    }
}

void shl_rvv_f16_to_f32(const __fp16 *input, float *output, float *scale, uint32_t length)
{
    int vl = vsetvl_e32m4(length);
    if (fabs(*scale - 1) > FLT_EPSILON) {
        vfloat32m4_t _s = vfmv_v_f_f32m4(*scale, vl);
        while (length > 0) {
            vl = vsetvl_e16m2(length);
            vfloat16m2_t _f16 = vle16_v_f16m2(input, vl);
            input += vl;
            vfloat32m4_t _f32 = vfwcvt_f_f_v_f32m4(_f16, vl);
            // dequantize
            _f32 = vfmul_vv_f32m4(_f32, _s, vl);
            vse32_v_f32m4(output, _f32, vl);
            output += vl;
            length -= vl;
        }
    } else {
        while (length > 0) {
            vl = vsetvl_e16m2(length);
            vfloat16m2_t _f16 = vle16_v_f16m2(input, vl);
            input += vl;
            vfloat32m4_t _f32 = vfwcvt_f_f_v_f32m4(_f16, vl);
            vse32_v_f32m4(output, _f32, vl);
            output += vl;
            length -= vl;
        }
    }
}

void shl_rvv_f32_to_f16(const float *input, __fp16 *output, float *scale, uint32_t length)
{
    int vl = vsetvl_e32m4(length);
    if (fabs(*scale - 1) > FLT_EPSILON) {
        vfloat32m4_t _1_s = vfmv_v_f_f32m4(1 / *scale, vl);
        while (length > 0) {
            vl = vsetvl_e32m4(length);
            vfloat32m4_t _f32 = vle32_v_f32m4(input, vl);
            input += vl;
            // quantize
            _f32 = vfmul_vv_f32m4(_f32, _1_s, vl);
            vfloat16m2_t _f16 = vfncvt_f_f_w_f16m2(_f32, vl);
            vse16_v_f16m2(output, _f16, vl);
            output += vl;
            length -= vl;
        }
    } else {
        while (length > 0) {
            vl = vsetvl_e32m4(length);
            vfloat32m4_t _f32 = vle32_v_f32m4(input, vl);
            input += vl;
            vfloat16m2_t _f16 = vfncvt_f_f_w_f16m2(_f32, vl);
            vse16_v_f16m2(output, _f16, vl);
            output += vl;
            length -= vl;
        }
    }
}

bool shl_rvv_get_binary_model_op_init(struct csinn_session *sess)
{
    struct shl_rvv_option *option = shl_rvv_get_graph_option(sess);
    if (option && option->binary_model_op_init) {
        return true;
    } else {
        return false;
    }
}

void shl_rvv_nc1xc0_fp16_to_nchw_fp32(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    int batch = src->dim[0];
    int channel = src->dim[1];
    int inner_size = 1;
    for (int i = 2; i < src->dim_count - 1; i++) {
        inner_size *= src->dim[i];
    }

    float scale = src->qinfo->scale;
    int stride = inner_size * sizeof(float);

    __fp16 *src_data = (__fp16 *)src->data;
    float *dst_data = (float *)dest->data;

    if (fabs(scale - 1) > FLT_EPSILON) {
        vfloat32m2_t _s = vfmv_v_f_f32m2(scale, packn);
        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < channel; c++) {
                float *out_ptr = (float *)dst_data + (n * channel + c) * inner_size * packn;
                for (int i = 0; i < inner_size; i++) {
                    vfloat16m1_t _in = vle16_v_f16m1(src_data, packn);
                    vfloat32m2_t _f32 = vfwcvt_f_f_v_f32m2(_in, packn);
                    _f32 = vfmul_vv_f32m2(_f32, _s, packn);
                    vsse32_v_f32m2(out_ptr, stride, _f32, packn);
                    src_data += packn;
                    out_ptr += 1;
                }
            }
        }
    } else {
        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < channel; c++) {
                float *out_ptr = (float *)dst_data + (n * channel + c) * inner_size * packn;
                for (int i = 0; i < inner_size; i++) {
                    vfloat16m1_t _in = vle16_v_f16m1(src_data, packn);
                    vfloat32m2_t _f32 = vfwcvt_f_f_v_f32m2(_in, packn);
                    vsse32_v_f32m2(out_ptr, stride, _f32, packn);
                    src_data += packn;
                    out_ptr += 1;
                }
            }
        }
    }
}

int shl_rvv_transpose_get_tail(int32_t *permute, int32_t permute_num)
{
    int tail = 0;
    for (int i = permute_num - 1; i >= 0; i--) {
        if (permute[i] != i) {
            break;
        }
        tail += 1;
    }
    return tail;
}

int shl_rvv_transpose_get_in_index(int32_t *dim, int32_t *idx, int32_t dim_count)
{
    int res = idx[0];
    for (int i = 1; i < dim_count; i++) {
        res = res * dim[i] + idx[i];
    }
    return res;
}

int shl_rvv_transpose_get_out_index(int32_t *dim, int32_t *idx, int32_t *permute, int32_t dim_count)
{
    int res = idx[permute[0]];
    for (int i = 1; i < dim_count; i++) {
        res = res * dim[i] + idx[permute[i]];
    }
    return res;
}
