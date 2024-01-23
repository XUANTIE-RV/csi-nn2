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

static int get_index(int32_t *dim, int32_t *idx, int32_t dim_count)
{
    int res = idx[0];
    for (int i = 1; i < dim_count; i++) {
        res = res * dim[i] + idx[i];
    }
    return res;
}

int shl_rvv_strided_slice_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_strided_slice_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    }
    if (output->layout >= CSINN_LAYOUT_NC1C0 && output->layout <= CSINN_LAYOUT_NC1DHWC0) {
        int in_c1 = output->dim[1];
        const int packn = csrr_vlenb() / sizeof(__fp16);
        output->dim[1] = in_c1 * packn;
        output->dim[output->dim_count - 1] = 0;
        output->dim_count = output->dim_count - 1;
        if (output->layout == CSINN_LAYOUT_NC1DHWC0) {
            output->layout = CSINN_LAYOUT_NCDHW;
        } else if (output->layout == CSINN_LAYOUT_NC1HWC0) {
            output->layout = CSINN_LAYOUT_NCHW;
        } else if (output->layout == CSINN_LAYOUT_NC1WC0) {
            output->layout = CSINN_LAYOUT_NCW;
        } else if (output->layout == CSINN_LAYOUT_NC1C0) {
            output->layout = CSINN_LAYOUT_NC;
        }
    }

    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    for (int i = 0; i < input->dim_count; i++) {
        if (params->begin[i] < -input->dim[i]) params->begin[i] = -input->dim[i];
        if (params->begin[i] < 0) params->begin[i] += input->dim[i];
        if (params->begin[i] > input->dim[i]) params->begin[i] = input->dim[i];
        if (params->end[i] < -input->dim[i]) params->end[i] = -input->dim[i];
        if (params->end[i] < 0) params->end[i] += input->dim[i];
        if (params->end[i] > input->dim[i]) params->end[i] = input->dim[i];
    }

    if (input->dim_count == 1) {
        int b0 = params->begin[0];
        int e0 = params->end[0];
        int s0 = params->stride[0];

        __fp16 *in = (__fp16 *)input_data + b0;
        int size = output->dim[output->dim_count - 1];
        while (size > 0) {
            int vl = vsetvl_e16m1(size);
            vfloat16m1_t _in = vlse16_v_f16m1(in, s0 * sizeof(__fp16), vl);
            vse16_v_f16m1(output_data, _in, vl);
            in += s0 * vl;
            output_data += vl;
            size -= vl;
        }
    } else if (input->dim_count == 2) {
        int b0 = params->begin[0], b1 = params->begin[1];
        int e0 = params->end[0], e1 = params->end[1];
        int s0 = params->stride[0], s1 = params->stride[1];

        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            __fp16 *in = (__fp16 *)input_data + b * input->dim[1] + b1;
            int size = output->dim[output->dim_count - 1];
            while (size > 0) {
                int vl = vsetvl_e16m1(size);
                vfloat16m1_t _in = vlse16_v_f16m1(in, s1 * sizeof(__fp16), vl);
                vse16_v_f16m1(output_data, _in, vl);
                in += s1 * vl;
                output_data += vl;
                size -= vl;
            }
        }
    } else if (input->dim_count == 3) {
        int b0 = params->begin[0], b1 = params->begin[1], b2 = params->begin[2];
        int e0 = params->end[0], e1 = params->end[1], e2 = params->end[2];
        int s0 = params->stride[0], s1 = params->stride[1], s2 = params->stride[2];

        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            for (int c = b1; s1 > 0 ? (c < e1) : (c > e1); c += s1) {
                __fp16 *in = (__fp16 *)input_data + (b * input->dim[1] + c) * input->dim[2] + b2;
                int size = output->dim[output->dim_count - 1];
                while (size > 0) {
                    int vl = vsetvl_e16m1(size);
                    vfloat16m1_t _in = vlse16_v_f16m1(in, s2 * sizeof(__fp16), vl);
                    vse16_v_f16m1(output_data, _in, vl);
                    in += s2 * vl;
                    output_data += vl;
                    size -= vl;
                }
            }
        }
    } else if (input->dim_count == 4) {
        int b0 = params->begin[0], b1 = params->begin[1], b2 = params->begin[2],
            b3 = params->begin[3];
        int e0 = params->end[0], e1 = params->end[1], e2 = params->end[2], e3 = params->end[3];
        int s0 = params->stride[0], s1 = params->stride[1], s2 = params->stride[2],
            s3 = params->stride[3];

        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            for (int c = b1; s1 > 0 ? (c < e1) : (c > e1); c += s1) {
                for (int h = b2; s2 > 0 ? (h < e2) : (h > e2); h += s2) {
                    __fp16 *in = (__fp16 *)input_data + shl_ref_get_index(input->dim, b, c, h, b3);
                    int size = output->dim[output->dim_count - 1];
                    while (size > 0) {
                        int vl = vsetvl_e16m1(size);
                        vfloat16m1_t _in = vlse16_v_f16m1(in, s3 * sizeof(__fp16), vl);
                        vse16_v_f16m1(output_data, _in, vl);
                        in += s3 * vl;
                        output_data += vl;
                        size -= vl;
                    }
                }
            }
        }
    } else if (input->dim_count == 5) {
        int b0 = params->begin[0], b1 = params->begin[1], b2 = params->begin[2],
            b3 = params->begin[3], b4 = params->begin[4];
        int e0 = params->end[0], e1 = params->end[1], e2 = params->end[2], e3 = params->end[3],
            e4 = params->end[4];
        int s0 = params->stride[0], s1 = params->stride[1], s2 = params->stride[2],
            s3 = params->stride[3], s4 = params->stride[4];
        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            for (int c = b1; s1 > 0 ? (c < e1) : (c > e1); c += s1) {
                for (int h = b2; s2 > 0 ? (h < e2) : (h > e2); h += s2) {
                    for (int w = b3; s3 > 0 ? (w < e3) : (w > e3); w += s3) {
                        __fp16 *in =
                            (__fp16 *)input_data + shl_ref_get_index_5(input->dim, b, c, h, w, b4);
                        int size = output->dim[output->dim_count - 1];
                        while (size > 0) {
                            int vl = vsetvl_e16m1(size);
                            vfloat16m1_t _in = vlse16_v_f16m1(in, s4 * sizeof(__fp16), vl);
                            vse16_v_f16m1(output_data, _in, vl);
                            in += s4 * vl;
                            output_data += vl;
                            size -= vl;
                        }
                    }
                }
            }
        }
    } else {
        int32_t *begin = params->begin;
        int32_t *end = params->end;
        int32_t *stride = params->stride;

        int in_dim = input->dim_count;
        int32_t *idx = (int32_t *)shl_mem_alloc(in_dim * sizeof(int32_t));
        for (int i = 0; i < in_dim; i++) {
            idx[i] = begin[i];
        }
        int cur = 0;
        int inner_size = output->dim[output->dim_count - 1];

        while (stride[0] > 0 ? (idx[0] < end[0]) : (idx[0] > end[0])) {
            if (cur == input->dim_count - 1) {
                int size = inner_size;
                __fp16 *in = (__fp16 *)input_data + get_index(input->dim, idx, in_dim);
                while (size > 0) {
                    int vl = vsetvl_e16m1(size);
                    vfloat16m1_t _in = vlse16_v_f16m1(in, stride[cur] * sizeof(__fp16), vl);
                    vse16_v_f16m1(output_data, _in, vl);
                    in += stride[cur] * vl;
                    output_data += vl;
                    size -= vl;
                }
                if (cur == 0) {
                    break;
                }
                cur -= 1;
                idx[cur] += stride[cur];
            } else {
                if (stride[cur] > 0 ? (idx[cur] < end[cur]) : (idx[cur] > end[cur])) {
                    cur += 1;
                } else {
                    idx[cur] = begin[cur];
                    cur -= 1;
                    idx[cur] += stride[cur];
                }
            }
        }

        shl_mem_free(idx);
    }
    return CSINN_TRUE;
}