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

#include "reference/ref.h"

/* fixme: */
int shl_ref_strided_slice(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_strided_slice_params *params)
{
    int output_size = 1;
    int outer_size = 1;
    int inner_size = csinn_tensor_size(input);
    int copy_num = 1;

    void *buffer1 = shl_mem_alloc(csinn_tensor_byte_size(input));
    void *buffer2 = shl_mem_alloc(csinn_tensor_byte_size(input));
    memcpy(buffer1, input->data, csinn_tensor_byte_size(input));

    int element_size = 1;  // float:4 fp16:2
    switch (input->dtype) {
        case CSINN_DTYPE_UINT8:
        case CSINN_DTYPE_INT8:
            break;
        case CSINN_DTYPE_INT16:
        case CSINN_DTYPE_UINT16:
        case CSINN_DTYPE_FLOAT16:
        case CSINN_DTYPE_BFLOAT16:
            element_size = 2;
            break;
        case CSINN_DTYPE_INT32:
        case CSINN_DTYPE_UINT32:
        case CSINN_DTYPE_FLOAT32:
            element_size = 4;
            break;
        case CSINN_DTYPE_FLOAT64:
            element_size = 8;
            break;
        default:
            shl_debug_error("unsupport input dtype for strided_slice\n");
            return CSINN_FALSE;
    }

    for (int slice_dim = 0; slice_dim < params->slice_count; slice_dim++) {
        int begin = params->begin[slice_dim];
        int end = params->end[slice_dim];
        int stride = params->stride[slice_dim];

        if (begin < -input->dim[slice_dim]) begin = -input->dim[slice_dim];
        if (begin < 0) begin += input->dim[slice_dim];
        if (begin > input->dim[slice_dim]) begin = input->dim[slice_dim];
        if (end < -input->dim[slice_dim]) end = -input->dim[slice_dim];
        if (end < 0) end += input->dim[slice_dim];
        if (end > input->dim[slice_dim]) end = input->dim[slice_dim];

        inner_size /= input->dim[slice_dim];
        outer_size *= copy_num;

        copy_num = 1 + (abs(end - begin) - 1) / abs(stride);
        output_size *= copy_num;

        float *p1 = buffer1;
        float *p2 = buffer2;
        for (int n = 0; n < outer_size; n++) {
            for (int i = begin; stride > 0 ? (i < end) : (i > end); i += stride) {
                memcpy(p2, p1 + i * inner_size, inner_size * element_size);
                p2 += inner_size;
            }
            p1 += inner_size * input->dim[slice_dim];
        }
        float *tmp = buffer1;
        buffer1 = buffer2;
        buffer2 = tmp;
    }
    output_size = output_size * inner_size;
    memcpy(output->data, buffer1, output_size * element_size);
    shl_mem_free(buffer1);
    shl_mem_free(buffer2);
    return CSINN_TRUE;
}

/* reference
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/strided_slice.h
 * // TODO: support onnx axis param?
 */
int shl_ref_strided_slice_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_strided_slice_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

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
        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            *output_data++ = input_data[b];
        }
    } else if (input->dim_count == 2) {
        int b0 = params->begin[0], b1 = params->begin[1];
        int e0 = params->end[0], e1 = params->end[1];
        int s0 = params->stride[0], s1 = params->stride[1];

        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            for (int c = b1; s1 > 0 ? (c < e1) : (c > e1); c += s1) {
                int32_t input_index = b * input->dim[1] + c;
                *output_data++ = input_data[input_index];
            }
        }
    } else if (input->dim_count == 3) {
        int b0 = params->begin[0], b1 = params->begin[1], b2 = params->begin[2];
        int e0 = params->end[0], e1 = params->end[1], e2 = params->end[2];
        int s0 = params->stride[0], s1 = params->stride[1], s2 = params->stride[2];

        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            for (int c = b1; s1 > 0 ? (c < e1) : (c > e1); c += s1) {
                for (int h = b2; s2 > 0 ? (h < e2) : (h > e2); h += s2) {
                    int32_t input_index = (b * input->dim[1] + c) * input->dim[2] + h;
                    *output_data++ = input_data[input_index];
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
                    for (int w = b3; s3 > 0 ? (w < e3) : (w > e3); w += s3) {
                        int32_t input_index = shl_ref_get_index(input->dim, b, c, h, w);
                        *output_data++ = input_data[input_index];
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
                        for (int d = b4; s4 > 0 ? (d < e4) : (d > e4); d += s4) {
                            int32_t input_index = shl_ref_get_index_5(input->dim, b, c, h, w, d);
                            *output_data++ = input_data[input_index];
                        }
                    }
                }
            }
        }
    } else {
        shl_debug_error("unsupport input dim_count=%d\n", input->dim_count);
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

#if __riscv
int shl_ref_strided_slice_f16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_strided_slice_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    for (int i = 0; i < params->slice_count; i++) {
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
        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            *output_data++ = input_data[b];
        }
    } else if (input->dim_count == 2) {
        int b0 = params->begin[0], b1 = params->begin[1];
        int e0 = params->end[0], e1 = params->end[1];
        int s0 = params->stride[0], s1 = params->stride[1];

        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            for (int c = b1; s1 > 0 ? (c < e1) : (c > e1); c += s1) {
                int32_t input_index = b * input->dim[1] + c;
                *output_data++ = input_data[input_index];
            }
        }
    } else if (input->dim_count == 3) {
        int b0 = params->begin[0], b1 = params->begin[1], b2 = params->begin[2];
        int e0 = params->end[0], e1 = params->end[1], e2 = params->end[2];
        int s0 = params->stride[0], s1 = params->stride[1], s2 = params->stride[2];

        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            for (int c = b1; s1 > 0 ? (c < e1) : (c > e1); c += s1) {
                for (int h = b2; s2 > 0 ? (h < e2) : (h > e2); h += s2) {
                    int32_t input_index = (b * input->dim[1] + c) * input->dim[2] + h;
                    *output_data++ = input_data[input_index];
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
                    for (int w = b3; s3 > 0 ? (w < e3) : (w > e3); w += s3) {
                        int32_t input_index = shl_ref_get_index(input->dim, b, c, h, w);
                        *output_data++ = input_data[input_index];
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
                        for (int d = b4; s4 > 0 ? (d < e4) : (d > e4); d += s4) {
                            int32_t input_index = shl_ref_get_index_5(input->dim, b, c, h, w, d);
                            *output_data++ = input_data[input_index];
                        }
                    }
                }
            }
        }
    } else {
        shl_debug_error("unsupport input dim_count=%d\n", input->dim_count);
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}
#endif  // __riscv

int shl_ref_strided_slice_i8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_strided_slice_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    for (int i = 0; i < params->slice_count; i++) {
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
        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            *output_data++ = input_data[b];
        }
    } else if (input->dim_count == 2) {
        int b0 = params->begin[0], b1 = params->begin[1];
        int e0 = params->end[0], e1 = params->end[1];
        int s0 = params->stride[0], s1 = params->stride[1];

        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            for (int c = b1; s1 > 0 ? (c < e1) : (c > e1); c += s1) {
                int32_t input_index = b * input->dim[1] + c;
                *output_data++ = input_data[input_index];
            }
        }
    } else if (input->dim_count == 3) {
        int b0 = params->begin[0], b1 = params->begin[1], b2 = params->begin[2];
        int e0 = params->end[0], e1 = params->end[1], e2 = params->end[2];
        int s0 = params->stride[0], s1 = params->stride[1], s2 = params->stride[2];

        for (int b = b0; s0 > 0 ? (b < e0) : (b > e0); b += s0) {
            for (int c = b1; s1 > 0 ? (c < e1) : (c > e1); c += s1) {
                for (int h = b2; s2 > 0 ? (h < e2) : (h > e2); h += s2) {
                    int32_t input_index = (b * input->dim[1] + c) * input->dim[2] + h;
                    *output_data++ = input_data[input_index];
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
                    for (int w = b3; s3 > 0 ? (w < e3) : (w > e3); w += s3) {
                        int32_t input_index = shl_ref_get_index(input->dim, b, c, h, w);
                        *output_data++ = input_data[input_index];
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
                        for (int d = b4; s4 > 0 ? (d < e4) : (d > e4); d += s4) {
                            int32_t input_index = shl_ref_get_index_5(input->dim, b, c, h, w, d);
                            *output_data++ = input_data[input_index];
                        }
                    }
                }
            }
        }
    } else {
        shl_debug_error("unsupport input dim_count=%d\n", input->dim_count);
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

int shl_ref_strided_slice_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_strided_slice_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_strided_slice_f32);
}
