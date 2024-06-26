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

static int check_input_dim_count(struct csinn_tensor *input, struct csinn_tensor *output)
{
    int target_dim_count = output->dim_count;
    if (output->layout >= CSINN_LAYOUT_NC1C0 && output->layout <= CSINN_LAYOUT_NC1DHWC0) {
        target_dim_count -= 1;
    }
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        target_dim_count += 1;
    }
    if (input->dim_count <= target_dim_count) {
        return CSINN_TRUE;
    }
    return CSINN_FALSE;
}

static inline void infer_layout_by_dim(struct csinn_tensor *t)
{
    if (t->layout >= CSINN_LAYOUT_NC1C0 && t->layout <= CSINN_LAYOUT_NC1DHWC0) {
        if (t->dim_count == 3)
            t->layout = CSINN_LAYOUT_NC1C0;
        else if (t->dim_count == 4)
            t->layout = CSINN_LAYOUT_NC1WC0;
        else if (t->dim_count == 5)
            t->layout = CSINN_LAYOUT_NC1HWC0;
        else if (t->dim_count == 6)
            t->layout = CSINN_LAYOUT_NC1DHWC0;
    } else if (t->layout >= CSINN_LAYOUT_N && t->layout <= CSINN_LAYOUT_NCDHW) {
        if (t->dim_count == 1)
            t->layout = CSINN_LAYOUT_N;
        else if (t->dim_count == 2)
            t->layout = CSINN_LAYOUT_NC;
        else if (t->dim_count == 3)
            t->layout = CSINN_LAYOUT_NCW;
        else if (t->dim_count == 4)
            t->layout = CSINN_LAYOUT_NCHW;
        else if (t->dim_count == 5)
            t->layout = CSINN_LAYOUT_NCDHW;
        else if (t->dim_count == 6)
            t->layout = CSINN_LAYOUT_NLCDHW;
    }
}

static void adjust_input_dim(struct csinn_tensor *input, struct csinn_tensor *output)
{
    int in_dim_count = input->dim_count;
    int target_dim_count = output->dim_count;
    if (output->layout >= CSINN_LAYOUT_NC1C0 && output->layout <= CSINN_LAYOUT_NC1DHWC0) {
        target_dim_count -= 1;
    }
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        target_dim_count += 1;
    }
    if (in_dim_count < target_dim_count) {
        input->dim_count = target_dim_count;
        infer_layout_by_dim(input);
        for (int i = target_dim_count - 1; i >= target_dim_count - in_dim_count; i--) {
            input->dim[i] = input->dim[i - (target_dim_count - in_dim_count)];
        }
        for (int i = target_dim_count - in_dim_count - 1; i >= 0; i--) {
            input->dim[i] = 1;
        }
    } else if (in_dim_count > target_dim_count) {
        bool reduce = true;
        int b = in_dim_count - target_dim_count;
        for (int i = 0; i < b; i++) {
            if (input->dim[i] != 1) {
                reduce = false;
                break;
            }
        }
        if (reduce) {
            input->dim_count = target_dim_count;
            infer_layout_by_dim(input);
            for (int i = 0; i < target_dim_count; i++) {
                input->dim[i] = input->dim[i + b];
            }
        }
    }
}

static int check_broadcast_rule(struct csinn_tensor *input, struct csinn_tensor *output)
{
    for (int i = 0; i < input->dim_count; i++) {
        if ((input->dim[input->dim_count - i - 1] != output->dim[output->dim_count - i - 1]) &&
            (input->dim[input->dim_count - i - 1] != 1)) {
            return CSINN_FALSE;
        }
    }
    return CSINN_TRUE;
}

static int broadcast_get_index(int32_t *dim, int32_t *idx, int32_t dim_count)
{
    int res = 0;
    for (int i = 0; i < dim_count; i++) {
        if (dim[i] != 1) {
            int tmp = idx[i];
            for (int j = i + 1; j < dim_count; j++) {
                tmp *= dim[j];
            }
            res += tmp;
        }
    }
    return res;
}

static int layout_try_ndarray_to_nc1xc0(struct csinn_tensor *t, int packn)
{
    if (t->layout >= CSINN_LAYOUT_NC && t->layout <= CSINN_LAYOUT_NCDHW) {
        if (t->dim[1] % packn == 0) {
            t->dim[1] /= packn;
            t->dim_count = t->dim_count + 1;
            t->dim[t->dim_count - 1] = packn;
        } else if (t->dim[1] == 1) {
            t->dim_count = t->dim_count + 1;
            t->dim[t->dim_count - 1] = 1;
        } else {
            shl_debug_error("The dimension of tensor do not meet the rules of broadcast!");
            return CSINN_FALSE;
        }
        if (t->layout == CSINN_LAYOUT_NCDHW) {
            t->layout = CSINN_LAYOUT_NC1DHWC0;
        } else if (t->layout == CSINN_LAYOUT_NCHW) {
            t->layout = CSINN_LAYOUT_NC1HWC0;
        } else if (t->layout == CSINN_LAYOUT_NCW) {
            t->layout = CSINN_LAYOUT_NC1WC0;
        } else if (t->layout == CSINN_LAYOUT_NC) {
            t->layout = CSINN_LAYOUT_NC1C0;
        }
        return CSINN_TRUE;
    }
    return CSINN_FALSE;
}

static int layout_try_nc1xc0_to_ndarray(struct csinn_tensor *t)
{
    if (t->layout >= CSINN_LAYOUT_NC1C0 && t->layout <= CSINN_LAYOUT_NC1DHWC0) {
        int in_c1 = t->dim[1];
        int in_c0 = t->dim[t->dim_count - 1];
        t->dim[1] = in_c1 * in_c0;
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
        return CSINN_TRUE;
    }
    return CSINN_FALSE;
}

static void transform_layout_weight_to_activation(struct csinn_tensor *t)
{
    if (t->layout == CSINN_LAYOUT_O) {
        t->layout = CSINN_LAYOUT_N;
    } else if (t->layout == CSINN_LAYOUT_O) {
        t->layout = CSINN_LAYOUT_NC;
    } else if (t->layout == CSINN_LAYOUT_OI) {
        t->layout = CSINN_LAYOUT_NCW;
    } else if (t->layout == CSINN_LAYOUT_OIHW) {
        t->layout = CSINN_LAYOUT_NCHW;
    } else if (t->layout == CSINN_LAYOUT_OIDHW) {
        t->layout = CSINN_LAYOUT_NCDHW;
    }
}

static void tensor_try_ndarray_to_nc1xc0_fp32(struct csinn_tensor *t)
{
    const int packn = csrr_vlenb() / sizeof(float);
    int batch = t->dim[0];
    int in_c = t->dim[1];
    int inner_size = 1;
    for (int i = 2; i < t->dim_count; i++) {
        inner_size *= t->dim[i];
    }

    if (layout_try_ndarray_to_nc1xc0(t, packn)) {
        if (t->dim[t->dim_count - 1] != 1) {
            float *src = t->data;
            float *dst = (float *)shl_mem_alloc(csinn_tensor_byte_size(t));

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
            shl_mem_free(t->data);
            t->data = dst;
        }
    }
}

static void tensor_try_ndarray_to_nc1xc0_fp16(struct csinn_tensor *t)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    int batch = t->dim[0];
    int in_c = t->dim[1];
    int inner_size = 1;
    for (int i = 2; i < t->dim_count; i++) {
        inner_size *= t->dim[i];
    }

    if (layout_try_ndarray_to_nc1xc0(t, packn)) {
        if (t->dim[t->dim_count - 1] != 1) {
            __fp16 *src = t->data;
            __fp16 *dst = (__fp16 *)shl_mem_alloc(csinn_tensor_byte_size(t));

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
            shl_mem_free(t->data);
            t->data = dst;
        }
    }
}

static void tensor_try_ndarray_to_nc1xc0_int8(struct csinn_tensor *t)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int batch = t->dim[0];
    int in_c = t->dim[1];
    int inner_size = 1;
    for (int i = 2; i < t->dim_count; i++) {
        inner_size *= t->dim[i];
    }

    if (layout_try_ndarray_to_nc1xc0(t, packn)) {
        if (t->dim[t->dim_count - 1] != 1) {
            int8_t *src = t->data;
            int8_t *dst = (int8_t *)shl_mem_alloc(csinn_tensor_byte_size(t));

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
            shl_mem_free(t->data);
            t->data = dst;
        }
    }
}

int shl_rvv_binary_op_broadcast_fp32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                     struct csinn_tensor *output, void *binary_op_callback[])
{
    adjust_input_dim(input0, output);
    adjust_input_dim(input1, output);

    if (!check_input_dim_count(input0, output)) {
        shl_debug_error("input0 dim_count greater than output!\n");
        return CSINN_FALSE;
    }
    if (!check_input_dim_count(input1, output)) {
        shl_debug_error("input1 dim_count greater than output!\n");
        return CSINN_FALSE;
    }

    const int packn = csrr_vlenb() / sizeof(float);

    struct csinn_tensor *in1_extra;
    bool in1_extra_flag = false;

    if (input0->layout >= CSINN_LAYOUT_NC1C0 && input0->layout <= CSINN_LAYOUT_NC1DHWC0) {
        in1_extra = csinn_alloc_tensor(NULL);
        csinn_tensor_copy(in1_extra, input1);
        in1_extra->data = shl_mem_alloc(csinn_tensor_byte_size(input1));
        memcpy(in1_extra->data, input1->data, csinn_tensor_byte_size(input1));
        in1_extra_flag = true;
        if (input1->is_const) {
            transform_layout_weight_to_activation(in1_extra);
        }
        input1 = in1_extra;
        tensor_try_ndarray_to_nc1xc0_fp32(input1);
        layout_try_ndarray_to_nc1xc0(output, packn);
    } else if (input0->layout >= CSINN_LAYOUT_N && input0->layout <= CSINN_LAYOUT_NCDHW) {
        if (input1->layout >= CSINN_LAYOUT_NC1C0 && input1->layout <= CSINN_LAYOUT_NC1DHWC0) {
            shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(input1);
        }
        layout_try_nc1xc0_to_ndarray(output);
    }

    if (!check_broadcast_rule(input0, output)) {
        shl_debug_error("The dimension of input0 do not meet the rules of broadcast!\n");
        return CSINN_FALSE;
    }
    if (!check_broadcast_rule(input1, output)) {
        shl_debug_error("The dimension of input1 do not meet the rules of broadcast!\n");
        return CSINN_FALSE;
    }

    float *input0_data = (float *)input0->data;
    float *input1_data = (float *)input1->data;
    float *output_data = (float *)output->data;

    int32_t *in0_dim = input0->dim;
    int32_t *in1_dim = input1->dim;
    int32_t *out_dim = output->dim;
    int32_t dim_count = output->dim_count;

    // Mark an index that traverses each dimension.
    int32_t *idx = (int32_t *)shl_mem_alloc(dim_count * sizeof(int32_t));
    int cur = 0;

    void (*binary_op)();
    if (in0_dim[dim_count - 1] == in1_dim[dim_count - 1]) {
        binary_op = binary_op_callback[CSINN_BROADCAST_VV];
    } else if (in1_dim[dim_count - 1] == 1) {
        binary_op = binary_op_callback[CSINN_BROADCAST_VS];
    } else if (in0_dim[dim_count - 1] == 1) {
        binary_op = binary_op_callback[CSINN_BROADCAST_SV];
    }

    // Work like a stack, "push" the higher dimension until reach the last dimension,
    // "pop" when done traversing current dimension.
    while (idx[0] < out_dim[0]) {
        if (cur == dim_count - 1) {
            // Do broadcast in the last dimension
            float *in0_ptr = input0_data + broadcast_get_index(in0_dim, idx, dim_count);
            float *in1_ptr = input1_data + broadcast_get_index(in1_dim, idx, dim_count);
            float *out_ptr = output_data + broadcast_get_index(out_dim, idx, dim_count);
            binary_op(in0_ptr, in1_ptr, out_ptr, out_dim[cur]);
            if (cur == 0) {
                break;
            }
            cur -= 1;
            idx[cur] += 1;
        } else {
            // If the current index is less than the current dim size, traverse the next dimension;
            // Otherwise, set the index to 0, and return to the previous dimension.
            if (idx[cur] < out_dim[cur]) {
                cur += 1;
            } else {
                idx[cur] = 0;
                cur -= 1;
                idx[cur] += 1;
            }
        }
    }

    shl_mem_free(idx);
    if (in1_extra_flag) {
        shl_mem_free(in1_extra->data);
        csinn_free_tensor(in1_extra);
    }

    return CSINN_TRUE;
}

int shl_rvv_binary_op_broadcast_fp16(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                     struct csinn_tensor *output, void *binary_op_callback[])
{
    adjust_input_dim(input0, output);
    adjust_input_dim(input1, output);

    if (!check_input_dim_count(input0, output)) {
        shl_debug_error("input0 dim_count greater than output!\n");
        return CSINN_FALSE;
    }
    if (!check_input_dim_count(input1, output)) {
        shl_debug_error("input1 dim_count greater than output!\n");
        return CSINN_FALSE;
    }

    const int packn = csrr_vlenb() / sizeof(__fp16);

    struct csinn_tensor *in1_extra;
    bool in1_extra_flag = false;

    if (input0->layout >= CSINN_LAYOUT_NC1C0 && input0->layout <= CSINN_LAYOUT_NC1DHWC0) {
        in1_extra = csinn_alloc_tensor(NULL);
        csinn_tensor_copy(in1_extra, input1);
        in1_extra->data = shl_mem_alloc(csinn_tensor_byte_size(input1));
        memcpy(in1_extra->data, input1->data, csinn_tensor_byte_size(input1));
        in1_extra_flag = true;
        if (input1->is_const) {
            transform_layout_weight_to_activation(in1_extra);
        }
        input1 = in1_extra;
        tensor_try_ndarray_to_nc1xc0_fp16(input1);
        layout_try_ndarray_to_nc1xc0(output, packn);
    } else if (input0->layout >= CSINN_LAYOUT_N && input0->layout <= CSINN_LAYOUT_NCDHW) {
        if (input1->layout >= CSINN_LAYOUT_NC1C0 && input1->layout <= CSINN_LAYOUT_NC1DHWC0) {
            shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input1);
        }
        layout_try_nc1xc0_to_ndarray(output);
    }

    if (!check_broadcast_rule(input0, output)) {
        shl_debug_error("The dimension of input0 do not meet the rules of broadcast!\n");
        return CSINN_FALSE;
    }
    if (!check_broadcast_rule(input1, output)) {
        shl_debug_error("The dimension of input1 do not meet the rules of broadcast!\n");
        return CSINN_FALSE;
    }

    __fp16 *input0_data = (__fp16 *)input0->data;
    __fp16 *input1_data = (__fp16 *)input1->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int32_t *in0_dim = input0->dim;
    int32_t *in1_dim = input1->dim;
    int32_t *out_dim = output->dim;
    int32_t dim_count = output->dim_count;

    int32_t *idx = (int32_t *)shl_mem_alloc(dim_count * sizeof(int32_t));
    int cur = 0;

    void (*binary_op)();
    if (in0_dim[dim_count - 1] == in1_dim[dim_count - 1]) {
        binary_op = binary_op_callback[CSINN_BROADCAST_VV];
    } else if (in1_dim[dim_count - 1] == 1) {
        binary_op = binary_op_callback[CSINN_BROADCAST_VS];
    } else if (in0_dim[dim_count - 1] == 1) {
        binary_op = binary_op_callback[CSINN_BROADCAST_SV];
    }

    while (idx[0] < out_dim[0]) {
        if (cur == dim_count - 1) {
            __fp16 *in0_ptr = input0_data + broadcast_get_index(in0_dim, idx, dim_count);
            __fp16 *in1_ptr = input1_data + broadcast_get_index(in1_dim, idx, dim_count);
            __fp16 *out_ptr = output_data + broadcast_get_index(out_dim, idx, dim_count);
            binary_op(in0_ptr, in1_ptr, out_ptr, out_dim[cur]);
            if (cur == 0) {
                break;
            }
            cur -= 1;
            idx[cur] += 1;
        } else {
            if (idx[cur] < out_dim[cur]) {
                cur += 1;
            } else {
                idx[cur] = 0;
                cur -= 1;
                idx[cur] += 1;
            }
        }
    }

    shl_mem_free(idx);
    if (in1_extra_flag) {
        shl_mem_free(in1_extra->data);
        csinn_free_tensor(in1_extra);
    }

    return CSINN_TRUE;
}

int shl_rvv_binary_op_broadcast_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                     struct csinn_tensor *output, void *binary_op_callback[])
{
    adjust_input_dim(input0, output);
    adjust_input_dim(input1, output);

    if (!check_input_dim_count(input0, output)) {
        shl_debug_error("input0 dim_count greater than output!\n");
        return CSINN_FALSE;
    }
    if (!check_input_dim_count(input1, output)) {
        shl_debug_error("input1 dim_count greater than output!\n");
        return CSINN_FALSE;
    }

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;

    struct csinn_tensor *in1_extra;
    bool in1_extra_flag = false;

    if (input0->layout >= CSINN_LAYOUT_NC1C0 && input0->layout <= CSINN_LAYOUT_NC1DHWC0) {
        in1_extra = csinn_alloc_tensor(NULL);
        csinn_tensor_copy(in1_extra, input1);
        in1_extra->data = shl_mem_alloc(csinn_tensor_byte_size(input1));
        memcpy(in1_extra->data, input1->data, csinn_tensor_byte_size(input1));
        in1_extra_flag = true;
        if (input1->is_const) {
            transform_layout_weight_to_activation(in1_extra);
        }
        input1 = in1_extra;
        tensor_try_ndarray_to_nc1xc0_int8(input1);
        layout_try_ndarray_to_nc1xc0(output, packn);
    } else if (input0->layout >= CSINN_LAYOUT_N && input0->layout <= CSINN_LAYOUT_NCDHW) {
        if (input1->layout >= CSINN_LAYOUT_NC1C0 && input1->layout <= CSINN_LAYOUT_NC1DHWC0) {
            shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input1);
        }
        layout_try_nc1xc0_to_ndarray(output);
    }

    if (!check_broadcast_rule(input0, output)) {
        shl_debug_error("The dimension of input0 do not meet the rules of broadcast!\n");
        return CSINN_FALSE;
    }
    if (!check_broadcast_rule(input1, output)) {
        shl_debug_error("The dimension of input1 do not meet the rules of broadcast!\n");
        return CSINN_FALSE;
    }

    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;

    int32_t *in0_dim = input0->dim;
    int32_t *in1_dim = input1->dim;
    int32_t *out_dim = output->dim;
    int32_t dim_count = output->dim_count;

    int32_t *idx = (int32_t *)shl_mem_alloc(dim_count * sizeof(int32_t));
    int cur = 0;

    void (*binary_op)();
    if (in0_dim[dim_count - 1] == in1_dim[dim_count - 1]) {
        binary_op = binary_op_callback[CSINN_BROADCAST_VV];
    } else if (in1_dim[dim_count - 1] == 1) {
        binary_op = binary_op_callback[CSINN_BROADCAST_VS];
    } else if (in0_dim[dim_count - 1] == 1) {
        binary_op = binary_op_callback[CSINN_BROADCAST_SV];
    }

    float scale[3] = {input0->qinfo->scale, input1->qinfo->scale, output->qinfo->scale};
    int32_t zero_point[3] = {input0->qinfo->zero_point, input1->qinfo->zero_point,
                             output->qinfo->zero_point};

    while (idx[0] < out_dim[0]) {
        if (cur == dim_count - 1) {
            int8_t *in0_ptr = input0_data + broadcast_get_index(in0_dim, idx, dim_count);
            int8_t *in1_ptr = input1_data + broadcast_get_index(in1_dim, idx, dim_count);
            int8_t *out_ptr = output_data + broadcast_get_index(out_dim, idx, dim_count);
            binary_op(in0_ptr, in1_ptr, out_ptr, out_dim[cur], scale, zero_point);
            if (cur == 0) {
                break;
            }
            cur -= 1;
            idx[cur] += 1;
        } else {
            if (idx[cur] < out_dim[cur]) {
                cur += 1;
            } else {
                idx[cur] = 0;
                cur -= 1;
                idx[cur] += 1;
            }
        }
    }

    shl_mem_free(idx);
    if (in1_extra_flag) {
        shl_mem_free(in1_extra->data);
        csinn_free_tensor(in1_extra);
    }

    return CSINN_TRUE;
}
