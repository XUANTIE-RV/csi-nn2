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

#include "shl_gref.h"

int shl_gref_graph_insert(struct shl_node *node, struct shl_ref_graph *graph)
{
    if (graph->layer_size == 0 || graph->layer_index == graph->layer_size - 1) {
        graph->layer_size += 128;
        graph->layer = shl_mem_realloc(graph->layer, graph->layer_size * sizeof(struct shl_node *),
                                       (graph->layer_size - 128) * sizeof(struct shl_node *));
    }
    graph->layer[graph->layer_index] = node;
    graph->layer_index++;
    return CSINN_TRUE;
}

int shl_gref_siso_op(struct csinn_tensor *input, struct csinn_tensor *output, int op, void *params)
{
    struct csinn_params_base *ptr = params;
    struct shl_node *layer = shl_node_alloc(op, ptr->name, 1, 1, params);
    struct shl_node *in0 = (struct shl_node *)input->data;
    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_in(layer, in0, 0);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(input->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

int shl_gref_diso_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, int op, void *params)
{
    struct csinn_params_base *ptr = params;
    struct shl_node *layer = shl_node_alloc(op, ptr->name, 2, 1, params);
    struct shl_node *in0;
    struct shl_node *in1;
    if (input0->is_const) {
        in0 = shl_node_const_var_alloc(input0->name, input0);
    } else {
        in0 = (struct shl_node *)input0->data;
    }
    if (input1->is_const) {
        in1 = shl_node_const_var_alloc(input1->name, input1);
    } else {
        in1 = (struct shl_node *)input1->data;
    }
    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_in(layer, in0, 0);
    shl_node_add_in(layer, in1, 1);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(input0->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

/* single input double const single output */
int shl_gref_sidcso_op(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *const0, struct csinn_tensor *const1, int op,
                       void *params)
{
    struct csinn_params_base *ptr = params;
    struct shl_node *layer = shl_node_alloc(op, ptr->name, 3, 1, params);
    struct shl_node *in0 = (struct shl_node *)input->data;
    struct shl_node *in1 = shl_node_const_var_alloc(const0->name, const0);
    struct shl_node *in2 = shl_node_const_var_alloc(const1->name, const1);
    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_in(layer, in0, 0);
    shl_node_add_in(layer, in1, 1);
    shl_node_add_in(layer, in2, 2);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(input->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

int shl_gref_siso_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output, void *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    output->layout = input->layout;
    output->dim_count = input->dim_count;
    for (int i = 0; i < input->dim_count; i++) {
        output->dim[i] = input->dim[i];
    }
    return CSINN_TRUE;
}

int shl_gref_diso_infer_shape(struct csinn_tensor *input0, struct csinn_tensor *input1,
                              struct csinn_tensor *output, void *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(input0);
    shl_tensor_try_nc1xc0_to_ndarray_shape(input1);
    int32_t dim_count =
        input0->dim_count > input1->dim_count ? input0->dim_count : input1->dim_count;

    for (int i = 0; i < dim_count; ++i) {
        const int d1 = input0->dim_count - 1 - i;
        const int d2 = input1->dim_count - 1 - i;
        const int s1 = d1 >= 0 ? input0->dim[d1] : 1;
        const int s2 = d2 >= 0 ? input1->dim[d2] : 1;
        if (s1 == s2) {
            output->dim[dim_count - 1 - i] = s1;
        } else if (s1 == 1) {
            output->dim[dim_count - 1 - i] = s2;
        } else if (s2 == 1) {
            output->dim[dim_count - 1 - i] = s1;
        } else {
            shl_debug_error("%s: Invalid shapes for broadcast!\n", __func__);
            return CSINN_FALSE;
        }
    }

    if (input0->dim_count == input1->dim_count) {
        output->layout = input0->is_const ? input1->layout : input0->layout;
    } else {
        bool use_input0_layout = input0->dim_count >= input1->dim_count ? true : false;
        if ((use_input0_layout && input0->is_const) || (!use_input0_layout && input1->is_const)) {
            shl_debug_error("%s: Diso shape infer fail!\n", __func__);
        } else {
            output->layout = use_input0_layout ? input0->layout : input1->layout;
        }
    }

    output->dim_count = dim_count;
    return CSINN_TRUE;
}

int shl_gref_pooling2d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_pool_params *params)
{
    int c, h, w;
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    if (input->layout == CSINN_LAYOUT_NCHW) {
        c = 1;
        h = 2;
        w = 3;
    } else if (input->layout == CSINN_LAYOUT_NHWC) {
        h = 1;
        w = 2;
        c = 3;
    } else {
        shl_debug_error("%s: Invalid input tensor layout!\n", __func__);
        return CSINN_UNSUPPORT_LAYOUT;
    }

    int32_t in_h = input->dim[h];
    int32_t in_w = input->dim[w];
    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;
    int32_t padding_h = params->pad_top + params->pad_down;
    int32_t padding_w = params->pad_left + params->pad_right;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    int32_t ceil_h = 0;
    int32_t ceil_w = 0;
    if (params->ceil_mode == 1) {
        ceil_h = stride_h - 1;
        ceil_w = stride_w - 1;
    }
    output->layout = input->layout;
    output->dim_count = 4;
    output->dim[0] = input->dim[0];
    output->dim[c] = input->dim[c];
    output->dim[h] = (in_h + padding_h - kernel_h + ceil_h) / stride_h + 1;
    output->dim[w] = (in_w + padding_w - kernel_w + ceil_w) / stride_w + 1;

    return CSINN_TRUE;
}

int shl_gref_pooling3d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_pool_params *params)
{
    int c, d, h, w;
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    if (input->layout == CSINN_LAYOUT_NCDHW) {
        c = 1;
        d = 2;
        h = 3;
        w = 4;
    } else if (input->layout == CSINN_LAYOUT_NDHWC) {
        d = 1;
        h = 2;
        w = 3;
        c = 4;
    } else {
        shl_debug_error("%s: Invalid input tensor layout!\n", __func__);
        return CSINN_UNSUPPORT_LAYOUT;
    }

    int32_t in_d = input->dim[d];
    int32_t in_h = input->dim[h];
    int32_t in_w = input->dim[w];
    int32_t kernel_d = params->filter_depth;
    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;
    int32_t padding_d = params->pad_front + params->pad_back;
    int32_t padding_h = params->pad_top + params->pad_down;
    int32_t padding_w = params->pad_left + params->pad_right;
    int32_t stride_d = params->stride_depth;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    int32_t ceil_d = 0;
    int32_t ceil_h = 0;
    int32_t ceil_w = 0;
    if (params->ceil_mode == 1) {
        ceil_d = stride_d - 1;
        ceil_h = stride_h - 1;
        ceil_w = stride_w - 1;
    }
    output->layout = input->layout;
    output->dim_count = 5;
    output->dim[0] = input->dim[0];
    output->dim[c] = input->dim[c];
    output->dim[d] = (in_d + padding_d - kernel_d + ceil_d) / stride_d + 1;
    output->dim[h] = (in_h + padding_h - kernel_h + ceil_h) / stride_h + 1;
    output->dim[w] = (in_w + padding_w - kernel_w + ceil_w) / stride_w + 1;

    return CSINN_TRUE;
}

int shl_gref_global_pooling2d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_pool_params *params)
{
    int c, h, w;
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    if (input->layout == CSINN_LAYOUT_NCHW) {
        c = 1;
        h = 2;
        w = 3;
    } else if (input->layout == CSINN_LAYOUT_NHWC) {
        h = 1;
        w = 2;
        c = 3;
    } else {
        shl_debug_error("%s: Invalid input tensor layout!\n", __func__);
        return CSINN_UNSUPPORT_LAYOUT;
    }
    output->layout = input->layout;
    output->dim_count = 4;
    output->dim[0] = input->dim[0];
    output->dim[c] = input->dim[c];
    output->dim[h] = 1;
    output->dim[w] = 1;

    return CSINN_TRUE;
}

int shl_gref_reduce_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_reduce_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    if (params->axis[0] == -1) {
        output->dim_count = 1;
        output->dim[0] = 1;
    } else {
        output->dim_count = input->dim_count;
        for (int i = 0; i < output->dim_count; i++) {
            if (params->axis[0] == i) {
                output->dim[i] = 1;
            } else {
                output->dim[i] = input->dim[i];
            }
        }
    }
    return CSINN_TRUE;
}

int shl_gref_segment_infer_shape(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                 struct csinn_tensor *output, struct csinn_segment_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(input0);
    output->dim_count = input0->dim_count;
    output->dim[0] = params->num_segments;
    for (int i = 1; i < output->dim_count; i++) {
        output->dim[i] = input0->dim[i];
    }
    return CSINN_TRUE;
}

int shl_gref_stride_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_reduce_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    output->dim_count = input->dim_count - 1;
    for (int i = 0; i < input->dim_count; i++) {
        if (i < params->axis[0]) {
            output->dim[i] = input->dim[i];
        } else if (i > params->axis[0]) {
            output->dim[i - 1] = input->dim[i];
        }
    }
    return CSINN_TRUE;
}

void shl_tensor_try_nc1xc0_to_ndarray_shape(struct csinn_tensor *t)
{
    if (t->layout >= CSINN_LAYOUT_NC1C0 && t->layout <= CSINN_LAYOUT_NC1DHWC0) {
        int in_c1 = t->dim[1];
        int in_c0 = t->dim[t->dim_count - 1];
        t->dim[1] = in_c1 * in_c0;
        t->dim[t->dim_count - 1] = 0;
        t->dim_count = t->dim_count - 1;
    }
    if (t->layout == CSINN_LAYOUT_NC1DHWC0) {
        t->layout = CSINN_LAYOUT_NCDHW;
    } else if (t->layout == CSINN_LAYOUT_NC1HWC0) {
        t->layout = CSINN_LAYOUT_NCHW;
    } else if (t->layout == CSINN_LAYOUT_NC1WC0) {
        t->layout = CSINN_LAYOUT_NCW;
    } else if (t->layout == CSINN_LAYOUT_NC1C0) {
        t->layout = CSINN_LAYOUT_NC;
    }
}
