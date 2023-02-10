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

#include "csi_nn.h"
#include "shl_gref.h"
#include "shl_utils.h"

char *shl_bm_header_str()
{
    static char ret_str[4096] =
        "Heterogeneous Honey Badger binary model\n\nbinary model version 2.0\n\nHHB_VERSION ";
    csinn_version(ret_str + 79);
    return ret_str;
}

float check_bm_version(char *header_str)
{
    char *version_str = header_str + 62;
    float version_value = atof(version_str);
    if (version_value < 2.0) {
        shl_debug_warning(
            "Binary model version %f is deprecated! Will unsupport in next release.\n",
            version_value);
    }
    return version_value;
}

void shl_dump_bm_header(FILE *f)
{
    char *header = shl_bm_header_str();
    fwrite(header, 1, 4096, f);
}

void shl_dump_bm_section_info(FILE *f, struct shl_binary_model_section_info *info)
{
    if (info->section_info_size == 0) {
        info->section_info_size = 4096;
    }
    fwrite(info, 1, info->section_info_size, f);
}

union offset_or_pointer {
    int64_t offset;
    char *pointer;
};

static inline int64_t read_offset(void *ptr)
{
    if (sizeof(int64_t) != sizeof(char *)) {
        shl_debug_error("unsupport save_mode\n");
        return 0;
    }
    union offset_or_pointer op;
    op.pointer = ptr;
    return op.offset;
}

static inline char *offset_to_ptr(int64_t offset)
{
    if (sizeof(int64_t) != sizeof(char *)) {
        shl_debug_error("unsupport save_mode\n");
        return NULL;
    }
    union offset_or_pointer op;
    op.offset = offset;
    return op.pointer;
}

static inline void *ptr_offset_to_addr(void *base_addr, void *offset)
{
    return (char *)base_addr + read_offset(offset);
}

static inline void *copy_from_bm(char *bm_addr, int size)
{
    char *ret = shl_mem_alloc(size);
    memcpy(ret, bm_addr, size);
    return ret;
}

static inline int64_t encode_node_location_input(int layer, int number)
{
    int64_t ret = 0;
    int64_t input_magic = 0x81;
    if (layer >= 0xffff || number >= 0xff) {
        shl_debug_error("node_location arg is too large\n");
    }
    ret = (input_magic << 56) | ((layer & 0xffff) << 8) | (number & 0xff);
    return ret;
}

static inline int64_t encode_node_location_output(int layer, int number)
{
    int64_t ret = 0;
    int64_t output_magic = 0x82;
    if (layer >= 0xffff || number >= 0xff) {
        shl_debug_error("node_location arg is too large\n");
    }
    ret = (output_magic << 56) | ((layer & 0xffff) << 8) | (number & 0xff);
    return ret;
}

static inline void decode_node_location(int64_t code, int *layer, int *number)
{
    if (((code >> 56) & 0xff) != 0x81 && ((code >> 56) & 0xff) != 0x82) {
        shl_debug_error("node_location to decode error\n");
    }
    *layer = (code & 0xffff00) >> 8;
    *number = code & 0xff;
}

/* find node first appear location */
static int64_t find_node_first_location(struct shl_node *node, struct shl_ref_graph *graph)
{
    /* search output at first, normal node should appear as output */
    for (int i = 0; i < graph->layer_index; i++) {
        struct shl_node *n = graph->layer[i];
        for (int j = 0; j < n->out_num; j++) {
            if (node == n->out[j]) {
                return encode_node_location_output(i, j);
            }
        }
    }

    /* for model input or const, node only appears as input */
    for (int i = 0; i < graph->layer_index; i++) {
        struct shl_node *n = graph->layer[i];
        for (int k = 0; k < n->in_num; k++) {
            if (node == n->in[k]) {
                return encode_node_location_input(i, k);
            }
        }
    }

    /* no find */
    return 0;
}

static inline bool is_location_code(int64_t code)
{
    if (((code >> 56) & 0xff) == 0x81 || ((code >> 56) & 0xff) == 0x82) {
        return true;
    } else {
        return false;
    }
}

static inline bool is_location_output(int64_t code)
{
    if (((code >> 56) & 0xff) == 0x82) {
        return true;
    } else {
        return false;
    }
}

static char *tensor_dump(struct csinn_tensor *tensor, int *size)
{
    int tensor_size = sizeof(struct csinn_tensor);
    size_t name_size = strlen(tensor->name) + 1;
    tensor_size += name_size;
    int qinfo_size = tensor->quant_channel * sizeof(struct csinn_quant_info);
    tensor_size += qinfo_size;

    struct csinn_tensor *ret = shl_mem_alloc(tensor_size);
    /* ignore sess */
    ret->sess = 0;
    char *append_ptr = (char *)ret + sizeof(struct csinn_tensor);
    memcpy(append_ptr, tensor->name, name_size);
    /* offset from base */
    ret->name = (char *)(append_ptr - (char *)ret);
    append_ptr += name_size;
    memcpy(append_ptr, tensor->qinfo, qinfo_size);
    ret->qinfo = (struct csinn_quant_info *)(append_ptr - (char *)ret);

    ret->dtype = tensor->dtype;
    ret->mtype = tensor->mtype;
    ret->dim_count = tensor->dim_count;
    memcpy(ret->dim, tensor->dim, MAX_DIM * 4);
    ret->is_const = tensor->is_const;
    ret->layout = tensor->layout;
    ret->quant_channel = tensor->quant_channel;

    if (tensor->is_const) {
        ret = shl_mem_realloc(ret, tensor_size + csinn_tensor_byte_size(tensor), tensor_size);
        append_ptr = (char *)ret + tensor_size;
        memcpy(append_ptr, tensor->data, csinn_tensor_byte_size(tensor));
        ret->data = offset_to_ptr(tensor_size);
        tensor_size += csinn_tensor_byte_size(tensor);
    } else {
        /* ignore data */
        ret->data = 0;
    }

    *size = tensor_size;
    return (char *)ret;
}

static void tensor_load(struct csinn_tensor *dest, struct csinn_tensor *src)
{
    dest->data = src->data;
    dest->dtype = src->dtype;
    dest->mtype = src->mtype;
    memcpy(dest->dim, src->dim, MAX_DIM * 4);
    dest->dim_count = src->dim_count;
    dest->name = read_offset(src->name) + (char *)src;
    dest->layout = src->layout;
    if (src->quant_channel != dest->quant_channel && src->quant_channel != 0) {
        csinn_realloc_quant_info(dest, src->quant_channel);
    }
    dest->is_const = src->is_const;
    char *src_qinfo = (char *)src + read_offset(src->qinfo);
    memcpy(dest->qinfo, src_qinfo, sizeof(struct csinn_quant_info) * src->quant_channel);
    if (src->is_const) {
        dest->data = copy_from_bm(ptr_offset_to_addr(src, src->data), csinn_tensor_byte_size(src));
    }
}

static char *session_dump(struct csinn_session *sess, int *size)
{
    int sess_size = sizeof(struct csinn_session);

    char *input_buf[sess->input_num];
    int input_size[sess->input_num];
    char *output_buf[sess->output_num];
    int output_size[sess->output_num];

    for (int i = 0; i < sess->input_num; i++) {
        input_buf[i] = tensor_dump(sess->input[i], &input_size[i]);
        sess_size += input_size[i];
    }

    for (int i = 0; i < sess->output_num; i++) {
        output_buf[i] = tensor_dump(sess->output[i], &output_size[i]);
        sess_size += output_size[i];
    }

    sess_size += sizeof(struct csinn_tensor *) * (sess->input_num + sess->output_num);

    struct csinn_session *ret = shl_mem_alloc(sess_size);
    ret->input = shl_mem_alloc(sizeof(struct csinn_tensor *) * sess->input_num);
    ret->output = shl_mem_alloc(sizeof(struct csinn_tensor *) * sess->output_num);

    char *append_ptr = (char *)ret + sizeof(struct csinn_session);
    int input_offset = append_ptr - (char *)ret;
    append_ptr += sizeof(char *) * sess->input_num;
    for (int i = 0; i < sess->input_num; i++) {
        memcpy(append_ptr, input_buf[i], input_size[i]);
        ret->input[i] = (struct csinn_tensor *)(append_ptr - (char *)ret);
        append_ptr += input_size[i];
        shl_mem_free(input_buf[i]);
    }
    memcpy(input_offset + (char *)ret, ret->input, sizeof(char *) * sess->input_num);

    int output_offset = append_ptr - (char *)ret;
    append_ptr += sizeof(char *) * sess->output_num;
    for (int i = 0; i < sess->output_num; i++) {
        memcpy(append_ptr, output_buf[i], output_size[i]);
        ret->output[i] = (struct csinn_tensor *)(append_ptr - (char *)ret);
        append_ptr += output_size[i];
        shl_mem_free(output_buf[i]);
    }
    memcpy(output_offset + (char *)ret, ret->output, sizeof(char *) * sess->output_num);

    ret->base_dtype = sess->base_dtype;
    ret->base_layout = sess->base_layout;
    ret->base_api = sess->base_api;
    ret->base_run_mode = sess->base_run_mode;
    ret->base_quant_type = sess->base_quant_type;
    ret->model.bm_addr = sess->model.bm_addr;
    ret->model.bm_path = sess->model.bm_path;
    ret->model.bm_size = sess->model.bm_size;
    ret->model.priority = sess->model.priority;
    ret->model.save_mode = sess->model.save_mode;
    ret->debug_level = sess->debug_level;
    ret->profiler_level = sess->profiler_level;
    ret->input_num = sess->input_num;
    ret->output_num = sess->output_num;
    ret->input = (struct csinn_tensor **)offset_to_ptr(input_offset);
    ret->output = (struct csinn_tensor **)offset_to_ptr(output_offset);

    /* TODO: dump target data */

    *size = sess_size;
    return (char *)ret;
}

void shl_bm_session_load(struct csinn_session *dest, struct csinn_session *src)
{
    dest->base_quant_type = src->base_quant_type;
    dest->model.priority = src->model.priority;
    dest->base_api = src->base_api;
    dest->base_dtype = src->base_dtype;
    dest->base_run_mode = src->base_run_mode;
    dest->debug_level = src->debug_level;
    csinn_session_init(dest);
    csinn_set_input_number(src->input_num, dest);
    csinn_set_output_number(src->output_num, dest);

    src->input = (struct csinn_tensor **)((char *)src + read_offset(src->input));
    for (int i = 0; i < src->input_num; i++) {
        dest->input[i] = csinn_alloc_tensor(dest);
        struct csinn_tensor *src_input =
            (struct csinn_tensor *)((char *)src + read_offset(src->input[i]));
        tensor_load(dest->input[i], src_input);
        csinn_set_tensor_entry(dest->input[i], dest);
        csinn_set_input(i, dest->input[i], dest);
    }

    src->output = (struct csinn_tensor **)((char *)src + read_offset(src->output));
    for (int i = 0; i < src->output_num; i++) {
        dest->output[i] = csinn_alloc_tensor(dest);
        struct csinn_tensor *src_output =
            (struct csinn_tensor *)((char *)src + read_offset(src->output[i]));
        tensor_load(dest->output[i], src_output);
        csinn_set_tensor_entry(dest->output[i], dest);
        csinn_set_output(i, dest->output[i], dest);
    }
}

int shl_dump_bm_graph_info_section(FILE *f, struct csinn_session *sess)
{
    int size = 0;
    char *buf = session_dump(sess, &size);
    fwrite(buf, 1, size, f);
    shl_mem_free(buf);
    return size;
}

static char *node_dump(struct shl_node *node, int *size)
{
    int node_size = sizeof(struct shl_node);

    size_t name_size = strlen(node->name) + 1;
    node_size += name_size;

    int tensor_data_size;
    struct csinn_tensor *tensor = node->data;
    char *tensor_data_buf = tensor_dump(tensor, &tensor_data_size);
    node_size += tensor_data_size;

    struct shl_node *ret = shl_mem_alloc(node_size);

    char *append_ptr = (char *)ret + sizeof(struct shl_node);
    int name_offset = append_ptr - (char *)ret;
    memcpy(append_ptr, node->name, name_size);

    append_ptr += name_size;
    int data_offset = append_ptr - (char *)ret;
    memcpy(append_ptr, tensor_data_buf, tensor_data_size);

    ret->type = node->type;
    /* ignore in and out */
    ret->in = NULL;
    ret->out = NULL;
    ret->subgraph_idx = node->subgraph_idx;
    ret->in_num = node->in_num;
    ret->out_num = node->out_num;
    ret->name = offset_to_ptr(name_offset);
    ret->data = offset_to_ptr(data_offset);
    ret->ref_count = node->ref_count;
    ret->ref_count_init = node->ref_count_init;
    ret->visited = node->visited;
    /* ignore restricted map */
    ret->restricted_map = NULL;
    ret->restricted_map_num = node->restricted_map_num;

    *size = node_size;
    return (char *)ret;
}

static void node_load(struct shl_node *dest, struct shl_node *src)
{
    dest->type = src->type;
    dest->in = NULL;
    dest->out = NULL;
    dest->subgraph_idx = src->subgraph_idx;
    dest->in_num = src->in_num;
    dest->out_num = src->out_num;
    char *src_name = ptr_offset_to_addr(src, src->name);
    dest->name = copy_from_bm(src_name, strlen(src_name) + 1);
    dest->data = csinn_alloc_tensor(NULL);
    struct csinn_tensor *src_data = ptr_offset_to_addr(src, src->data);
    tensor_load(dest->data, src_data);
    dest->ref_count = src->ref_count;
    dest->ref_count_init = src->ref_count_init;
    dest->visited = src->visited;
    /* ignore restricted map */
    dest->restricted_map = NULL;
    dest->restricted_map_num = src->restricted_map_num;
}

static char *layer_data_dump(struct shl_node *layer, int *size)
{
    /* only dump op layer */
    if (layer->type >= CSINN_OP_SIZE) {
        *size = 0;
        return layer->data;
    }
    /* conv3d's params is biggest params */
    int layer_data_size = sizeof(struct csinn_conv3d_params);

    struct csinn_params_base *params = layer->data;
    int name_size = strlen(params->name) + 1;
    int extend_size = layer_data_size + name_size;
    /* ignore callback pointer space */
    struct csinn_params_base *ret = shl_mem_alloc(extend_size);
    memcpy(ret, layer->data, layer_data_size);
    ret->name = offset_to_ptr(layer_data_size);
    memcpy((char *)ret + layer_data_size, params->name, name_size);

    *size = extend_size;

    if (layer->type == CSINN_OP_RESHAPE) {
        struct csinn_reshape_params *reshape_params = layer->data;
        int shape_size = reshape_params->shape_num * sizeof(int32_t);
        ret = shl_mem_realloc(ret, extend_size + shape_size, extend_size);

        struct csinn_reshape_params *ret_reshape_params = (struct csinn_reshape_params *)ret;
        ret_reshape_params->shape = (int32_t *)offset_to_ptr(extend_size);
        memcpy((char *)ret + extend_size, reshape_params->shape, shape_size);
        extend_size += shape_size;
        *size = extend_size;
    } else if (layer->type == CSINN_OP_TRANSPOSE) {
        struct csinn_transpose_params *transpose_params = layer->data;
        int permute_size = transpose_params->permute_num * sizeof(int32_t);
        ret = shl_mem_realloc(ret, extend_size + permute_size, extend_size);

        struct csinn_transpose_params *ret_transpose_params = (struct csinn_transpose_params *)ret;
        ret_transpose_params->permute = (int32_t *)offset_to_ptr(extend_size);
        memcpy((char *)ret + extend_size, transpose_params->permute, permute_size);
        extend_size += permute_size;
        *size = extend_size;
    } else if (layer->type == CSINN_OP_PAD) {
        struct csinn_pad_params *pad_params = layer->data;
        if (pad_params->pad_num > 8) {
            shl_debug_error("Error: pad_num cannot = %d\n", pad_params->pad_num);
        }
        int pad_size = pad_params->pad_num * sizeof(int32_t);
        ret = shl_mem_realloc(ret, extend_size + pad_size * 2, extend_size);

        struct csinn_pad_params *ret_pad_params = (struct csinn_pad_params *)ret;
        ret_pad_params->pad_before = (int32_t *)offset_to_ptr(extend_size);
        memcpy((char *)ret + extend_size, pad_params->pad_before, pad_size);
        ret_pad_params->pad_after = (int32_t *)offset_to_ptr(extend_size + pad_size);
        memcpy((char *)ret + extend_size + pad_size, pad_params->pad_after, pad_size);
        extend_size += pad_size * 2;
        *size = extend_size;
    } else if (layer->type == CSINN_OP_SPLIT) {
        struct csinn_split_params *split_params = layer->data;
        int split_size = split_params->output_num * sizeof(int32_t);
        ret = shl_mem_realloc(ret, extend_size + split_size, extend_size);

        struct csinn_split_params *ret_split_params = (struct csinn_split_params *)ret;
        ret_split_params->split_index = (int32_t *)offset_to_ptr(extend_size);
        memcpy((char *)ret + extend_size, split_params->split_index, split_size);
        extend_size += split_size;
        *size = extend_size;
    } else if (layer->type == CSINN_OP_REDUCE_SUM || layer->type == CSINN_OP_REDUCE_MAX ||
               layer->type == CSINN_OP_REDUCE_MIN || layer->type == CSINN_OP_REDUCE_MEAN ||
               layer->type == CSINN_OP_REDUCE_PROD || layer->type == CSINN_OP_REDUCE_LOGSUMEXP ||
               layer->type == CSINN_OP_MEAN || layer->type == CSINN_OP_SUM ||
               layer->type == CSINN_OP_MAX || layer->type == CSINN_OP_MIN ||
               layer->type == CSINN_OP_PROD || layer->type == CSINN_OP_ARGMIN ||
               layer->type == CSINN_OP_ARGMAX || layer->type == CSINN_OP_ALL ||
               layer->type == CSINN_OP_ANY) {
        struct csinn_reduce_params *reduce_params = layer->data;

        int outer_size = reduce_params->n * sizeof(int32_t);
        ret = shl_mem_realloc(ret, extend_size + outer_size * 2, extend_size);
        struct csinn_reduce_params *ret_reduce_params = (struct csinn_reduce_params *)ret;
        ret_reduce_params->out_strides = (int32_t *)offset_to_ptr(extend_size);
        memcpy((char *)ret + extend_size, reduce_params->out_strides, outer_size);
        ret_reduce_params->out_extents = (int32_t *)offset_to_ptr(extend_size + outer_size);
        memcpy((char *)ret + extend_size + outer_size, reduce_params->out_extents, outer_size);
        extend_size += outer_size * 2;

        int inner_size = reduce_params->m * sizeof(int32_t);
        ret = shl_mem_realloc(ret, extend_size + inner_size * 2, extend_size);
        ret_reduce_params = (struct csinn_reduce_params *)ret;

        ret_reduce_params->inner_strides = (int32_t *)offset_to_ptr(extend_size);
        memcpy((char *)ret + extend_size, reduce_params->inner_strides, inner_size);
        ret_reduce_params->inner_extents = (int32_t *)offset_to_ptr(extend_size + inner_size);
        memcpy((char *)ret + extend_size + inner_size, reduce_params->inner_extents, inner_size);
        extend_size += inner_size * 2;

        int axis_size = reduce_params->axis_count * sizeof(int32_t);
        ret = shl_mem_realloc(ret, extend_size + axis_size, extend_size);
        ret_reduce_params = (struct csinn_reduce_params *)ret;

        ret_reduce_params->axis = (int32_t *)offset_to_ptr(extend_size);
        memcpy((char *)ret + extend_size, reduce_params->axis, axis_size);

        extend_size += axis_size;
        *size = extend_size;
    } else if (layer->type == CSINN_OP_BROADCOST) {
        struct csinn_broadcast_to_params *broadcast_params = layer->data;
        int broadcast_size = broadcast_params->shape_count * sizeof(int32_t);
        ret = shl_mem_realloc(ret, extend_size + broadcast_size, extend_size);

        struct csinn_broadcast_to_params *ret_broadcast_params =
            (struct csinn_broadcast_to_params *)ret;
        ret_broadcast_params->shape = (int32_t *)offset_to_ptr(extend_size);
        memcpy((char *)ret + extend_size, broadcast_params->shape, broadcast_size);
        extend_size += broadcast_size;
        *size = extend_size;
    } else if (layer->type == CSINN_OP_STRIDED_SLICE) {
        struct csinn_strided_slice_params *stride_slice_params = layer->data;
        int slice_size = stride_slice_params->slice_count * sizeof(int32_t);
        ret = shl_mem_realloc(ret, extend_size + slice_size * 3, extend_size);

        struct csinn_strided_slice_params *ret_stride_slice_params = (struct csinn_strided_slice_params *)ret;
        ret_stride_slice_params->begin = (int32_t *)offset_to_ptr(extend_size);
        memcpy((char *)ret + extend_size, stride_slice_params->begin, slice_size);
        ret_stride_slice_params->end = (int32_t *)offset_to_ptr(extend_size + slice_size);
        memcpy((char *)ret + extend_size + slice_size, stride_slice_params->end, slice_size);
        ret_stride_slice_params->stride = (int32_t *)offset_to_ptr(extend_size + slice_size * 2);
        memcpy((char *)ret + extend_size + slice_size * 2, stride_slice_params->stride, slice_size);
        extend_size += slice_size * 3;

        *size = extend_size;
    } else if (layer->type == CSINN_OP_L2N || layer->type == CSINN_OP_PROPOSAL ||
               layer->type == CSINN_OP_CROP || layer->type == CSINN_OP_SLICE ||
               layer->type == CSINN_OP_TILE || layer->type == CSINN_OP_SQUEEZE ||
               layer->type == CSINN_OP_SPACE_TO_BATCH_ND ||
               layer->type == CSINN_OP_BATCH_TO_SPACE_ND || layer->type == CSINN_OP_CACHE_MATMUL ||
               layer->type == CSINN_OP_CACHE_CONV1D) {
        shl_debug_error("%d params save unsupported\n", layer->type);
    }

    return (char *)ret;
}

static void layer_data_load(struct shl_node *dest, struct shl_node *src)
{
    /* only load op layer */
    if (src->type >= CSINN_OP_SIZE) {
        dest->data = src->data;
        return;
    }
    /* same with layer data dump, conv3d's params is biggest params */
    struct csinn_params_base *ret =
        copy_from_bm(ptr_offset_to_addr(src, src->data), sizeof(struct csinn_conv3d_params));

    char *name_addr = ptr_offset_to_addr(src, src->name);
    ret->name = copy_from_bm(name_addr, strlen(name_addr) + 1);
    ret->cb = shl_mem_alloc(sizeof(struct csinn_callback));
    // /* dest's input have been loaded */
    // struct csinn_tensor *input = dest->in[0]->data;
    // shl_op_callback_map(ret, src->type, input->dtype);

    if (src->type == CSINN_OP_RESHAPE) {
        struct csinn_reshape_params *reshape_params = (struct csinn_reshape_params *)ret;
        char *shape_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), reshape_params->shape);
        reshape_params->shape =
            copy_from_bm(shape_addr, reshape_params->shape_num * sizeof(int32_t));
    } else if (src->type == CSINN_OP_TRANSPOSE) {
        struct csinn_transpose_params *transpose_params = (struct csinn_transpose_params *)ret;
        char *permute_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), transpose_params->permute);
        transpose_params->permute =
            copy_from_bm(permute_addr, transpose_params->permute_num * sizeof(int32_t));
    } else if (src->type == CSINN_OP_PAD) {
        struct csinn_pad_params *pad_params = (struct csinn_pad_params *)ret;
        char *pad_before_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), pad_params->pad_before);
        pad_params->pad_before =
            copy_from_bm(pad_before_addr, pad_params->pad_num * sizeof(int32_t));
        char *pad_after_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), pad_params->pad_after);
        pad_params->pad_after = copy_from_bm(pad_after_addr, pad_params->pad_num * sizeof(int32_t));
    } else if (src->type == CSINN_OP_SPLIT) {
        struct csinn_split_params *split_params = (struct csinn_split_params *)ret;
        char *split_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), split_params->split_index);
        split_params->split_index =
            copy_from_bm(split_addr, split_params->output_num * sizeof(int32_t));
    } else if (src->type == CSINN_OP_REDUCE_SUM || src->type == CSINN_OP_REDUCE_MAX ||
               src->type == CSINN_OP_REDUCE_MIN || src->type == CSINN_OP_REDUCE_MEAN ||
               src->type == CSINN_OP_REDUCE_PROD || src->type == CSINN_OP_REDUCE_LOGSUMEXP ||
               src->type == CSINN_OP_MEAN || src->type == CSINN_OP_SUM ||
               src->type == CSINN_OP_MAX || src->type == CSINN_OP_MIN ||
               src->type == CSINN_OP_PROD || src->type == CSINN_OP_ARGMIN ||
               src->type == CSINN_OP_ARGMAX || src->type == CSINN_OP_ALL ||
               src->type == CSINN_OP_ANY) {
        struct csinn_reduce_params *reduce_params = (struct csinn_reduce_params *)ret;
        char *outer_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), reduce_params->out_extents);
        reduce_params->out_extents = copy_from_bm(outer_addr, reduce_params->n * sizeof(int32_t));
        outer_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), reduce_params->out_strides);
        reduce_params->out_strides = copy_from_bm(outer_addr, reduce_params->n * sizeof(int32_t));
        char *inner_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), reduce_params->inner_extents);
        reduce_params->inner_extents = copy_from_bm(inner_addr, reduce_params->m * sizeof(int32_t));
        inner_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), reduce_params->inner_strides);
        reduce_params->inner_strides = copy_from_bm(inner_addr, reduce_params->m * sizeof(int32_t));
        char *axis_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), reduce_params->axis);
        reduce_params->axis = copy_from_bm(axis_addr, reduce_params->axis_count * sizeof(int32_t));
    } else if (src->type == CSINN_OP_BROADCOST) {
        struct csinn_broadcast_to_params *broadcast_params =
            (struct csinn_broadcast_to_params *)ret;
        char *broadcast_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), broadcast_params->shape);
        broadcast_params->shape =
            copy_from_bm(broadcast_addr, broadcast_params->shape_count * sizeof(int32_t));
    } else if (src->type == CSINN_OP_STRIDED_SLICE) {
        struct csinn_strided_slice_params *stride_slice_params =
            (struct csinn_strided_slice_params *)ret;
        char *begin_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), stride_slice_params->begin);
        stride_slice_params->begin =
            copy_from_bm(begin_addr, stride_slice_params->slice_count * sizeof(int32_t));
        char *end_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), stride_slice_params->end);
        stride_slice_params->end =
            copy_from_bm(end_addr, stride_slice_params->slice_count * sizeof(int32_t));
        char *stride_addr =
            ptr_offset_to_addr(ptr_offset_to_addr(src, src->data), stride_slice_params->stride);
        stride_slice_params->stride =
            copy_from_bm(stride_addr, stride_slice_params->slice_count * sizeof(int32_t));
    } else if (src->type == CSINN_OP_L2N || src->type == CSINN_OP_PROPOSAL ||
               src->type == CSINN_OP_CROP || src->type == CSINN_OP_SLICE ||
               src->type == CSINN_OP_TILE || src->type == CSINN_OP_SQUEEZE ||
               src->type == CSINN_OP_SPACE_TO_BATCH_ND || src->type == CSINN_OP_BATCH_TO_SPACE_ND ||
               src->type == CSINN_OP_CACHE_MATMUL || src->type == CSINN_OP_CACHE_CONV1D) {
        shl_debug_error("%d params load unsupported\n", src->type);
    }

    dest->data = ret;
}

static char *layer_dump(struct shl_node *layer, int *size, int layer_index,
                        struct shl_ref_graph *graph)
{
    int layer_size = sizeof(struct shl_node);

    char *input_buf[layer->in_num];
    int input_size[layer->in_num];
    char *output_buf[layer->out_num];
    int output_size[layer->out_num];

    for (int i = 0; i < layer->in_num; i++) {
        int64_t location = find_node_first_location(layer->in[i], graph);

        if (is_location_output(location)) {
            /* have appeared as output */
            input_buf[i] = offset_to_ptr(location);
        } else {
            /* first appear */
            input_buf[i] = node_dump(layer->in[i], &input_size[i]);
            layer_size += input_size[i];
        }
    }

    for (int i = 0; i < layer->out_num; i++) {
        if (layer->out[i] != NULL) {
            output_buf[i] = node_dump(layer->out[i], &output_size[i]);
            layer_size += output_size[i];
        } else {
            output_buf[i] = NULL;
            output_size[i] = 0;
        }
    }

    layer_size += sizeof(struct shl_node *) * (layer->in_num + layer->out_num);
    int name_size = strlen(layer->name) + 1;
    layer_size += name_size;
    int layer_data_size;
    char *layer_data_buf = layer_data_dump(layer, &layer_data_size);
    layer_size += layer_data_size;

    struct shl_node *ret = shl_mem_alloc(layer_size);
    ret->in = shl_mem_alloc(sizeof(struct shl_node *) * layer->in_num);
    ret->out = shl_mem_alloc(sizeof(struct shl_node *) * layer->out_num);

    char *append_ptr = (char *)ret + sizeof(struct shl_node);
    int input_offset = append_ptr - (char *)ret;
    append_ptr += sizeof(char *) * layer->in_num;
    for (int i = 0; i < layer->in_num; i++) {
        if (is_location_code(read_offset(input_buf[i]))) {
            ret->in[i] = (struct shl_node *)input_buf[i];
        } else {
            memcpy(append_ptr, input_buf[i], input_size[i]);
            ret->in[i] = (struct shl_node *)(append_ptr - (char *)ret);
            append_ptr += input_size[i];
            shl_mem_free(input_buf[i]);
        }
    }
    memcpy(input_offset + (char *)ret, ret->in, sizeof(char *) * layer->in_num);

    int output_offset = append_ptr - (char *)ret;
    append_ptr += sizeof(char *) * layer->out_num;
    for (int i = 0; i < layer->out_num; i++) {
        if (output_size[i] != 0) {
            memcpy(append_ptr, output_buf[i], output_size[i]);
            ret->out[i] = (struct shl_node *)(append_ptr - (char *)ret);
            append_ptr += output_size[i];
            shl_mem_free(output_buf[i]);
        }
    }
    memcpy(output_offset + (char *)ret, ret->out, sizeof(char *) * layer->out_num);

    int name_offset = append_ptr - (char *)ret;
    memcpy(append_ptr, layer->name, name_size);
    append_ptr += name_size;

    int data_offset = append_ptr - (char *)ret;
    if (layer_data_size != 0) {
        memcpy(append_ptr, layer_data_buf, layer_data_size);
    }

    ret->type = layer->type;
    ret->in = (struct shl_node **)offset_to_ptr(input_offset);
    ret->out = (struct shl_node **)offset_to_ptr(output_offset);
    ret->subgraph_idx = layer->subgraph_idx;
    ret->in_num = layer->in_num;
    ret->out_num = layer->out_num;
    ret->name = offset_to_ptr(name_offset);
    ret->data = offset_to_ptr(data_offset);
    ret->ref_count = layer->ref_count;
    ret->ref_count_init = layer->ref_count_init;
    ret->visited = layer->visited;
    /* ignore restricted map */
    ret->restricted_map = NULL;
    ret->restricted_map_num = layer->restricted_map_num;

    *size = layer_size;
    return (char *)ret;
}

static void layer_load(struct shl_node *dest, struct shl_node *src, struct shl_ref_graph *graph)
{
    dest->type = src->type;
    dest->subgraph_idx = src->subgraph_idx;
    dest->in_num = src->in_num;
    dest->out_num = src->out_num;
    char *name_addr = ptr_offset_to_addr(src, src->name);
    dest->name = copy_from_bm(name_addr, strlen(name_addr) + 1);
    dest->ref_count = src->ref_count;
    dest->ref_count_init = src->ref_count_init;
    dest->visited = src->visited;
    /* ignore restricted map */
    dest->restricted_map = NULL;
    dest->restricted_map_num = src->restricted_map_num;

    dest->in =
        copy_from_bm(ptr_offset_to_addr(src, src->in), sizeof(struct shl_node *) * src->in_num);
    for (int i = 0; i < src->in_num; i++) {
        int64_t dest_in_offset = read_offset(dest->in[i]);
        if (is_location_code(dest_in_offset)) {
            int layer, number;
            decode_node_location(dest_in_offset, &layer, &number);
            dest->in[i] = graph->layer[layer]->out[number];
        } else {
            struct shl_node *src_in = ptr_offset_to_addr(src, dest->in[i]);
            struct shl_node *dest_in = shl_mem_alloc(sizeof(struct shl_node));
            node_load(dest_in, src_in);
            dest->in[i] = dest_in;
        }
    }
    dest->out =
        copy_from_bm(ptr_offset_to_addr(src, src->out), sizeof(struct shl_node *) * src->out_num);
    for (int i = 0; i < src->out_num; i++) {
        struct shl_node *src_out = ptr_offset_to_addr(src, dest->out[i]);
        struct shl_node *dest_out = shl_mem_alloc(sizeof(struct shl_node));
        node_load(dest_out, src_out);
        dest->out[i] = dest_out;
    }

    /* after input load */
    layer_data_load(dest, src);
}

static char *graph_dump(struct shl_ref_graph *graph, int *size)
{
    int graph_size = sizeof(struct shl_ref_graph);

    char *layer_buf[graph->layer_index];
    int layer_size[graph->layer_index];
    for (int i = 0; i < graph->layer_index; i++) {
        layer_buf[i] = layer_dump(graph->layer[i], &layer_size[i], i, graph);
        graph_size += layer_size[i];
    }

    char *output_buf[graph->output_num];
    int output_size[graph->output_num];
    for (int i = 0; i < graph->output_num; i++) {
        int64_t location = find_node_first_location(graph->output[i], graph);
        if (location) {
            output_buf[i] = NULL;
            output_size[i] = 0;
        } else {
            /* global graph have sub graph's output */
            output_buf[i] = node_dump(graph->output[i], &output_size[i]);
            graph_size += output_size[i];
        }
    }

    graph_size += sizeof(char *) * graph->layer_index +
                  sizeof(int64_t) * (graph->input_num + graph->output_num);

    struct shl_ref_graph *ret = shl_mem_alloc(graph_size);

    ret->layer = shl_mem_alloc(sizeof(char *) * graph->layer_index);

    char *append_ptr = (char *)ret + sizeof(struct shl_ref_graph);

    int layer_offset = append_ptr - (char *)ret;
    append_ptr += sizeof(char *) * graph->layer_index;
    for (int i = 0; i < graph->layer_index; i++) {
        memcpy(append_ptr, layer_buf[i], layer_size[i]);
        shl_debug_debug("%s layer%d: graph start offset %d\n", __func__, i,
                        append_ptr - (char *)ret);
        ret->layer[i] = (struct shl_node *)(append_ptr - (char *)ret);
        append_ptr += layer_size[i];
        shl_mem_free(layer_buf[i]);
    }
    memcpy(layer_offset + (char *)ret, ret->layer, sizeof(char *) * graph->layer_index);

    /* update input and output pointer to be index */
    int64_t input_index[graph->input_num];
    int64_t output_index[graph->output_num];

    for (int i = 0; i < graph->input_num; i++) {
        int64_t location = find_node_first_location(graph->input[i], graph);
        if (location) {
            input_index[i] = location;
        } else {
            shl_debug_debug("%s: error graph input node\n", __func__);
        }
    }

    int input_offset = append_ptr - (char *)ret;
    memcpy(append_ptr, input_index, sizeof(int64_t) * graph->input_num);
    append_ptr += sizeof(int64_t) * graph->input_num;

    for (int i = 0; i < graph->output_num; i++) {
        int64_t location = find_node_first_location(graph->output[i], graph);
        if (location) {
            output_index[i] = location;
        } else {
            if (output_size[i] != 0) {
                memcpy(append_ptr, output_buf[i], output_size[i]);
                output_index[i] = append_ptr - (char *)ret;
                append_ptr += output_size[i];
                shl_mem_free(output_buf[i]);
            }
        }
    }

    int output_offset = append_ptr - (char *)ret;
    memcpy(output_offset + (char *)ret, output_index, sizeof(int64_t) * graph->output_num);
    // append_ptr += sizeof(int64_t) * graph->output_num;

    ret->input_num = graph->input_num;
    ret->output_num = graph->output_num;
    shl_mem_free(ret->input);
    shl_mem_free(ret->output);
    shl_mem_free(ret->layer);
    ret->input = (struct shl_node **)offset_to_ptr(input_offset);
    ret->output = (struct shl_node **)offset_to_ptr(output_offset);
    ret->layer = (struct shl_node **)offset_to_ptr(layer_offset);
    ret->layer_size = graph->layer_size;
    ret->layer_index = graph->layer_index;

    *size = graph_size;
    return (char *)ret;
}

void shl_bm_graph_struct_load(struct shl_ref_graph *dest, struct shl_ref_graph *src)
{
    dest->input_num = src->input_num;
    dest->output_num = src->output_num;
    dest->layer_size = src->layer_size;
    dest->layer_index = src->layer_index;

    dest->layer = copy_from_bm(ptr_offset_to_addr(src, src->layer),
                               sizeof(struct shl_node **) * src->layer_size);
    for (int i = 0; i < dest->layer_index; i++) {
        struct shl_node *src_layer = ptr_offset_to_addr(src, dest->layer[i]);
        shl_debug_debug("%s layer%d: graph start offset %d\n", __func__, i, dest->layer[i]);
        struct shl_node *dest_layer = shl_mem_alloc(sizeof(struct shl_node));
        layer_load(dest_layer, src_layer, dest);
        dest->layer[i] = dest_layer;
    }
    dest->input = copy_from_bm(ptr_offset_to_addr(src, src->input),
                               sizeof(struct shl_node *) * src->input_num);
    for (int i = 0; i < dest->input_num; i++) {
        int64_t dest_input_offset = read_offset(dest->input[i]);
        if (is_location_code(dest_input_offset)) {
            int layer, number;
            decode_node_location(dest_input_offset, &layer, &number);
            dest->input[i] = dest->layer[layer]->in[number];
        } else {
            shl_debug_debug("%s: error input encode\n", __func__);
        }
    }
    dest->output = copy_from_bm(ptr_offset_to_addr(src, src->output),
                                sizeof(struct shl_node *) * src->output_num);
    for (int i = 0; i < dest->output_num; i++) {
        int64_t dest_output_offset = read_offset(dest->output[i]);
        if (is_location_code(dest_output_offset)) {
            int layer, number;
            decode_node_location(dest_output_offset, &layer, &number);
            dest->output[i] = dest->layer[layer]->out[number];
        } else {
            /* global graph have sub graph's output */
            struct shl_node *src_out = ptr_offset_to_addr(src, dest->output[i]);
            struct shl_node *dest_out = shl_mem_alloc(sizeof(struct shl_node));
            node_load(dest_out, src_out);
            dest->output[i] = dest_out;
        }
    }
}

int shl_dump_bm_graph_struct_section(FILE *f, struct shl_ref_graph *ggraph)
{
    int size = 0;
    char *buf = graph_dump(ggraph, &size);
    fwrite(buf, 1, size, f);
    shl_mem_free(buf);
    return size;
}

/**
 * @addtogroup SESSION
 * @{
 */
struct csinn_session *__attribute__((weak)) csinn_import_binary_model(char *bm_addr)
{
    struct shl_binary_model_section_info *sinfo =
        (struct shl_binary_model_section_info *)(bm_addr + 4096);
    struct csinn_session *bm_sess =
        (struct csinn_session *)(bm_addr + sinfo->sections->info_offset * 4096);
    struct csinn_session *sess = csinn_alloc_session();
    shl_bm_session_load(sess, bm_sess);
    float version = check_bm_version(bm_addr);
    if (version == 2.0) {
        if (sess->base_run_mode != CSINN_RM_NPU_GRAPH) {
            /* load binary model in GREF */
            sess->model.bm_addr = bm_addr;
        } else {
            sess->model.bm_addr = bm_addr + sinfo->sections->graph_offset * 4096;
            sess->model.bm_size = sinfo->sections->graph_size;
        }
    } else if (version == 1.0) {
        sess->model.bm_addr = bm_addr + sinfo->sections->graph_offset * 4096;
        sess->model.bm_size = sinfo->sections->graph_size;
    } else {
        shl_debug_error("Unsupport binary model\n");
    }

    csinn_load_binary_model(sess);
    return sess;
}
/**
 * @}
 */
