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

#include "csi_nn.h"
#include "shl_utils.h"

char *shl_bm_header_str()
{
    static char ret_str[4096] =
        "Heterogeneous Honey Badger binary model\n\nbinary model version 1.0\n\nHHB_VERSION ";
    csinn_version(ret_str + 79);
    return ret_str;
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

static inline int32_t read_offset(void *ptr)
{
    /* when 64bit, get 32bit too */
    int32_t ret = *(int32_t *)&ptr;
    return ret;
}

static inline char *offset_to_ptr(int offset)
{
    char *ret;
    *(int *)(&ret) = offset;
    return ret;
}

static char *tensor_dump(struct csinn_tensor *tensor, int *size)
{
    int tensor_size = sizeof(struct csinn_tensor);
    size_t name_size = strlen(tensor->name);
    tensor_size += name_size;
    int qinfo_size = tensor->quant_channel * sizeof(struct csinn_quant_info);
    tensor_size += qinfo_size;

    struct csinn_tensor *ret = shl_mem_alloc(tensor_size);
    /* ignore data */
    ret->data = 0;
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

void shl_dump_bm_graph_info_section(FILE *f, struct csinn_session *sess)
{
    int size = 0;
    char *buf = session_dump(sess, &size);
    fwrite(buf, 1, size, f);
    shl_mem_free(buf);
}

struct csinn_session *__attribute__((weak)) csinn_import_binary_model(char *bm_addr)
{
    struct shl_binary_model_section_info *sinfo =
        (struct shl_binary_model_section_info *)(bm_addr + 4096);
    struct csinn_session *bm_sess =
        (struct csinn_session *)(bm_addr + sinfo->sections->info_offset * 4096);
    struct csinn_session *sess = csinn_alloc_session();
    shl_bm_session_load(sess, bm_sess);
    sess->model.bm_addr = bm_addr + sinfo->sections->graph_offset * 4096;
    sess->model.bm_size = sinfo->sections->graph_size;
    csinn_load_binary_model(sess);
    return sess;
}
