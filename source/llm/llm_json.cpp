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

#include "json/json.hpp"
#include "llm/shl_llm_json.h"
using json = nlohmann::json;

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <fstream>
#include <iomanip>
static void *shl_llm_mmap(std::string path)
{
    int fd = open(path.c_str(), O_RDWR);
    struct stat sb;
    fstat(fd, &sb);
    void *addr = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        printf("mmap error\n");
        return NULL;
    }
    return addr;
}

static struct csinn_tensor *load_csinn_tensor(char *base, json &jdata, std::string name)
{
    struct csinn_tensor *ret = csinn_alloc_tensor(NULL);
    int64_t data_offset = jdata[name]["data_offset"];
    ret->data = base + data_offset;
    ret->dtype = jdata[name]["dtype"];
    ret->mtype = jdata[name]["mtype"];
    ret->dim_count = jdata[name]["dim_count"];
    for (int i = 0; i < ret->dim_count; i++) {
        ret->dim[i] = jdata[name]["dim"][std::to_string(i)];
    }
    std::string tensor_name = jdata[name]["name"];
    ret->name = (char *)shl_mem_alloc(tensor_name.length() + 1);
    strcpy(ret->name, tensor_name.c_str());
    return ret;
}

static void load_shl_model(struct shl_llm_model *model, char *base, json &jdata)
{
    model->tok_embeddings = load_csinn_tensor(base, jdata, "embd_weight");
    model->output_norm = load_csinn_tensor(base, jdata, "output_norm");
    model->output = load_csinn_tensor(base, jdata, "output");
    model->layers_num = jdata["layers_num"];

    for (int i = 0; i < model->layers_num; i++) {
        json jlayer = jdata["layer"][i];
        model->layers[i].wq =
            load_csinn_tensor(base, jlayer, "blk." + std::to_string(i) + ".attn_q.weight");
        model->layers[i].wk =
            load_csinn_tensor(base, jlayer, "blk." + std::to_string(i) + ".attn_k.weight");
        model->layers[i].wv =
            load_csinn_tensor(base, jlayer, "blk." + std::to_string(i) + ".attn_v.weight");
        model->layers[i].wo =
            load_csinn_tensor(base, jlayer, "blk." + std::to_string(i) + ".attn_output.weight");
        model->layers[i].w1 =
            load_csinn_tensor(base, jlayer, "blk." + std::to_string(i) + ".ffn_gate.weight");
        model->layers[i].w2 =
            load_csinn_tensor(base, jlayer, "blk." + std::to_string(i) + ".ffn_down.weight");
        model->layers[i].w3 =
            load_csinn_tensor(base, jlayer, "blk." + std::to_string(i) + ".ffn_up.weight");
        model->layers[i].attn_norm =
            load_csinn_tensor(base, jlayer, "blk." + std::to_string(i) + ".attn_norm.weight");
        model->layers[i].ffn_norm =
            load_csinn_tensor(base, jlayer, "blk." + std::to_string(i) + ".ffn_norm.weight");
    }
}

struct shl_llm_model *shl_llm_load_json(char *dir_path)
{
    struct shl_llm_model *model =
        (struct shl_llm_model *)shl_mem_alloc(sizeof(struct shl_llm_model));
    std::string model_dir = dir_path;
    std::ifstream json_file(model_dir + "shl.llm.json");
    std::string weight_path = model_dir + "shl.llm.weight.bm";
    json data = json::parse(json_file);
    if (data["config"]["shl_model_type"] == "weight_only") {
        void *base_addr = shl_llm_mmap(weight_path);
        load_shl_model(model, (char *)base_addr, data["model"]);
    } else {
        shl_debug_error("Unsupport json model file\n");
    }
    return model;
}

static int save_csinn_tensor(struct csinn_tensor *tensor, int64_t offset, json &jdata,
                             std::string name)
{
    jdata[name]["data_offset"] = offset;
    jdata[name]["dtype"] = tensor->dtype;
    jdata[name]["mtype"] = tensor->mtype;
    jdata[name]["dim_count"] = tensor->dim_count;
    for (int i = 0; i < tensor->dim_count; i++) {
        jdata[name]["dim"][std::to_string(i)] = tensor->dim[i];
    }
    jdata[name]["name"] = tensor->name;

    return CSINN_TRUE;
}

static int64_t dump_data(std::ofstream &file, struct csinn_tensor *tensor)
{
    int64_t size = 0;
    if (tensor->dtype == CSINN_DTYPE_FLOAT16) {
        size = csinn_tensor_byte_size(tensor);
    } else if (tensor->dtype == CSINN_DTYPE_FLOAT32) {
        size = csinn_tensor_byte_size(tensor);
    } else if (tensor->dtype == CSINN_DTYPE_INT8 && tensor->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) {
        size = csinn_tensor_size(tensor) + csinn_tensor_size(tensor) / 32 * sizeof(int16_t);
    } else if (tensor->dtype == CSINN_DTYPE_INT4 && tensor->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) {
        size = csinn_tensor_size(tensor) / 2 + csinn_tensor_size(tensor) / 32 * sizeof(int16_t);
    } else {
        shl_debug_error("unsupport dump data type\n");
    }
    file.write((const char *)tensor->data, size);
    return size;
}

static void save_shl_model(struct shl_llm_model *model, json &jdata, std::ofstream &weight)
{
    // void *base = model->tok_embeddings->data;
    int64_t base = 0;

    save_csinn_tensor(model->tok_embeddings, base, jdata, "embd_weight");
    base += dump_data(weight, model->tok_embeddings);
    save_csinn_tensor(model->output_norm, base, jdata, "output_norm");
    base += dump_data(weight, model->output_norm);
    save_csinn_tensor(model->output, base, jdata, "output");
    base += dump_data(weight, model->output);

    jdata["layers_num"] = model->layers_num;

    for (int i = 0; i < model->layers_num; i++) {
        json jlayer;
        save_csinn_tensor(model->layers[i].wq, base, jlayer,
                          "blk." + std::to_string(i) + ".attn_q.weight");
        base += dump_data(weight, model->layers[i].wq);
        save_csinn_tensor(model->layers[i].wk, base, jlayer,
                          "blk." + std::to_string(i) + ".attn_k.weight");
        base += dump_data(weight, model->layers[i].wk);
        save_csinn_tensor(model->layers[i].wv, base, jlayer,
                          "blk." + std::to_string(i) + ".attn_v.weight");
        base += dump_data(weight, model->layers[i].wv);
        save_csinn_tensor(model->layers[i].wo, base, jlayer,
                          "blk." + std::to_string(i) + ".attn_output.weight");
        base += dump_data(weight, model->layers[i].wo);
        save_csinn_tensor(model->layers[i].w1, base, jlayer,
                          "blk." + std::to_string(i) + ".ffn_gate.weight");
        base += dump_data(weight, model->layers[i].w1);
        save_csinn_tensor(model->layers[i].w2, base, jlayer,
                          "blk." + std::to_string(i) + ".ffn_down.weight");
        base += dump_data(weight, model->layers[i].w2);
        save_csinn_tensor(model->layers[i].w3, base, jlayer,
                          "blk." + std::to_string(i) + ".ffn_up.weight");
        base += dump_data(weight, model->layers[i].w3);
        save_csinn_tensor(model->layers[i].attn_norm, base, jlayer,
                          "blk." + std::to_string(i) + ".attn_norm.weight");
        base += dump_data(weight, model->layers[i].attn_norm);
        save_csinn_tensor(model->layers[i].ffn_norm, base, jlayer,
                          "blk." + std::to_string(i) + ".ffn_norm.weight");
        base += dump_data(weight, model->layers[i].ffn_norm);
        jdata["layer"].push_back(jlayer);
    }
}

int shl_llm_save_json(char *dir_path, struct shl_llm_model *model)
{
    std::string model_dir = dir_path;
    std::string json_path = model_dir + "shl.llm.json";
    std::ofstream json_file(json_path);
    std::ofstream weight_file;
    std::string weight_path = model_dir + "shl.llm.weight.bm";
    weight_file.open(weight_path, std::ios::out | std::ios::binary);

    json data;
    json jmodel;

    data["config"]["architectures"] = "llama2";
    data["config"]["shl_model_type"] = "weight_only";
    save_shl_model(model, jmodel, weight_file);
    data["model"] = jmodel;

    json_file << std::setw(4) << data << std::endl;
    return CSINN_TRUE;
}
