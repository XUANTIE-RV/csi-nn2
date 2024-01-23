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
#ifndef INCLUDE_SHL_LLM_H_
#define INCLUDE_SHL_LLM_H_
#ifdef __cplusplus
extern "C" {
#endif
#include "backend/reference/ref.h"
#include "csinn/csi_nn.h"
#include "graph/shl_gref.h"
#include "shl_utils.h"

struct shl_llm_layer {
    // normalization
    struct csinn_tensor *attn_norm;
    struct csinn_tensor *attn_norm_b;
    struct csinn_tensor *attn_norm_2;
    struct csinn_tensor *attn_norm_2_b;
    struct csinn_tensor *attn_q_norm;
    struct csinn_tensor *attn_q_norm_b;
    struct csinn_tensor *attn_k_norm;
    struct csinn_tensor *attn_k_norm_b;

    // attention
    struct csinn_tensor *wq;
    struct csinn_tensor *wk;
    struct csinn_tensor *wv;
    struct csinn_tensor *wo;
    struct csinn_tensor *wqkv;

    // attention bias
    struct csinn_tensor *bo;
    struct csinn_tensor *bqkv;

    // normalization
    struct csinn_tensor *ffn_norm;
    struct csinn_tensor *ffn_norm_b;

    // ff
    struct csinn_tensor *w1;  // ffn_gate
    struct csinn_tensor *w2;  // ffn_down
    struct csinn_tensor *w3;  // ffn_up

    // ff bias
    struct csinn_tensor *b2;  // ffn_down
    struct csinn_tensor *b3;  // ffn_up
};

struct shl_llm_model {
    struct csinn_tensor *tok_embeddings;
    struct csinn_tensor *pos_embeddings;
    struct csinn_tensor *tok_norm;
    struct csinn_tensor *tok_norm_b;

    struct csinn_tensor *output_norm;
    struct csinn_tensor *output_norm_b;
    struct csinn_tensor *output;

    struct shl_llm_layer layers[32];
    int layers_num;
};

struct shl_transformer_block {
    int layer_id;
    struct csinn_session *session;
    struct csinn_tensor *cache_k;
    void *cache_k_buffer;
    struct csinn_tensor *cache_v;
    void *cache_v_buffer;
    struct csinn_tensor *freqs_cis;
    struct csinn_tensor *mask;
};

struct shl_llm_ctx {
    int layers_num;
    struct shl_transformer_block **transformer_block;
    struct csinn_session *embeding_session;
    struct csinn_session *output_session;
    struct csinn_tensor *start_pos;
    char *path;

    struct shl_llm_model *shl_model;
};

struct shl_llm_input {
    int32_t n_tokens;
    int32_t *token;
    int32_t *pos;
};

struct llama_config {
    int dim;
    int multiple_of;
    int n_heads;
    int n_layers;
    float nor_eps;
    int vocab_size;

    struct shl_llm_model *shl_model;
};

struct shl_llm_ctx *llama2_build(struct llama_config *config);
int llm_run(struct shl_llm_ctx *ctx, struct shl_llm_input *embd);
int shl_block_quantize(struct csinn_tensor *src, struct csinn_tensor *dst);
struct csinn_tensor *quantize_tensor(struct csinn_tensor *src, enum csinn_mem_type_enum mtype);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_LLM_H_
