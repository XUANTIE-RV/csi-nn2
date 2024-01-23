#include "llm/shl_llm.h"

static char *alloc_name(char *name)
{
    char *ret = shl_mem_alloc(strlen(name));
    sprintf(ret, "%s", name);
    return ret;
}

static char *alloc_index_name(int index, char *name)
{
    char *ret = shl_mem_alloc(strlen(name) + 10);
    sprintf(ret, "%s_%d_", name, index);
    return ret;
}

static char *concat_name(char *name, char *append)
{
    char *ret = shl_mem_alloc(strlen(name) + strlen(append) + 10);
    sprintf(ret, "%s%s", name, append);
    return ret;
}

static struct csinn_tensor *alloc_weight_tensor(struct csinn_tensor *tensor,
                                                struct csinn_session *sess, char *name)
{
    struct csinn_tensor *ret = csinn_alloc_tensor(sess);
    ret->name = name;
    ret->dim_count = tensor->dim_count;
    for (int i = 0; i < ret->dim_count; i++) {
        ret->dim[i] = tensor->dim[i];
    }

    ret->dtype = tensor->dtype;
    ret->mtype = tensor->mtype;
    ret->data = tensor->data;
    ret->is_const = 1;
    return ret;
}

static struct csinn_tensor *linear(struct csinn_session *sess, struct csinn_tensor *x,
                                   struct csinn_tensor *y, char *name)
{
    struct csinn_tensor *linear_output = csinn_alloc_tensor(sess);
    linear_output->name = concat_name(name, "output");

    y->is_const = 1;
    struct csinn_matmul_params *linear_params =
        csinn_alloc_params(sizeof(struct csinn_matmul_params), sess);
    linear_params->trans_b = true;
    csinn_matmul_init(x, y, linear_output, linear_params);
    csinn_matmul(x, y, linear_output, linear_params);
    return linear_output;
}

static struct csinn_tensor *matmul(struct csinn_session *sess, struct csinn_tensor *x,
                                   struct csinn_tensor *y, char *name)
{
    struct csinn_tensor *matmul_output = csinn_alloc_tensor(sess);
    matmul_output->name = concat_name(name, "output");
    matmul_output->dtype = CSINN_DTYPE_FLOAT32;

    struct csinn_matmul_params *matmul_params =
        csinn_alloc_params(sizeof(struct csinn_matmul_params), sess);
    csinn_matmul_init(x, y, matmul_output, matmul_params);
    csinn_matmul(x, y, matmul_output, matmul_params);
    return matmul_output;
}

static struct csinn_tensor *silu(struct csinn_session *sess, struct csinn_tensor *x, char *name)
{
    struct csinn_tensor *silu_output = csinn_alloc_tensor(sess);
    silu_output->name = concat_name(name, "output");
    silu_output->dtype = CSINN_DTYPE_FLOAT32;

    struct csinn_sigmoid_params *silu_params =
        csinn_alloc_params(sizeof(struct csinn_sigmoid_params), sess);
    csinn_silu_init(x, silu_output, silu_params);
    csinn_silu(x, silu_output, silu_params);
    return silu_output;
}

static struct csinn_tensor *norm(struct csinn_session *sess, struct csinn_tensor *x,
                                 struct csinn_tensor *weight, char *name)
{
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    output->name = concat_name(name, "output");
    output->dtype = CSINN_DTYPE_FLOAT32;
    /*
     * output = x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
     */
    struct csinn_rms_norm_params *rms_params =
        csinn_alloc_params(sizeof(struct csinn_rms_norm_params), sess);

    // FIXME: from params.json's norm_eps
    rms_params->epsilon = 1e-05;
    // last dim
    rms_params->axis = -1;
    weight->is_const = 1;
    csinn_rms_norm_init(x, weight, output, rms_params);
    csinn_rms_norm(x, weight, output, rms_params);
    return output;
}

static struct csinn_tensor *view(struct csinn_session *sess, struct csinn_tensor *in, char *name)
{
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    output->name = concat_name(name, "output");
    output->dtype = CSINN_DTYPE_FLOAT32;

    struct csinn_reshape_params *params =
        csinn_alloc_params(sizeof(struct csinn_reshape_params), sess);

    csinn_reshape_init(in, output, params);
    csinn_reshape(in, output, params);
    return output;
}

static struct csinn_tensor *attention(struct shl_transformer_block *block, struct csinn_tensor *x,
                                      struct shl_llm_layer *llayer, char *name)
{
    struct csinn_session *sess = block->session;

    int bsz = x->dim[0];
    int seqlen = x->dim[1];
    int n_heads = 32;
    int head_dim = 128;

    // xk = linear(x)
    struct csinn_tensor *xk_weight = alloc_weight_tensor(llayer->wk, sess, concat_name(name, "wk"));
    struct csinn_tensor *xk = linear(sess, x, xk_weight, concat_name(name, "xk_linear"));

    // xq = linear(x)
    struct csinn_tensor *xq_weight = alloc_weight_tensor(llayer->wq, sess, concat_name(name, "wq"));
    struct csinn_tensor *xq = linear(sess, x, xq_weight, concat_name(name, "xq_linear"));

    // xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    struct csinn_reshape_params *xk_reshape_params =
        csinn_alloc_params(sizeof(struct csinn_reshape_params), sess);
    xk_reshape_params->shape_num = 4;
    xk_reshape_params->shape = shl_mem_alloc(4 * sizeof(int32_t));
    xk_reshape_params->shape[0] = bsz;
    xk_reshape_params->shape[1] = seqlen;
    xk_reshape_params->shape[2] = n_heads;
    xk_reshape_params->shape[3] = head_dim;

    struct csinn_tensor *xk_reshape_output = csinn_alloc_tensor(sess);
    xk_reshape_output->name = alloc_name("xk_reshape_output");
    csinn_reshape_init(xk, xk_reshape_output, xk_reshape_params);
    csinn_reshape(xk, xk_reshape_output, xk_reshape_params);

    // xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    struct csinn_reshape_params *xq_reshape_params =
        csinn_alloc_params(sizeof(struct csinn_reshape_params), sess);
    xq_reshape_params->shape_num = 4;
    xq_reshape_params->shape = shl_mem_alloc(4 * sizeof(int32_t));
    xq_reshape_params->shape[0] = bsz;
    xq_reshape_params->shape[1] = seqlen;
    xq_reshape_params->shape[2] = n_heads;
    xq_reshape_params->shape[3] = head_dim;

    struct csinn_tensor *xq_reshape_output = csinn_alloc_tensor(sess);
    xq_reshape_output->name = alloc_name("xq_reshape_output");
    csinn_reshape_init(xq, xq_reshape_output, xq_reshape_params);
    csinn_reshape(xq, xq_reshape_output, xq_reshape_params);

    struct csinn_tensor *xq_rope = csinn_alloc_tensor(sess);
    xq_rope->name = concat_name(name, "xq_rope");
    struct csinn_rope_params *rope_params =
        csinn_alloc_params(sizeof(struct csinn_rope_params), sess);
    rope_params->freq_base = 10000;
    rope_params->freq_scale = 1;
    rope_params->xpos_base = 0;
    rope_params->xpos_down = 0;
    rope_params->n_dims = 128;

    csinn_rope_init(xq_reshape_output, xq_rope, rope_params);
    csinn_rope(xq_reshape_output, xq_rope, rope_params);

    struct csinn_tensor *xk_rope = csinn_alloc_tensor(sess);
    xk_rope->name = concat_name(name, "xk_rope");
    csinn_rope_init(xk_reshape_output, xk_rope, rope_params);
    csinn_rope(xk_reshape_output, xk_rope, rope_params);

    // xv = linear(x)
    struct csinn_tensor *xv_weight = alloc_weight_tensor(llayer->wv, sess, concat_name(name, "wv"));
    struct csinn_tensor *xv = linear(sess, x, xv_weight, concat_name(name, "xv_linear"));

    // xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    struct csinn_reshape_params *xv_reshape_params =
        csinn_alloc_params(sizeof(struct csinn_reshape_params), sess);
    xv_reshape_params->shape_num = 4;
    xv_reshape_params->shape = shl_mem_alloc(4 * sizeof(int32_t));
    xv_reshape_params->shape[0] = bsz;
    xv_reshape_params->shape[1] = seqlen;
    xv_reshape_params->shape[2] = n_heads;
    xv_reshape_params->shape[3] = head_dim;

    struct csinn_tensor *xv_reshape_output = csinn_alloc_tensor(sess);
    xv_reshape_output->name = alloc_name("xv_reshape_output");
    csinn_reshape_init(xv, xv_reshape_output, xv_reshape_params);
    csinn_reshape(xv, xv_reshape_output, xv_reshape_params);

    // cache_k[:bsz, start_pos : start_pos + seqlen] = xk
    struct csinn_tensor *cache_k = csinn_alloc_tensor(sess);
    cache_k->name = alloc_name("cache_k");
    cache_k->dtype = CSINN_DTYPE_FLOAT32;
    cache_k->dim_count = 4;
    cache_k->dim[0] = 1;
    cache_k->dim[1] = 2048;  // max_seq_len
    cache_k->dim[2] = n_heads;
    cache_k->dim[3] = head_dim;

    block->cache_k = cache_k;
    block->cache_k_buffer = shl_mem_alloc(csinn_tensor_byte_size(cache_k));

    struct csinn_llm_pos_params *xk_cache_params =
        csinn_alloc_params(sizeof(struct csinn_llm_pos_params), sess);
    xk_cache_params->bsz = bsz;
    xk_cache_params->seqlen = seqlen;
    xk_cache_params->mode = CSINN_LLM_POS_CACHE_COPY_IN;
    xk_cache_params->cache_buffer = block->cache_k_buffer;
    csinn_llm_pos_init(xk_rope, cache_k, xk_cache_params);
    csinn_llm_pos(xk_rope, cache_k, xk_cache_params);

    // cache_v[:bsz, start_pos : start_pos + seqlen] = xv
    struct csinn_tensor *cache_v = csinn_alloc_tensor(sess);
    cache_v->name = alloc_name("cache_v");
    cache_v->dtype = CSINN_DTYPE_FLOAT32;
    cache_v->dim_count = 4;
    cache_v->dim[0] = 1;
    cache_v->dim[1] = 2048;  // max_seq_len
    cache_v->dim[2] = n_heads;
    cache_v->dim[3] = head_dim;

    block->cache_v = cache_v;
    block->cache_v_buffer = shl_mem_alloc(csinn_tensor_byte_size(cache_v));

    struct csinn_llm_pos_params *xv_cache_params =
        csinn_alloc_params(sizeof(struct csinn_llm_pos_params), sess);
    xv_cache_params->bsz = bsz;
    xv_cache_params->seqlen = seqlen;
    xv_cache_params->mode = CSINN_LLM_POS_CACHE_COPY_IN;
    xv_cache_params->cache_buffer = block->cache_v_buffer;
    csinn_llm_pos_init(xv_reshape_output, cache_v, xv_cache_params);
    csinn_llm_pos(xv_reshape_output, cache_v, xv_cache_params);

    // keys = self.cache_k[:bsz, : start_pos + seqlen]
    struct csinn_tensor *keys = csinn_alloc_tensor(sess);
    keys->name = concat_name(name, "keys");

    struct csinn_llm_pos_params *keys_params =
        csinn_alloc_params(sizeof(struct csinn_llm_pos_params), sess);
    keys_params->bsz = bsz;
    keys_params->seqlen = seqlen;
    keys_params->mode = CSINN_LLM_POS_CACHE_COPY_OUT;
    keys_params->cache_buffer = block->cache_k_buffer;
    csinn_llm_pos_init(cache_k, keys, keys_params);
    csinn_llm_pos(cache_k, keys, keys_params);

    // keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
    // values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

    // xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    struct csinn_tensor *xq_transpose = csinn_alloc_tensor(sess);
    xq_transpose->name = alloc_name("xq_transpose");

    struct csinn_transpose_params *xq_transpose_params =
        csinn_alloc_params(sizeof(struct csinn_transpose_params), sess);
    xq_transpose_params->permute_num = 4;
    xq_transpose_params->permute = shl_mem_alloc(4 * sizeof(int32_t));
    xq_transpose_params->permute[0] = 0;
    xq_transpose_params->permute[1] = 2;
    xq_transpose_params->permute[2] = 1;
    xq_transpose_params->permute[3] = 3;
    csinn_transpose_init(xq_rope, xq_transpose, xq_transpose_params);
    csinn_transpose(xq_rope, xq_transpose, xq_transpose_params);

    // keys = keys.transpose(1, 2)
    struct csinn_tensor *keys_transpose = csinn_alloc_tensor(sess);
    keys_transpose->name = alloc_name("keys_transpose");

    struct csinn_transpose_params *keys_transpose_params =
        csinn_alloc_params(sizeof(struct csinn_transpose_params), sess);
    keys_transpose_params->permute_num = 4;
    keys_transpose_params->permute = shl_mem_alloc(4 * sizeof(int32_t));
    keys_transpose_params->permute[0] = 0;
    keys_transpose_params->permute[1] = 2;
    keys_transpose_params->permute[2] = 1;
    keys_transpose_params->permute[3] = 3;
    csinn_transpose_init(keys, keys_transpose, keys_transpose_params);
    csinn_transpose(keys, keys_transpose, keys_transpose_params);

    // scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    struct csinn_tensor *scores = csinn_alloc_tensor(sess);
    scores->name = concat_name(name, "scores");

    struct csinn_matmul_params *scores_matmul_params =
        csinn_alloc_params(sizeof(struct csinn_matmul_params), sess);
    scores_matmul_params->trans_b = true;
    csinn_matmul_init(xq_transpose, keys_transpose, scores, scores_matmul_params);
    csinn_matmul(xq_transpose, keys_transpose, scores, scores_matmul_params);

    struct csinn_tensor *scores_mul = csinn_alloc_tensor(sess);
    scores_mul->name = concat_name(name, "scores_mul");

    struct csinn_tensor *scale = csinn_alloc_tensor(sess);
    scale->is_const = 1;
    float *scale_value = shl_mem_alloc(4);
    scale_value[0] = 0.088388347648318;
    scale->data = scale_value;
    scale->dim_count = 1;
    scale->dim[0] = 1;
    struct csinn_diso_params *scores_mul_params =
        csinn_alloc_params(sizeof(struct csinn_diso_params), sess);
    csinn_mul_init(scores, scale, scores_mul, scores_mul_params);
    csinn_mul(scores, scale, scores_mul, scores_mul_params);

    // if mask is not None:
    //     scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
    struct csinn_tensor *scores_mask = csinn_alloc_tensor(sess);
    scores_mask->name = concat_name(name, "scores_mask");

    struct csinn_llm_pos_params *scores_mask_params =
        csinn_alloc_params(sizeof(struct csinn_llm_pos_params), sess);
    scores_mask_params->bsz = bsz;
    scores_mask_params->seqlen = seqlen;
    scores_mask_params->mode = CSINN_LLM_POS_MASK;
    csinn_llm_pos_init(scores_mul, scores_mask, scores_mask_params);
    csinn_llm_pos(scores_mul, scores_mask, scores_mask_params);

    // scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    struct csinn_tensor *scores_softmax = csinn_alloc_tensor(sess);
    scores_softmax->name = concat_name(name, "scores_softmax");

    struct csinn_softmax_params *scores_softmax_params =
        csinn_alloc_params(sizeof(struct csinn_softmax_params), sess);
    scores_softmax_params->axis = 3;
    csinn_softmax_init(scores_mask, scores_softmax, scores_softmax_params);
    csinn_softmax(scores_mask, scores_softmax, scores_softmax_params);

    // values = self.cache_v[:bsz, : start_pos + seqlen]
    struct csinn_tensor *values = csinn_alloc_tensor(sess);
    values->name = concat_name(name, "values");

    struct csinn_llm_pos_params *values_params =
        csinn_alloc_params(sizeof(struct csinn_llm_pos_params), sess);
    values_params->bsz = bsz;
    values_params->seqlen = seqlen;
    values_params->mode = CSINN_LLM_POS_CACHE_COPY_OUT;
    values_params->cache_buffer = block->cache_v_buffer;
    csinn_llm_pos_init(cache_v, values, values_params);
    csinn_llm_pos(cache_v, values, values_params);

    // values = values.transpose(1, 2)
    struct csinn_tensor *values_transpose = csinn_alloc_tensor(sess);
    values_transpose->name = alloc_name("values_transpose");

    struct csinn_transpose_params *values_transpose_params =
        csinn_alloc_params(sizeof(struct csinn_transpose_params), sess);
    values_transpose_params->permute_num = 4;
    values_transpose_params->permute = shl_mem_alloc(4 * sizeof(int32_t));
    values_transpose_params->permute[0] = 0;
    values_transpose_params->permute[1] = 2;
    values_transpose_params->permute[2] = 1;
    values_transpose_params->permute[3] = 3;
    csinn_transpose_init(values, values_transpose, values_transpose_params);
    csinn_transpose(values, values_transpose, values_transpose_params);

    // output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
    struct csinn_tensor *output_matmul = csinn_alloc_tensor(sess);
    output_matmul->name = concat_name(name, "kqv_matmul");

    struct csinn_matmul_params *output_matmul_params =
        csinn_alloc_params(sizeof(struct csinn_matmul_params), sess);
    csinn_matmul_init(scores_softmax, values_transpose, output_matmul, output_matmul_params);
    csinn_matmul(scores_softmax, values_transpose, output_matmul, output_matmul_params);

    // output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    struct csinn_tensor *output_transpose = csinn_alloc_tensor(sess);
    output_transpose->name = concat_name(name, "output_transpose");

    struct csinn_transpose_params *output_transpose_params =
        csinn_alloc_params(sizeof(struct csinn_transpose_params), sess);
    output_transpose_params->permute_num = 4;
    output_transpose_params->permute = shl_mem_alloc(4 * sizeof(int32_t));
    output_transpose_params->permute[0] = 0;
    output_transpose_params->permute[1] = 2;
    output_transpose_params->permute[2] = 1;
    output_transpose_params->permute[3] = 3;
    csinn_transpose_init(output_matmul, output_transpose, output_transpose_params);
    csinn_transpose(output_matmul, output_transpose, output_transpose_params);

    struct csinn_reshape_params *output_transpose_reshape_params =
        csinn_alloc_params(sizeof(struct csinn_reshape_params), sess);
    output_transpose_reshape_params->shape_num = 3;
    output_transpose_reshape_params->shape = shl_mem_alloc(3 * sizeof(int32_t));
    output_transpose_reshape_params->shape[0] = bsz;
    output_transpose_reshape_params->shape[1] = seqlen;
    output_transpose_reshape_params->shape[2] = n_heads * head_dim;

    struct csinn_tensor *output_transpose_reshape_output = csinn_alloc_tensor(sess);
    output_transpose_reshape_output->name = alloc_name("output_transpose_reshape_output");
    csinn_reshape_init(output_transpose, output_transpose_reshape_output,
                       output_transpose_reshape_params);
    csinn_reshape(output_transpose, output_transpose_reshape_output,
                  output_transpose_reshape_params);

    // return self.wo(output)
    struct csinn_tensor *xo_weight = alloc_weight_tensor(llayer->wo, sess, concat_name(name, "wo"));
    struct csinn_tensor *output =
        linear(sess, output_transpose_reshape_output, xo_weight, concat_name(name, "wo_linear"));
    return output;
}

static struct csinn_tensor *feed_forward(struct csinn_session *sess, struct csinn_tensor *x,
                                         struct csinn_tensor *w1, struct csinn_tensor *w2,
                                         struct csinn_tensor *w3, char *name)
{
    // x3 = linear(x, w3)
    struct csinn_tensor *x3 = linear(sess, x, w3, concat_name(name, "x3_linear"));

    // x1 = linear(x, w1)
    struct csinn_tensor *x1 = linear(sess, x, w1, concat_name(name, "x1_linear"));
    // x1 = silu(x1)
    struct csinn_tensor *silu_output = silu(sess, x1, concat_name(name, "x1_silu"));

    // x2 = matmul(x1, x3)
    struct csinn_tensor *x2 = csinn_alloc_tensor(sess);
    x2->name = concat_name(name, "ff_0_x2_mul_output");
    struct csinn_diso_params *x2_mul_params =
        csinn_alloc_params(sizeof(struct csinn_diso_params), sess);
    csinn_mul_init(silu_output, x3, x2, x2_mul_params);
    csinn_mul(silu_output, x3, x2, x2_mul_params);
    // struct csinn_tensor *x2 = matmul(sess, silu_output, x3, concat_name(name, "x2_matmul"));

    // x2 = linear(x2, w2)
    struct csinn_tensor *x2_linear_output = linear(sess, x2, w2, concat_name(name, "x2_linear"));
    return x2_linear_output;
}

static struct shl_transformer_block *layer(struct shl_llm_ctx *ctx, struct csinn_tensor *input,
                                           int layer_id)
{
    struct shl_transformer_block *ret = shl_mem_alloc(sizeof(struct shl_transformer_block));
    ret->layer_id = layer_id;

    struct shl_llm_layer *llayer = &(ctx->shl_model->layers[layer_id]);

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->base_quant_type = CSINN_QUANT_FLOAT32;
    sess->model.save_mode = CSINN_RUN_ONLY;
    sess->base_layout = CSINN_LAYOUT_NCHW;
    sess->base_api = CSINN_REF;
    sess->base_dtype = CSINN_DTYPE_FLOAT32;
    sess->dynamic_shape = CSINN_FALSE;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    csinn_session_init(sess);
    ret->session = sess;
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    struct csinn_tensor *x = csinn_alloc_tensor(sess);
    csinn_tensor_copy(x, input);
    x->sess = sess;
    csinn_set_tensor_entry(x, sess);
    csinn_set_input(0, x, sess);

    // h = x + attention(norm(x), start_pos, freqs_cis, mask)
    struct csinn_tensor *attention_norm_weight = alloc_weight_tensor(
        llayer->attn_norm, sess, alloc_index_name(layer_id, "attention_norm_weight"));

    char *norm_name = alloc_index_name(layer_id, "attention_norm");
    struct csinn_tensor *norm_output = norm(sess, x, attention_norm_weight, norm_name);
    char *attention_name = alloc_index_name(layer_id, "attention");
    struct csinn_tensor *attention_output = attention(ret, norm_output, llayer, attention_name);

    struct csinn_tensor *h_attention = csinn_alloc_tensor(sess);
    h_attention->name = alloc_index_name(layer_id, "h_attention");
    struct csinn_diso_params *x_add_params =
        csinn_alloc_params(sizeof(struct csinn_diso_params), sess);
    csinn_add_init(x, attention_output, h_attention, x_add_params);
    csinn_add(x, attention_output, h_attention, x_add_params);

    // out = h + feed_forward(norm(h))
    struct csinn_tensor *ff_norm_weight =
        alloc_weight_tensor(llayer->ffn_norm, sess, alloc_index_name(layer_id, "ffn_norm_weight"));
    char *ffn_norm_name = alloc_index_name(layer_id, "ffn_norm");
    struct csinn_tensor *ff_norm = norm(sess, h_attention, ff_norm_weight, ffn_norm_name);
    struct csinn_tensor *ff_w1 =
        alloc_weight_tensor(llayer->w1, sess, alloc_index_name(layer_id, "ffn_w1"));
    struct csinn_tensor *ff_w2 =
        alloc_weight_tensor(llayer->w2, sess, alloc_index_name(layer_id, "ffn_w2"));
    struct csinn_tensor *ff_w3 =
        alloc_weight_tensor(llayer->w3, sess, alloc_index_name(layer_id, "ffn_w3"));
    char *ffn_name = alloc_index_name(layer_id, "ff");
    struct csinn_tensor *ff_output = feed_forward(sess, ff_norm, ff_w1, ff_w2, ff_w3, ffn_name);

    struct csinn_tensor *h_ff = csinn_alloc_tensor(sess);
    h_ff->name = alloc_index_name(layer_id, "h_ff");

    struct csinn_diso_params *h_add_params =
        csinn_alloc_params(sizeof(struct csinn_diso_params), sess);
    csinn_add_init(h_attention, ff_output, h_ff, h_add_params);
    csinn_add(h_attention, ff_output, h_ff, h_add_params);

    csinn_set_output(0, h_ff, sess);

    csinn_session_setup(sess);

    return ret;
}

static struct csinn_session *tok_embedding(struct llama_config *config)
{
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->base_quant_type = CSINN_QUANT_FLOAT16;
    sess->model.save_mode = CSINN_RUN_ONLY;
    sess->base_layout = CSINN_LAYOUT_NCHW;
    sess->base_api = CSINN_REF;
    sess->base_dtype = CSINN_DTYPE_FLOAT16;
    sess->dynamic_shape = CSINN_TRUE;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    struct csinn_tensor *embd = csinn_alloc_tensor(sess);
    embd->dtype = CSINN_DTYPE_INT32;

    csinn_set_tensor_entry(embd, sess);
    csinn_set_input(0, embd, sess);
    struct csinn_tensor *embd_output = csinn_alloc_tensor(sess);
    embd_output->name = "embd_output";
    embd_output->dtype = CSINN_DTYPE_FLOAT32;

    embd_output->dim_count = 2;
    embd_output->dim[0] = 0;
    embd_output->dim[1] = config->shl_model->tok_embeddings->dim[1];

    struct csinn_tensor *embd_weight = csinn_alloc_tensor(sess);
    embd_weight->name = "embd_weight";
    embd_weight->is_const = 1;

    embd_weight->dtype = config->shl_model->tok_embeddings->dtype;
    embd_weight->mtype = config->shl_model->tok_embeddings->mtype;
    embd_weight->dim_count = config->shl_model->tok_embeddings->dim_count;
    embd_weight->dim[0] = config->shl_model->tok_embeddings->dim[0];
    embd_weight->dim[1] = config->shl_model->tok_embeddings->dim[1];

    embd_weight->data = config->shl_model->tok_embeddings->data;

    struct csinn_diso_params *embd_params =
        csinn_alloc_params(sizeof(struct csinn_diso_params), sess);
    csinn_embedding_init(embd, embd_weight, embd_output, embd_params);
    csinn_embedding(embd, embd_weight, embd_output, embd_params);

    csinn_set_output(0, embd_output, sess);

    csinn_session_setup(sess);
    return sess;
}

static struct csinn_session *llama2_output(struct shl_llm_ctx *ctx)
{
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->base_quant_type = CSINN_QUANT_FLOAT32;
    sess->model.save_mode = CSINN_RUN_ONLY;
    sess->base_layout = CSINN_LAYOUT_NCHW;
    sess->base_api = CSINN_REF;
    sess->base_dtype = CSINN_DTYPE_FLOAT32;
    sess->dynamic_shape = CSINN_FALSE;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    struct csinn_tensor *input = ctx->transformer_block[ctx->layers_num - 1]->session->output[0];

    struct csinn_tensor *h_in = csinn_alloc_tensor(sess);
    csinn_tensor_copy(h_in, input);
    h_in->sess = sess;
    csinn_set_tensor_entry(h_in, sess);
    csinn_set_input(0, h_in, sess);

    // h = norm(h)
    struct csinn_tensor *h_weight =
        alloc_weight_tensor(ctx->shl_model->output_norm, sess, alloc_name("output_norm_weight"));
    struct csinn_tensor *h_norm_output = norm(sess, h_in, h_weight, alloc_name("output_norm"));

    // output = linear(h)
    struct csinn_tensor *linear_weight =
        alloc_weight_tensor(ctx->shl_model->output, sess, alloc_name("output_weight"));
    struct csinn_tensor *linear_output =
        linear(sess, h_norm_output, linear_weight, "linear_output");

    csinn_set_output(0, linear_output, sess);

    csinn_session_setup(sess);
    return sess;
}

struct shl_llm_ctx *llama2_build(struct llama_config *config)
{
    struct shl_llm_ctx *ctx = shl_mem_alloc(sizeof(struct shl_llm_ctx));
    ctx->shl_model = config->shl_model;

    // h = tok_embedding(tokens)
    ctx->embeding_session = tok_embedding(config);

    // TransformerBlocks: h = layer(h, start_pos, freqes_cis, mask)
    ctx->layers_num = config->n_layers;
    ctx->transformer_block = shl_mem_alloc(sizeof(struct shl_transformer_block) * ctx->layers_num);
    ctx->transformer_block[0] = layer(ctx, ctx->embeding_session->output[0], 0);
    for (int i = 1; i < ctx->layers_num; i++) {
        ctx->transformer_block[i] =
            layer(ctx, ctx->transformer_block[i - 1]->session->output[0], i);
    }

    /*
     * h = norm(h)
     * output = linear(h)
     */
    ctx->output_session = llama2_output(ctx);

    return ctx;
}
