#include "llm/shl_llm.h"

static void llm_session_dynamic_infer_shape(struct csinn_session *sess, struct shl_llm_input *embd)
{
    // shl_debug_set_level(-1);
    shl_debug_info("\n\n#########llm_session_dynamic_infer_shape#########\n\n");
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    for (int i = 0; i < graph->layer_index; i++) {
        struct shl_node *n = graph->layer[i];

        struct csinn_params_base *params = n->data;
        struct csinn_tensor **inputs;
        struct csinn_tensor **outputs;
        struct csinn_tensor *output_tensor;
        struct csinn_tensor *input_tensor;
        struct csinn_llm_pos_params *pos_params;
        struct csinn_rope_params *rope_params;
        struct csinn_reshape_params *reshape_params;
        switch (n->type) {
            case CSINN_OP_ABS:
            case CSINN_OP_ACOS:
            case CSINN_OP_CLIP:
            case CSINN_OP_DIV:
            case CSINN_OP_LAYER_NORM:
            case CSINN_OP_RELU:
            case CSINN_OP_RELU1:
            case CSINN_OP_RELU6:
            case CSINN_OP_SIGMOID:
            case CSINN_OP_SOFTMAX:
            case CSINN_OP_SQRT:
            case CSINN_OP_ERF:
            case CSINN_OP_TANH:
                shl_gref_siso_infer_shape(n->in[0]->data, n->out[0]->data, params);
                break;
            case CSINN_OP_ADD:
            case CSINN_OP_MUL:
            case CSINN_OP_SUB:
            case CSINN_OP_POWER:
                shl_gref_diso_infer_shape(n->in[0]->data, n->in[1]->data, n->out[0]->data, params);
                break;
            case CSINN_OP_CONCAT:
                inputs = shl_mem_alloc(sizeof(struct csinn_tensor *) *
                                       ((struct csinn_concat_params *)params)->inputs_count);
                for (int i = 0; i < ((struct csinn_concat_params *)params)->inputs_count; i++) {
                    inputs[i] = n->in[i]->data;
                }
                shl_gref_concat_infer_shape(inputs, n->out[0]->data,
                                            (struct csinn_concat_params *)params);
                shl_mem_free(inputs);
                break;
            case CSINN_OP_CONV1D:
            case CSINN_OP_DEPTHWISE_CONV1D:
                shl_gref_conv1d_infer_shape(n->in[0]->data, n->out[0]->data, n->in[1]->data,
                                            n->in[2]->data, (struct csinn_conv1d_params *)params);
                break;
            case CSINN_OP_CONV2D:
            case CSINN_OP_GROUP_CONV2D:
            case CSINN_OP_DEPTHWISE_CONV2D:
                shl_gref_conv2d_infer_shape(n->in[0]->data, n->out[0]->data, n->in[1]->data,
                                            n->in[2]->data, (struct csinn_conv2d_params *)params);
                break;
            case CSINN_OP_FULLYCONNECTED:
                shl_gref_fullyconnected_infer_shape(n->in[0]->data, n->out[0]->data, n->in[1]->data,
                                                    n->in[2]->data,
                                                    (struct csinn_fc_params *)params);
                break;
            case CSINN_OP_GATHER:
                shl_gref_gather_infer_shape(n->in[0]->data, n->in[1]->data, n->out[0]->data,
                                            (struct csinn_gather_params *)params);
                break;
            case CSINN_OP_MATMUL:
                shl_gref_matmul_infer_shape(n->in[0]->data, n->in[1]->data, n->out[0]->data,
                                            (struct csinn_matmul_params *)params);
                break;
            case CSINN_OP_RESHAPE:
                reshape_params = (struct csinn_reshape_params *)params;
                reshape_params->shape[0] = 1;
                reshape_params->shape[1] = embd->n_tokens;
                shl_gref_reshape_infer_shape(n->in[0]->data, n->out[0]->data, reshape_params);
                break;
            case CSINN_OP_SPLIT:
                outputs = shl_mem_alloc(sizeof(struct csinn_tensor *) *
                                        ((struct csinn_split_params *)params)->output_num);
                for (int i = 0; i < ((struct csinn_split_params *)params)->output_num; i++) {
                    outputs[i] = n->out[i]->data;
                }
                shl_gref_split_infer_shape(n->in[0]->data, outputs,
                                           (struct csinn_split_params *)params);
                shl_mem_free(outputs);
                break;
            case CSINN_OP_STRIDED_SLICE:
                shl_gref_strided_slice_infer_shape(n->in[0]->data, n->out[0]->data,
                                                   (struct csinn_strided_slice_params *)params);
                break;
            case CSINN_OP_TRANSPOSE:
                shl_gref_transpose_infer_shape(n->in[0]->data, n->out[0]->data,
                                               (struct csinn_transpose_params *)params);
                break;
            case CSINN_OP_WHERE_SOFTMAX:
                shl_gref_where_softmax_infer_shape(n->in[0]->data, n->in[1]->data, n->out[0]->data,
                                                   (struct csinn_where_softmax_params *)params);
                break;
            case CSINN_OP_GLOBAL_AVGPOOL2D:
            case CSINN_OP_GLOBAL_MAXPOOL2D:
                shl_gref_global_pooling2d_infer_shape(n->in[0]->data, n->out[0]->data,
                                                      (struct csinn_pool_params *)params);
                break;
            case CSINN_OP_MEAN:
                shl_gref_mean_infer_shape(n->in[0]->data, n->out[0]->data,
                                          (struct csinn_reduce_params *)params);
                break;
            case CSINN_OP_EMBEDDING:
                shl_gref_embedding_infer_shape(n->in[0]->data, n->in[1]->data, n->out[0]->data,
                                               (struct csinn_diso_params *)params);
                break;
            case CSINN_OP_LLM_POS:
                pos_params = (struct csinn_llm_pos_params *)params;
                pos_params->pos = embd->pos;
                pos_params->bsz = 1;
                pos_params->seqlen = embd->n_tokens;
                shl_gref_llm_pos_infer_shape(n->in[0]->data, n->out[0]->data, pos_params);
                break;
            case CSINN_OP_ROPE:
                rope_params = (struct csinn_rope_params *)params;
                rope_params->pos = embd->pos;
                shl_gref_rope_infer_shape(n->in[0]->data, n->out[0]->data, rope_params);
                break;
            case CSINN_OP_RMS_NORM:
                shl_gref_rms_norm_infer_shape(n->in[0]->data, n->in[1]->data, n->out[0]->data,
                                              (struct csinn_rms_norm_params *)params);
                break;
            case CSINN_OP_SILU:
                shl_gref_silu_infer_shape(n->in[0]->data, n->out[0]->data,
                                          (struct csinn_sigmoid_params *)params);
                break;
            default:
                shl_debug_error("[llm_session_dynamic_infer_shape]:unknown op %d\n", n->type);
                break;
        }
    }
}

static void update_input(struct csinn_session *curr, struct csinn_session *prev)
{
    curr->input[0]->data = prev->output[0]->data;
    curr->input[0]->dim_count = prev->output[0]->dim_count;
    for (int i = 0; i < prev->output[0]->dim_count; i++) {
        curr->input[0]->dim[i] = prev->output[0]->dim[i];
    }
}

int llm_run(struct shl_llm_ctx *ctx, struct shl_llm_input *embd)
{
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim_count = 1;
    input->dim[0] = embd->n_tokens;
    input->data = embd->token;
    input->dtype = CSINN_DTYPE_INT32;

    csinn_update_input(0, input, ctx->embeding_session);
    csinn_session_run(ctx->embeding_session);

    struct csinn_session *cur_sess = ctx->transformer_block[0]->session;
    update_input(cur_sess, ctx->embeding_session);

    llm_session_dynamic_infer_shape(cur_sess, embd);
    csinn_session_run(cur_sess);
    for (int i = 1; i < ctx->layers_num; i++) {
        cur_sess = ctx->transformer_block[i]->session;
        update_input(cur_sess, ctx->transformer_block[i - 1]->session);

        llm_session_dynamic_infer_shape(cur_sess, embd);
        csinn_session_run(cur_sess);
    }

    cur_sess = ctx->output_session;
    update_input(cur_sess, ctx->transformer_block[ctx->layers_num - 1]->session);
    llm_session_dynamic_infer_shape(cur_sess, embd);
    csinn_session_run(ctx->output_session);

    return CSINN_TRUE;
}
