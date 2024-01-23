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
#include "shl_utils.h"

void shl_target_init_ref();
void shl_target_init_gref();
void shl_target_init_ovx();
void shl_target_init_c906();
void shl_target_init_pnna();
void shl_target_init_c908();
void shl_target_init_rvv();
void shl_target_init_rvm();
void shl_target_init_e907();
void shl_target_init_c920();
void shl_target_init_c920v2();

static int __shl_has_init;

void shl_init()
{
#ifdef SHL_BUILD_REF
    shl_target_init_ref();
#endif
#ifdef SHL_BUILD_GREF
    shl_target_init_gref();
#endif
#ifdef SHL_BUILD_C906
    shl_target_init_c906();
#endif
#ifdef SHL_BUILD_OPENVX
    shl_target_init_ovx();
#endif
#ifdef SHL_BUILD_PNNA
    shl_target_init_pnna();
#endif
#ifdef SHL_BUILD_C908
    shl_target_init_c908();
#endif
#ifdef SHL_BUILD_RVV
    shl_target_init_rvv();
#endif
#ifdef SHL_BUILD_RVM
    shl_target_init_rvm();
#endif
#ifdef SHL_BUILD_E907
    shl_target_init_e907();
#endif
#ifdef SHL_BUILD_C920
    shl_target_init_c920();
#endif
#ifdef SHL_BUILD_C920V2
    shl_target_init_c920v2();
#endif
}

/**
 * @addtogroup SESSION
 * @{
 */
struct csinn_session *csinn_alloc_session()
{
    if (__shl_has_init == 0) {
        shl_init();
        __shl_has_init = 1;
    }
    return shl_mem_alloc(sizeof(struct csinn_session));
}
/**
 * @}
 */

/**
 * @addtogroup SESSION
 * @{
 */
void csinn_free_session(struct csinn_session *sess) { shl_mem_free(sess); }
/**
 * @}
 */

static void *shl_cb_func_table[CSINN_API_SIZE];
void shl_register_op_callback(int api, void *cb) { shl_cb_func_table[api] = cb; }

int shl_op_callback_map(struct csinn_params_base *base, int op, int dtype)
{
    void *(*op_map)();
    /* FIXME: unuse CSINN_REF */
    if (base->sess &&
        ((base->sess->base_run_mode == CSINN_RM_CPU_GRAPH && base->sess->base_api == CSINN_REF) ||
         base->sess->base_run_mode == CSINN_RM_CPU_BASE_HYBRID)) {
        /* Heterogeneous use GREF */
        op_map = shl_cb_func_table[CSINN_GREF];
    } else {
        op_map = shl_cb_func_table[base->api];
    }

    if (op_map == NULL) {
        return CSINN_FALSE;
    }

    struct csinn_callback *cb = op_map(op, dtype);
    if (cb == NULL) {
        shl_debug_info("%s: Cannot find OP map\n", __func__);
    }
    memcpy(base->cb, cb, sizeof(struct csinn_callback));

    return CSINN_TRUE;
}

static void *shl_runtime_callback_table[CSINN_API_SIZE];

void shl_register_runtime_callback(int api, void *cb) { shl_runtime_callback_table[api] = cb; }

void *shl_get_runtime_callback(struct csinn_session *sess, int op)
{
    void *(*runtime_map)();
    /* FIXME: unuse CSINN_REF */
    if ((sess->base_run_mode == CSINN_RM_CPU_GRAPH && sess->base_api == CSINN_REF) ||
        sess->base_run_mode == CSINN_RM_CPU_BASE_HYBRID) {
        /* Heterogeneous use GREF */
        runtime_map = shl_runtime_callback_table[CSINN_GREF];
    } else {
        runtime_map = shl_runtime_callback_table[sess->base_api];
    }
    if (runtime_map == NULL) {
        return NULL;
    } else {
        return runtime_map(op);
    }
}

/**
 * @addtogroup SESSION
 * @{
 */
void csinn_session_init(struct csinn_session *sess)
{
    shl_debug_set_level(sess->debug_level);
    if (sess->profiler_level >= CSINN_PROFILER_LEVEL_TRACE) {
        struct shl_trace *trace = (struct shl_trace *)shl_mem_alloc(sizeof(struct shl_trace));
        trace->enable_trace = true;
        sess->trace = trace;

        SHL_TRACE_CALL(shl_trace_begin(trace, NULL));

        // add some meta-data
        SHL_TRACE_CALL(shl_trace_other_data(
            trace, shl_trace_create_dict(
                       4, "source", SHL_TRACE_STRING("csinn"), "base_api",
                       SHL_TRACE_STRING(shl_find_api_name(sess->base_api)), "base_run_mode",
                       SHL_TRACE_STRING(shl_find_rmod_name(sess->base_run_mode)), "base_quant_type",
                       SHL_TRACE_STRING(shl_find_quant_name(sess->base_quant_type)))));
    }

    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    void *(*func)() = shl_get_runtime_callback(sess, CSINN_SESSION_INIT);
    if (func != NULL) {
        func(sess);
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));
}
/**
 * @}
 */

/**
 * @addtogroup SESSION
 * @{
 */
void csinn_session_deinit(struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    void *(*func)();
    func = shl_get_runtime_callback(sess, CSINN_SESSION_DEINIT);
    if (func != NULL) {
        func(sess);
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    if (sess->profiler_level >= CSINN_PROFILER_LEVEL_TRACE) {
        SHL_TRACE_CALL(shl_trace_end(sess->trace));
        shl_mem_free(sess->trace);
    }
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
void csinn_set_output_number(int number, struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    sess->output_num = number;
    sess->output = shl_mem_alloc(sess->output_num * sizeof(struct csinn_tensor *));
    void (*func)();
    func = shl_get_runtime_callback(sess, CSINN_SET_OUTPUT_NUMBER);
    if (func != NULL) {
        func(number, sess);
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
void csinn_set_input_number(int number, struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    sess->input_num = number;
    sess->input = shl_mem_alloc(sess->input_num * sizeof(struct csinn_tensor *));
    void (*func)();
    func = shl_get_runtime_callback(sess, CSINN_SET_INPUT_NUMBER);
    if (func != NULL) {
        func(number, sess);
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
int csinn_get_output_number(struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = 0;
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_GET_OUTPUT_NUMBER);
    if (func != NULL) {
        ret = func(sess);
    } else {
        ret = sess->output_num;
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
int csinn_get_input_number(struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = 0;
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_GET_INPUT_NUMBER);
    if (func != NULL) {
        ret = func(sess);
    } else {
        ret = sess->input_num;
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
int csinn_set_output(int index, struct csinn_tensor *output, struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = CSINN_TRUE;
    sess->output[index] = output;
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_SET_OUTPUT);
    if (func != NULL) {
        ret = func(index, output, sess);
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
int csinn_set_input(int index, struct csinn_tensor *input, struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = CSINN_TRUE;
    sess->input[index] = input;
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_SET_INPUT);
    if (func != NULL) {
        ret = func(index, input, sess);
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
int csinn_get_output(int index, struct csinn_tensor *output, struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = CSINN_TRUE;
    csinn_tensor_copy(output, sess->output[index]);
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_GET_OUTPUT);
    if (func != NULL) {
        ret = func(index, output, sess);
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
int csinn_get_input(int index, struct csinn_tensor *input, struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = CSINN_TRUE;
    csinn_tensor_copy(input, sess->input[index]);
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_GET_INPUT);
    if (func != NULL) {
        ret = func(index, input, sess);
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
int csinn_update_input(int index, struct csinn_tensor *input, struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = CSINN_TRUE;
    sess->input[index]->data = input->data;
    if (sess->dynamic_shape) {
        memcpy(sess->input[index]->dim, input->dim, sizeof(int32_t) * MAX_DIM);
        sess->input[index]->dim_count = input->dim_count;
    }
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_UPDATE_INPUT);
    if (func != NULL) {
        if (sess->profiler_level == CSINN_PROFILER_LEVEL_TIMER) {
            uint64_t start = shl_get_timespec();
            ret = func(index, input, sess);
            uint64_t end = shl_get_timespec();
            shl_print_time_interval(start, end, __func__);
        } else {
            ret = func(index, input, sess);
        }
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
int csinn_update_output(int index, struct csinn_tensor *output, struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = CSINN_TRUE;
    sess->output[index]->data = output->data;
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_UPDATE_OUTPUT);
    if (func != NULL) {
        ret = func(index, output, sess);
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup SESSION
 * @{
 */
int csinn_session_setup(struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = CSINN_FALSE;
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_SESSION_SETUP);
    if (func != NULL) {
        if (sess->profiler_level == CSINN_PROFILER_LEVEL_TIMER) {
            uint64_t start = shl_get_timespec();
            ret = func(sess);
            uint64_t end = shl_get_timespec();
            shl_print_time_interval(start, end, __func__);
        } else {
            ret = func(sess);
        }
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup SESSION
 * @{
 */
int csinn_session_run(struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = CSINN_FALSE;
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_SESSION_RUN);
    if (func != NULL) {
        if (sess->profiler_level == CSINN_PROFILER_LEVEL_TIMER) {
            uint64_t start = shl_get_timespec();
            ret = func(sess);
            uint64_t end = shl_get_timespec();
            shl_print_time_interval(start, end, __func__);
        } else {
            ret = func(sess);
        }
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup IO
 * @{
 */
int csinn_set_tensor_entry(struct csinn_tensor *t, struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = CSINN_FALSE;
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_TENSOR_ENTRY);
    if (func != NULL) {
        ret = func(t, sess);
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */

/**
 * @addtogroup SESSION
 * @{
 */
int csinn_load_binary_model(struct csinn_session *sess)
{
    SHL_TRACE_CALL(shl_trace_duration_begin(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    int ret = CSINN_FALSE;
    int (*func)();
    func = shl_get_runtime_callback(sess, CSINN_LOAD_BG);
    if (func != NULL) {
        if (sess->profiler_level == CSINN_PROFILER_LEVEL_TIMER) {
            uint64_t start = shl_get_timespec();
            ret = func(sess);
            uint64_t end = shl_get_timespec();
            shl_print_time_interval(start, end, __func__);
        } else {
            ret = func(sess);
        }
    }

    SHL_TRACE_CALL(shl_trace_duration_end(sess->trace, __func__, SHL_TRACE_EVENT_RUNTIME, NULL));

    return ret;
}
/**
 * @}
 */
