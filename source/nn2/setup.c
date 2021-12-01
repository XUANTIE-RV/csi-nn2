/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.8.x */

#include "csi_nn.h"
#include "csi_utils.h"

struct csi_session *csi_alloc_session()
{
    return calloc(1, sizeof(struct csi_session));
}

void csi_free_session(struct csi_session *sess)
{
    free(sess);
}

void *csi_bc_map_ref(int op, int dtype);
void *csi_bc_map_gref(int op, int dtype);
void *csi_bc_map_ovx(int op, int dtype);
void *csi_bc_map_c906(int op, int dtype);
void *csi_bc_map_pnna(int op, int dtype);
void *csi_bc_map_dp1k(int op, int dtype);
void *csi_bc_map_ch8601(int op, int dtype);
void *csi_bc_func_table[CSINN_API_SIZE] = {
#ifdef CSI_BUILD_REF
    csi_bc_map_ref,
#else
    NULL, /* c code */
#endif
#ifdef CSI_BUILD_GREF
    csi_bc_map_gref,
#else
    NULL, /* gref */
#endif
    NULL, /* c860 */
#ifdef CSI_BUILD_C906
    csi_bc_map_c906,
#else
    NULL, /* c906 */
#endif
    NULL, /* c910 */
#ifdef CSI_BUILD_OPENVX
    csi_bc_map_ovx,
#else
    NULL, /* anole */
#endif
#ifdef CSI_BUILD_CH8601
    csi_bc_map_ch8601,
#else
    NULL, /* ch8601 */
#endif
#ifdef CSI_BUILD_PNNA
    csi_bc_map_pnna,
#else
    NULL, /* light */
#endif
#ifdef CSI_BUILD_DP1K
    csi_bc_map_dp1k,
#else
    NULL, /* dp1000 */
#endif
    NULL, /* tvmgen */
};

void *csi_bc_map(int api, int rmode, int op, int dtype)
{
    void* (*func)();
    if (rmode == CSINN_RM_CPU_GRAPH) {
        func = csi_bc_func_table[CSINN_GREF];
    } else {
        func = csi_bc_func_table[api];
    }
    return func(op, dtype);
}

void *csi_init_map_c906(int op, int dtype);
void *csi_init_map_ref(int op, int dtype);
void *csi_init_func_table[CSINN_API_SIZE] = {
#ifdef CSI_BUILD_REF
    csi_init_map_ref,/* c code */
#else
    NULL, /* c code */
#endif
    NULL, /* gref */
    NULL, /* c860 */
#ifdef CSI_BUILD_C906
    csi_init_map_c906,
#else
    NULL, /* c906 */
#endif
    NULL, /* c910 */
    NULL, /* anole */
    NULL, /* ch8601 */
    NULL, /* light */
    NULL, /* tvmgen */
};

void *csi_init_map(int api, int op, int dtype)
{
    void* (*func)() = csi_init_func_table[api];
    if (func != NULL) {
        return func(op, dtype);
    } else {
        return NULL;
    }
}

void csi_session_init(struct csi_session *sess)
{
    csi_debug_set_level(sess->debug_level);

    void* (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_SESSION_INIT, sess->base_dtype);
    if (func != NULL) {
        func(sess);
    }
}

void csi_session_deinit(struct csi_session *sess)
{
    void* (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_SESSION_DEINIT, sess->base_dtype);
    if (func != NULL) {
        func(sess);
    }
}

void csi_set_output_number(int number, struct csi_session *sess)
{
    sess->output_num = number;
    sess->output = calloc(sess->output_num, sizeof(struct csi_tensor *));
    void (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_SET_OUTPUT_NUMBER, sess->base_dtype);
    if (func != NULL) {
        func(number, sess);
    }
}

void csi_set_input_number(int number, struct csi_session *sess)
{
    sess->input_num = number;
    sess->input = calloc(sess->input_num, sizeof(struct csi_tensor *));
    void (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_SET_INPUT_NUMBER, sess->base_dtype);
    if (func != NULL) {
        func(number, sess);
    }
}

int csi_get_output_number(struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_GET_OUTPUT_NUMBER, sess->base_dtype);
    if (func != NULL) {
        return func(sess);
    } else {
        return sess->output_num;
    }
}

int csi_get_input_number(struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_GET_INPUT_NUMBER, sess->base_dtype);
    if (func != NULL) {
        return func(sess);
    } else {
        return sess->input_num;
    }
}

int csi_set_output(int index, struct csi_tensor *output, struct csi_session *sess)
{
    sess->output[index] = output;
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_SET_OUTPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, output, sess);
    }
    return CSINN_TRUE;
}

int csi_set_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    sess->input[index] = input;
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_SET_INPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, input, sess);
    }
    return CSINN_TRUE;
}

int csi_get_output(int index, struct csi_tensor *output, struct csi_session *sess)
{
    csi_tensor_copy(output, sess->output[index]);
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_GET_OUTPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, output, sess);
    }
    return CSINN_TRUE;
}

int csi_get_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    csi_tensor_copy(input, sess->input[index]);
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_GET_INPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, input, sess);
    }
    return CSINN_TRUE;
}

int csi_update_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    sess->input[index]->data = input->data;
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_UPDATE_INPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, input, sess);
    }
    return CSINN_TRUE;
}

int csi_update_output(int index, struct csi_tensor *output, struct csi_session *sess)
{
    sess->output[index]->data = output->data;
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_UPDATE_OUTPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, output, sess);
    }
    return CSINN_TRUE;
}

int csi_session_setup(struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_SESSION_SETUP, sess->base_dtype);
    if (func != NULL) {
        return func(sess);
    }
    return CSINN_FALSE;
}

int csi_session_run(struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_SESSION_RUN, sess->base_dtype);
    if (func != NULL) {
        return func(sess);
    }
    return CSINN_FALSE;
}

int csi_set_tensor_entry(struct csi_tensor *t, struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_TENSOR_ENTRY, sess->base_dtype);
    if (func != NULL) {
        return func(t, sess);
    }
    return CSINN_FALSE;
}

int csi_load_binary_model(char *path, struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, sess->base_run_mode, CSINN_LOAD_BG, sess->base_dtype);
    if (func != NULL) {
        return func(path, sess);
    }
    return CSINN_FALSE;
}
