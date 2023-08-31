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

/**
 * @file csinn_runtime.h
 */

#ifndef INCLUDE_CSINN_RUNTIME_H_
#define INCLUDE_CSINN_RUNTIME_H_

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if (!defined SHL_BUILD_RTOS)
#include <omp.h>
#endif
#include "csinn_data_structure.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup AUX Auxiliary function
 */

/**
 * @defgroup TENSOR Tensor
 * @ingroup AUX
 */

/**
 * @defgroup OP OP parameters
 * @ingroup AUX
 */

/**
 * @defgroup SESSION Session
 * @ingroup AUX
 */

/**
 * @defgroup IO Input/output
 * @ingroup AUX
 */

#define VERSION_MAJOR 2
#define VERSION_MINOR 0
#define VERSION_PATCH 0
#define VERSION_SHIFT 8
int csinn_version(char *vstr);

/* tensor */
/**
 * @brief       Get the number of elements in the tensor
 *
 * @param[in]   tensor  The tensor to be counted
 * @return      Number of elements
 */
int csinn_tensor_size(struct csinn_tensor *tensor);

/**
 * @brief       Get the number of bytes of elements in the tensor
 *
 * @param[in]   tensor  The tensor to be counted
 * @return      Number of bytes
 */
int csinn_tensor_byte_size(struct csinn_tensor *tensor);

/**
 * @brief       Allocate a tensor structure
 *
 * @param[in]   session Reference session. If no reference is required, NULL can be passed in
 * @return      Allocted tensor <code>csinn_tensor*</code>
 */
struct csinn_tensor *csinn_alloc_tensor(struct csinn_session *session);

/**
 * @brief       Release a tensor structure
 *
 * @param[in]   tensor  The tensor to be released
 */
void csinn_free_tensor(struct csinn_tensor *tensor);

/**
 * @brief       Reallocate a specified amount of quantitative information
 *
 * @param[in, out]  tensor          The tensor to be operated
 * @param[in]       quant_info_num  Number of quantitative information
 */
void csinn_realloc_quant_info(struct csinn_tensor *tensor, int quant_info_num);

/**
 * @brief       Copying tensor, excluding data in tensor
 *
 * @param[out]  dest    Target tensor
 * @param[in]   src     Source tensor
 */
void csinn_tensor_copy(struct csinn_tensor *dest, struct csinn_tensor *src);

/**
 * @brief       Numerical conversion according to the data type of source and target tensor
 *
 * @param[out]  dest    Target tensor
 * @param[in]   src     Source tensor
 * @return      The return value is greater than 0 on success
 */
int csinn_tensor_data_convert(struct csinn_tensor *dest, struct csinn_tensor *src);

/**
 * @brief       Convert the layout of source and target tensor
 *
 * @param[out]  dest    Target tensor
 * @param[in]   src     Source tensor
 * @return      The return value is greater than 0 on success
 */
int csinn_tensor_layout_convert(struct csinn_tensor *dest, struct csinn_tensor *src);

/* op parameters */
/**
 * @brief       Allocate a basic structure common to all operators
 *
 * @param[in]   params_size Structure size
 * @param[in]   session     The session to be obtained
 * @return      Point to the allocated base structure
 */
void *csinn_alloc_params(int params_size, struct csinn_session *session);

/**
 * @brief       Release a structure described by an operator parameter
 *
 * @param[in]   params  Point to the operator parameter description structure to be released
 */
void csinn_free_params(void *params);

/* session */
/**
 * @brief       Allocate a session
 *
 * @return      Point to the newly allocated session <code>csinn_session*</code>
 *
 * @details     This interface only allocates memory space. Initialization is performed by
 *              <code>csinn_session_init</code> complete. Each session refers to an instance of a
 *              model.
 */
struct csinn_session *csinn_alloc_session();

/**
 * @brief       Release a session
 *
 * @param[in]   session The session to be released
 */
void csinn_free_session(struct csinn_session *session);

/**
 * @brief       Initialize session
 *
 * @param[in, out]  session The session to be initialized
 *
 * @details     Board specific initialization can be done in the interface,
 *              such as calling some initialization function interfaces of NPU.
 */
void csinn_session_init(struct csinn_session *session);

/**
 * @brief       Uninitialize session
 *
 * @param[in, out]  session The session to be uninitialized
 */
void csinn_session_deinit(struct csinn_session *session);

/**
 * @brief       Setup function
 *
 * @param[in]   session The session to be processed
 * @return      The return value is greater than 0 on success
 *
 * @details     Build the interface of the graph when executing according to the graph.
 *              For example, you can call the compilation interface in the NPU driver here.
 */
int csinn_session_setup(struct csinn_session *session);

/**
 * @brief       Run function
 *
 * @param[in]   session The session to be executed
 * @return      The return value is greater than 0 on success
 *
 * @details     When executing according to the graph, actually implement the interface of
 *              the graph.
 */
int csinn_session_run(struct csinn_session *session);

/**
 * @brief       Load binary model
 *
 * @param[in]   session The session to be executed
 * @return      The return value is greater than 0 on success
 *
 * @details     There is no need to call <code>csinn_session_setup</code> when model binary
 *              execution is adopted.
 */
int csinn_load_binary_model(struct csinn_session *session);

/**
 * @brief       Import binary model
 *
 * @return      Load the completed session structure when successful.
 *
 * @details     For the interface of hhb model, it is a further encapsulation of
 *              <code>csinn_load_binary_model</code>, defined as a weak function,
 *              and can be covered by a user-defined function with the same name.
 */
struct csinn_session *__attribute__((weak)) csinn_import_binary_model(char *bm_addr);

/* input/output */
/**
 * @brief       Set the input number of the model
 *
 * @param[in]   number  Input number
 * @param[out]  sess    The session to be set
 */
void csinn_set_input_number(int number, struct csinn_session *sess);

/**
 * @brief       Set the output number of the model
 *
 * @param[in]   number  Output number
 * @param[out]  sess    The session to be set
 */
void csinn_set_output_number(int number, struct csinn_session *sess);

/**
 * @brief       Get the input number of the model
 *
 * @param[in]   sess    The session to be get
 * @return      Return the input number
 */
int csinn_get_input_number(struct csinn_session *sess);

/**
 * @brief       Get the output number of the model
 *
 * @param[in]   sess    The session to be get
 * @return      Return the output number
 */
int csinn_get_output_number(struct csinn_session *sess);

/**
 * @brief       Set the specified input of the model
 *
 * @param[in]   index   Input index
 * @param[in]   input   Input tensor
 * @param[out]  sess    The session to be set
 * @return      The return value is greater than 0 on success.
 */
int csinn_set_input(int index, struct csinn_tensor *input, struct csinn_session *sess);

/**
 * @brief       Set the specified output of the model
 *
 * @param[in]   index   Output index
 * @param[in]   output   Output tensor
 * @param[out]  sess    The session to be set
 * @return      The return value is greater than 0 on success.
 */
int csinn_set_output(int index, struct csinn_tensor *output, struct csinn_session *sess);

/**
 * @brief       Get the specified input of the model
 *
 * @param[in]       index   Input index
 * @param[in, out]  input   Input tensor
 * @param[in]       sess    The session to be obtained
 * @return          The return value is greater than 0 on success.
 *
 * @details     The interface will modify the input information of
 *              the specified serial number into the parameter input.
 */
int csinn_get_input(int index, struct csinn_tensor *input, struct csinn_session *sess);

/**
 * @brief       Get the specified output of the model
 *
 * @param[in]       index   Output index
 * @param[in, out]  output  Output tensor
 * @param[in]       sess    The session to be obtained
 * @return          The return value is greater than 0 on success.
 *
 * @details     The interface will modify the output information of
 *              the specified serial number into the parameter output.
 */
int csinn_get_output(int index, struct csinn_tensor *output, struct csinn_session *sess);

/**
 * @brief           Update specified input information
 *
 * @param[in]       index   Input index
 * @param[in, out]  input   Input tensor
 * @param[in]       sess    The session to be obtained
 * @return          The return value is greater than 0 on success.
 */
int csinn_update_input(int index, struct csinn_tensor *input, struct csinn_session *sess);
/**
 * @brief           Update specified output information
 *
 * @param[in]       index   Output index
 * @param[in, out]  output  Output tensor
 * @param[in]       sess    The session to be obtained
 * @return          The return value is greater than 0 on success.
 */
int csinn_update_output(int index, struct csinn_tensor *output, struct csinn_session *sess);

/**
 * @brief       Set input nodes
 *
 * @param[in]   tensor  The tensor needs to be set as input
 * @param[out]  sess    The session to be set
 * @return      The return value is greater than 0 on success.
 */
int csinn_set_tensor_entry(struct csinn_tensor *tensor, struct csinn_session *sess);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_CSINN_RUNTIME_H_
