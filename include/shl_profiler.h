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
#ifndef INCLUDE_PROFILER_H_
#define INCLUDE_PROFILER_H_

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "shl_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SHL_TRACE_VERSION "2.9.5"

#define SHL_TRACE_EVENT_ARGS_ITEM_KEY_MAX 64
#define SHL_TRACE_EVENT_ARGS_CAPACITY_STEP 8
#define SHL_TRACE_EVENT_CAPACITY_STEP 32
#define SHL_TRACE_FILENAME_LENGTH_MAX 128
#define SHL_TRACE_EVENT_NAME 64

enum shl_trace_event_category {
    SHL_TRACE_EVENT_RUNTIME = 0,
    SHL_TRACE_EVENT_CPU_OPERATOR,
    SHL_TRACE_EVENT_MEMORY,
    SHL_TRACE_EVENT_CPU_KERNEL,
    SHL_TRACE_EVENT_NPU_KERNEL,
    SHL_TRACE_EVENT_KERNEL,
    SHL_TRACE_EVENT_CATEGORY_MAX
};

extern const char *SHL_TRACE_EVENT_CATEGORY_NAMES[];

enum shl_trace_event_type {
    SHL_TRACE_EVENT_TYPE_DURATION_B = 0,
    SHL_TRACE_EVENT_TYPE_DURATION_E,
    SHL_TRACE_EVENT_TYPE_COMPLETE_X,
    SHL_TRACE_EVENT_TYPE_INSTANT_i,
    SHL_TRACE_EVENT_TYPE_COUNTER_C,
    SHL_TRACE_EVENT_TYPE_ASYNC_b,
    SHL_TRACE_EVENT_TYPE_ASYNC_n,
    SHL_TRACE_EVENT_TYPE_ASYNC_e,
    SHL_TRACE_EVENT_TYPE_FLOW_s,
    SHL_TRACE_EVENT_TYPE_FLOW_t,
    SHL_TRACE_EVENT_TYPE_FLOW_f,
    SHL_TRACE_EVENT_TYPE_METADATA_M,
    SHL_TRACE_EVENT_TYPE_MAX,
};

extern const char *SHL_TRACE_EVENT_TYPE_NAMES[];

enum shl_trace_value_type {
    SHL_TRACE_VALUE_TYPE_INT64,
    SHL_TRACE_VALUE_TYPE_UINT64,
    SHL_TRACE_VALUE_TYPE_DOUBLE,
    SHL_TRACE_VALUE_TYPE_STRING,
    SHL_TRACE_VALUE_TYPE_LIST,
};

union shl_trace_value_content {
    int64_t i64;
    uint64_t u64;
    double f64;
    char *str;
    struct shl_trace_value_list *list;
};

struct shl_trace_value {
    enum shl_trace_value_type type;
    union shl_trace_value_content content;
};

struct shl_trace_value_list {
    struct shl_trace_value **value;
    int size;
};

struct shl_trace_dict_item {
    char key[32];
    struct shl_trace_value *value;
};

struct shl_trace_dict {
    struct shl_trace_dict_item **items;
    uint32_t items_capacity;
    uint32_t items_size;
};

struct shl_trace_event_format {
    char name[SHL_TRACE_EVENT_NAME];   /** The name of the event.*/
    enum shl_trace_event_category cat; /** The event categories. */
    enum shl_trace_event_type ph;      /** The event type. */
    uint64_t ts;                       /** The tracing clock timestamps */
    // uint64_t tts;                        /** The thread clock timestamps */
    uint32_t pid;                /** The process id. */
    uint32_t tid;                /** The thread id. */
    struct shl_trace_dict *args; /** Any arguments provided for the event. */
};

struct shl_trace_other_data {
    char version[32];
    struct shl_trace_dict *data;
};

struct shl_trace {
    bool enable_trace;
    bool is_init;
    char filename[SHL_TRACE_FILENAME_LENGTH_MAX];
    struct shl_trace_event_format **events;
    uint32_t events_capacity;
    uint32_t events_size;

    struct shl_trace_other_data *other_data;
};

uint32_t shl_trace_get_current_pid();
uint32_t shl_trace_get_current_tid();
uint64_t shl_trace_get_timestamps_us();

struct shl_trace_value *shl_trace_create_string(const char *value);
struct shl_trace_value *shl_trace_create_int64(int64_t value);
struct shl_trace_value *shl_trace_create_uint64(uint64_t value);
struct shl_trace_value *shl_trace_create_double(double value);
/**
 * Create variable length list
 *
 * For example:
 *  value = shl_trace_create_list(2,
 *              shl_trace_create_string("value"),
 *              shl_trace_create_int64(10)
 *          )
 */
struct shl_trace_value *shl_trace_create_list(int num, ...);
struct shl_trace_value *shl_trace_create_list_int(int num, int *arr);

#define SHL_TRACE_STRING(value) shl_trace_create_string(value)
#define SHL_TRACE_INT64(value) shl_trace_create_int64(value)
#define SHL_TRACE_UINT64(value) shl_trace_create_uint64(value)
#define SHL_TRACE_DOUBLE(value) shl_trace_create_double(value)
#define SHL_TRACE_LIST(num, ...) shl_trace_create_list(num, __VA_ARGS__)
#define SHL_TRACE_LIST_INT(num, ptr) shl_trace_create_list_int(num, ptr)

/* release value itself and its members. */
void shl_trace_release_value(struct shl_trace_value *value);

struct shl_trace_dict_item *shl_trace_create_dict_item(const char *key,
                                                       struct shl_trace_value *value);
struct shl_trace_dict *shl_trace_create_dict_by_item(int argc, ...);

/**
 * Create dict with variable length (key, value)
 *
 * For example:
 *  dict = shl_trace_create_dict(5,
 *       "string", shl_trace_create_string("string"),
 *       "int64", shl_trace_create_int64(-5),
 *       "uint64", shl_trace_create_uint64(4),
 *       "double", shl_trace_create_double(4.6),
 *       "list", shl_trace_create_list(
 *           2,
 *           shl_trace_create_int64(256),
 *           shl_trace_create_int64(256)
 *       )
 *   );
 */
struct shl_trace_dict *shl_trace_create_dict(int argc, ...);

/* release value itself and its members. */
void shl_trace_release_dict(struct shl_trace_dict *dict);

struct shl_trace_event_format *shl_trace_create_common_event();
void shl_trace_insert_event(struct shl_trace *trace, struct shl_trace_event_format *event);
void shl_trace_init(struct shl_trace *trace);
void shl_trace_deinit(struct shl_trace *trace);
void shl_trace_to_json(struct shl_trace *trace);
void shl_trace_move_events(struct shl_trace *from_trace, struct shl_trace *to_trace);

/************************** Main functions ***************************/
#ifdef SHL_TRACE
#define SHL_TRACE_CALL(func) func
void shl_trace_begin(struct shl_trace *trace, const char *filename);
void shl_trace_end(struct shl_trace *trace);
void shl_trace_other_data(struct shl_trace *trace, struct shl_trace_dict *data);
void shl_trace_duration_begin(struct shl_trace *trace, const char *name,
                              enum shl_trace_event_category cat, struct shl_trace_dict *args);
void shl_trace_duration_end(struct shl_trace *trace, const char *name,
                            enum shl_trace_event_category cat, struct shl_trace_dict *args);
#else
#define SHL_TRACE_CALL(func)
inline void shl_trace_begin(struct shl_trace *trace, const char *filename) {}
inline void shl_trace_end(struct shl_trace *trace) {}
inline void shl_trace_other_data(struct shl_trace *trace, struct shl_trace_dict *data) {}
inline void shl_trace_duration_begin(struct shl_trace *trace, const char *name,
                                     enum shl_trace_event_category cat, struct shl_trace_dict *args)
{
}
inline void shl_trace_duration_end(struct shl_trace *trace, const char *name,
                                   enum shl_trace_event_category cat, struct shl_trace_dict *args)
{
}
#endif

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_PROFILER_H_
