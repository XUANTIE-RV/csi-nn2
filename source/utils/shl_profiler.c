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

#include "shl_profiler.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "csinn_runtime.h"
#include "shl_debug.h"
#include "shl_memory.h"

#ifdef SHL_TRACE

const char *SHL_TRACE_EVENT_CATEGORY_NAMES[SHL_TRACE_EVENT_CATEGORY_MAX] = {
    "runtime", "cpu_operator", "memory", "cpu_kernel", "npu_kernel", "kernel",
};

const char *SHL_TRACE_EVENT_TYPE_NAMES[SHL_TRACE_EVENT_TYPE_MAX] = {"B", "E", "X", "i", "C", "b",
                                                                    "n", "e", "s", "t", "f", "M"};

uint32_t shl_trace_get_current_pid() { return (uint32_t)getpid(); }

uint32_t shl_trace_get_current_tid()
{
    uint32_t tid = syscall(__NR_gettid);
    return tid;
}

uint64_t shl_trace_get_timestamps_us()
{
    uint64_t ts = shl_get_timespec();  // ns
    ts /= 1000;
    return ts;
}

void *shl_trace_alloc(int64_t size) { return calloc(1, size); }

void shl_trace_free(void *ptr)
{
    if (ptr) {
        free(ptr);
    }
}

struct shl_trace_value *shl_trace_create_string(const char *value)
{
    char *cpy_value = (char *)shl_trace_alloc(strlen(value) + 1);
    memcpy(cpy_value, value, strlen(value) + 1);

    struct shl_trace_value *res =
        (struct shl_trace_value *)shl_trace_alloc(sizeof(struct shl_trace_value));
    res->type = SHL_TRACE_VALUE_TYPE_STRING;
    res->content.str = cpy_value;

    return res;
}

struct shl_trace_value *shl_trace_create_int64(int64_t value)
{
    struct shl_trace_value *res =
        (struct shl_trace_value *)shl_trace_alloc(sizeof(struct shl_trace_value));
    res->content.i64 = value;
    res->type = SHL_TRACE_VALUE_TYPE_INT64;
    return res;
}

struct shl_trace_value *shl_trace_create_uint64(uint64_t value)
{
    struct shl_trace_value *res =
        (struct shl_trace_value *)shl_trace_alloc(sizeof(struct shl_trace_value));
    res->content.u64 = value;
    res->type = SHL_TRACE_VALUE_TYPE_UINT64;
    return res;
}

struct shl_trace_value *shl_trace_create_double(double value)
{
    struct shl_trace_value *res =
        (struct shl_trace_value *)shl_trace_alloc(sizeof(struct shl_trace_value));
    res->content.f64 = value;
    res->type = SHL_TRACE_VALUE_TYPE_DOUBLE;
    return res;
}

struct shl_trace_value *shl_trace_create_list(int num, ...)
{
    struct shl_trace_value *res =
        (struct shl_trace_value *)shl_trace_alloc(sizeof(struct shl_trace_value));

    struct shl_trace_value_list *list =
        (struct shl_trace_value_list *)shl_trace_alloc(sizeof(struct shl_trace_value_list));
    list->size = num;
    list->value =
        (struct shl_trace_value **)shl_trace_alloc(sizeof(struct shl_trace_value *) * num);

    res->type = SHL_TRACE_VALUE_TYPE_LIST;
    res->content.list = list;

    va_list args;
    va_start(args, num);
    for (int i = 0; i < num; i++) {
        struct shl_trace_value *value = va_arg(args, struct shl_trace_value *);
        list->value[i] = value;
    }

    va_end(args);
    return res;
}

struct shl_trace_value *shl_trace_create_list_int(int num, int *arr)
{
    struct shl_trace_value *res =
        (struct shl_trace_value *)shl_trace_alloc(sizeof(struct shl_trace_value));

    struct shl_trace_value_list *list =
        (struct shl_trace_value_list *)shl_trace_alloc(sizeof(struct shl_trace_value_list));
    list->size = num;
    list->value =
        (struct shl_trace_value **)shl_trace_alloc(sizeof(struct shl_trace_value *) * num);

    res->type = SHL_TRACE_VALUE_TYPE_LIST;
    res->content.list = list;

    for (int i = 0; i < num; i++) {
        list->value[i] = shl_trace_create_int64(arr[i]);
    }

    return res;
}

struct shl_trace_event_format *shl_trace_create_common_event()
{
    struct shl_trace_event_format *event =
        (struct shl_trace_event_format *)shl_trace_alloc(sizeof(struct shl_trace_event_format));
    event->ts = shl_trace_get_timestamps_us();
    event->pid = shl_trace_get_current_pid();
    event->tid = shl_trace_get_current_tid();
    return event;
}

struct shl_trace_dict_item *shl_trace_create_dict_item(const char *key,
                                                       struct shl_trace_value *value)
{
    struct shl_trace_dict_item *item =
        (struct shl_trace_dict_item *)shl_trace_alloc(sizeof(struct shl_trace_dict_item));
    memcpy(item->key, key, strlen(key) + 1);
    item->value = value;
    return item;
}

struct shl_trace_dict *shl_trace_create_dict_by_item(int argc, ...)
{
    if (argc <= 0) return NULL;
    struct shl_trace_dict *data =
        (struct shl_trace_dict *)shl_trace_alloc(sizeof(struct shl_trace_dict));

    if (argc < SHL_TRACE_EVENT_ARGS_CAPACITY_STEP) {
        data->items_capacity = SHL_TRACE_EVENT_ARGS_CAPACITY_STEP;
    } else {
        data->items_capacity = argc;
    }
    data->items_size = 0;
    data->items = (struct shl_trace_dict_item **)shl_trace_alloc(
        sizeof(struct shl_trace_dict_item *) * data->items_capacity);

    va_list args;
    va_start(args, argc);
    for (int i = 0; i < argc; i++) {
        data->items[i] = va_arg(args, struct shl_trace_dict_item *);
        data->items_size++;
    }
    va_end(args);

    return data;
}

struct shl_trace_dict *shl_trace_create_dict(int argc, ...)
{
    if (argc <= 0) return NULL;
    struct shl_trace_dict *data =
        (struct shl_trace_dict *)shl_trace_alloc(sizeof(struct shl_trace_dict));

    if (argc < SHL_TRACE_EVENT_ARGS_CAPACITY_STEP) {
        data->items_capacity = SHL_TRACE_EVENT_ARGS_CAPACITY_STEP;
    } else {
        data->items_capacity = argc;
    }
    data->items_size = 0;
    data->items = (struct shl_trace_dict_item **)shl_trace_alloc(
        sizeof(struct shl_trace_dict_item *) * data->items_capacity);

    va_list args;
    va_start(args, argc);
    for (int i = 0; i < argc; i++) {
        struct shl_trace_dict_item *item =
            (struct shl_trace_dict_item *)shl_trace_alloc(sizeof(struct shl_trace_dict_item));

        char *key = va_arg(args, char *);
        memcpy(item->key, key, strlen(key) + 1);
        item->value = va_arg(args, struct shl_trace_value *);

        data->items[i] = item;
        data->items_size++;
    }
    va_end(args);

    return data;
}

void shl_trace_release_value(struct shl_trace_value *value)
{
    if (value->type == SHL_TRACE_VALUE_TYPE_STRING) {
        shl_trace_free(value->content.str);
        shl_trace_free(value);
    } else if (value->type == SHL_TRACE_VALUE_TYPE_LIST) {
        // there may be a list nested within a list
        for (int i = 0; i < value->content.list->size; i++) {
            shl_trace_release_value(value->content.list->value[i]);
        }
        shl_trace_free(value->content.list->value);
        shl_trace_free(value->content.list);
        shl_trace_free(value);
    } else {
        shl_trace_free(value);
    }
}

void shl_trace_release_dict(struct shl_trace_dict *args)
{
    for (int i = 0; i < args->items_size; i++) {
        struct shl_trace_dict_item *item = args->items[i];
        shl_trace_release_value(item->value);
        shl_trace_free(item);
    }
    shl_trace_free(args->items);
    shl_trace_free(args);
}

void shl_trace_insert_event(struct shl_trace *trace, struct shl_trace_event_format *event)
{
    if (trace->events_size + 1 > trace->events_capacity) {
        trace->events = (struct shl_trace_event_format **)shl_mem_realloc(
            trace->events,
            sizeof(struct shl_trace_event_format *) *
                (trace->events_capacity + SHL_TRACE_EVENT_CAPACITY_STEP),
            sizeof(struct shl_trace_event_format *) * trace->events_capacity);
        trace->events_capacity += SHL_TRACE_EVENT_CAPACITY_STEP;
    }
    trace->events[trace->events_size] = event;
    trace->events_size++;
}

void shl_trace_init(struct shl_trace *trace)
{
    // initialize data field
    trace->events_capacity = SHL_TRACE_EVENT_CAPACITY_STEP;
    trace->events_size = 0;
    trace->events = (struct shl_trace_event_format **)shl_trace_alloc(
        sizeof(struct shl_trace_event_format *) * trace->events_capacity);
    trace->other_data =
        (struct shl_trace_other_data *)shl_trace_alloc(sizeof(struct shl_trace_other_data));
    strcpy(trace->other_data->version, SHL_TRACE_VERSION);
    trace->is_init = true;

    uint64_t ts = shl_trace_get_timestamps_us();
    snprintf(trace->filename, sizeof(trace->filename), "model_csinn.trace.%lu.json", ts);
}

void shl_trace_deinit(struct shl_trace *trace)
{
    if (!trace->is_init) return;
    // release events
    for (int i = 0; i < trace->events_size; i++) {
        struct shl_trace_event_format *event = trace->events[i];
        if (event->args && event->args->items_size > 0) {
            shl_trace_release_dict(event->args);
        }
        shl_trace_free(event);
    }
    shl_trace_free(trace->events);
    trace->events = NULL;
    trace->events_capacity = 0;
    trace->events_size = 0;

    // release other_data
    if (trace->other_data->data && trace->other_data->data->items_size > 0) {
        shl_trace_release_dict(trace->other_data->data);
    }
    shl_trace_free(trace->other_data);
    trace->other_data = NULL;

    trace->is_init = false;
}

static void indent(FILE *file, int num)
{
    for (int i = 0; i < num; i++) {
        fprintf(file, " ");
    }
}

#define WRITE_ONELINE(file, indent_num, ...) \
    indent(file, indent_num);                \
    fprintf(file, __VA_ARGS__);

static void write_trace_value_to_file(FILE *file, struct shl_trace_value value)
{
    switch (value.type) {
        case SHL_TRACE_VALUE_TYPE_INT64:
            fprintf(file, "%ld", value.content.i64);
            break;
        case SHL_TRACE_VALUE_TYPE_UINT64:
            fprintf(file, "%lu", value.content.u64);
            break;
        case SHL_TRACE_VALUE_TYPE_DOUBLE:
            fprintf(file, "%f", value.content.f64);
            break;
        case SHL_TRACE_VALUE_TYPE_STRING:
            fprintf(file, "\"%s\"", value.content.str);
            break;
        case SHL_TRACE_VALUE_TYPE_LIST:
            if (value.content.list->size >= 0) {
                fprintf(file, "[");
                for (int i = 0; i < value.content.list->size; i++) {
                    write_trace_value_to_file(file, *value.content.list->value[i]);
                    if (i != value.content.list->size - 1) {
                        fprintf(file, ", ");
                    }
                }
                fprintf(file, "]");
            }
            break;
        default:
            break;
    }
}

static void write_trace_dict_to_file(FILE *file, struct shl_trace_dict *dict, int space)
{
    for (int i = 0; i < dict->items_size; i++) {
        struct shl_trace_dict_item *item = dict->items[i];
        WRITE_ONELINE(file, space, "\"%s\": ", item->key);
        write_trace_value_to_file(file, *item->value);
        if (i == dict->items_size - 1) {
            fprintf(file, "\n");
        } else {
            fprintf(file, ",\n");
        }
    }
}

void shl_trace_to_json(struct shl_trace *trace)
{
    if (!trace->events || trace->events_size == 0 || trace->events_capacity == 0) return;

    int space_step = 2;
    FILE *file = fopen(trace->filename, "w");
    if (!file) {
        shl_debug_error("Failed to open file: %s\n", trace->filename);
        return;
    }
    fprintf(file, "{\n");

    // other data
    WRITE_ONELINE(file, space_step, "\"otherData\": {\n");
    WRITE_ONELINE(file, space_step * 2, "\"version\": \"%s\"", trace->other_data->version);
    struct shl_trace_dict *extra_data = trace->other_data->data;
    if (extra_data && extra_data->items_size > 0) {
        fprintf(file, ",\n");
        write_trace_dict_to_file(file, extra_data, space_step * 2);
    } else {
        fprintf(file, "\n");
    }
    WRITE_ONELINE(file, space_step, "},\n");  // otherData end

    // events
    WRITE_ONELINE(file, space_step, "\"traceEvents\": [\n");
    for (int i = 0; i < trace->events_size; i++) {
        WRITE_ONELINE(file, space_step * 2, "{\n");

        struct shl_trace_event_format *event = trace->events[i];
        WRITE_ONELINE(file, space_step * 3, "\"name\": \"%s\",\n", event->name);
        WRITE_ONELINE(file, space_step * 3, "\"cat\": \"%s\",\n",
                      SHL_TRACE_EVENT_CATEGORY_NAMES[event->cat]);
        WRITE_ONELINE(file, space_step * 3, "\"ph\": \"%s\",\n",
                      SHL_TRACE_EVENT_TYPE_NAMES[event->ph]);
        WRITE_ONELINE(file, space_step * 3, "\"ts\": %lu,\n", event->ts);
        WRITE_ONELINE(file, space_step * 3, "\"pid\": %u,\n", event->pid);
        WRITE_ONELINE(file, space_step * 3, "\"tid\": %u", event->tid);

        if (event->args && event->args->items_size > 0) {
            fprintf(file, ",\n");
            WRITE_ONELINE(file, space_step * 3, "\"args\": {\n");
            write_trace_dict_to_file(file, event->args, space_step * 4);
            WRITE_ONELINE(file, space_step * 3, "}\n");
        } else {
            fprintf(file, "\n");
        }

        if (i == trace->events_size - 1) {
            WRITE_ONELINE(file, space_step * 2, "}\n");
        } else {
            WRITE_ONELINE(file, space_step * 2, "},\n");
        }
    }
    WRITE_ONELINE(file, space_step, "]\n");  // traceEvents end

    fprintf(file, "}\n");  // json end
    fclose(file);
    shl_debug_info("Trace data saved to %s\n", trace->filename);
}

void shl_trace_move_events(struct shl_trace *from_trace, struct shl_trace *to_trace)
{
    if (!from_trace || !from_trace->events_size) return;
    if (!to_trace || !to_trace->events_size) return;
    for (int i = 0; i < from_trace->events_size; i++) {
        shl_trace_insert_event(to_trace, from_trace->events[i]);
    }
    from_trace->events_size = 0;
    from_trace->events_capacity = 0;
}

void shl_trace_begin(struct shl_trace *trace, const char *filename)
{
    if (!trace || !trace->enable_trace) return;
    shl_trace_init(trace);
    if (filename != NULL) {
        memcpy(trace->filename, filename, strlen(filename) + 1);
    }
}

void shl_trace_end(struct shl_trace *trace)
{
    if (!trace) return;
    shl_trace_to_json(trace);
    shl_trace_deinit(trace);
}

void shl_trace_other_data(struct shl_trace *trace, struct shl_trace_dict *data)
{
    if (!trace || !trace->enable_trace || !trace->is_init) return;
    trace->other_data->data = data;
}

void shl_trace_duration_begin(struct shl_trace *trace, const char *name,
                              enum shl_trace_event_category cat, struct shl_trace_dict *args)
{
    if (!trace || !trace->enable_trace || !trace->is_init) return;
    struct shl_trace_event_format *event = shl_trace_create_common_event();
    memcpy(event->name, name, strlen(name) + 1);
    event->cat = cat;
    event->ph = SHL_TRACE_EVENT_TYPE_DURATION_B;
    event->args = args;

    // update
    shl_trace_insert_event(trace, event);
}

void shl_trace_duration_end(struct shl_trace *trace, const char *name,
                            enum shl_trace_event_category cat, struct shl_trace_dict *args)
{
    if (!trace || !trace->enable_trace || !trace->is_init) return;
    struct shl_trace_event_format *event = shl_trace_create_common_event();
    memcpy(event->name, name, strlen(name) + 1);
    event->cat = cat;
    event->ph = SHL_TRACE_EVENT_TYPE_DURATION_E;
    event->args = args;

    // update
    shl_trace_insert_event(trace, event);
}
#endif
