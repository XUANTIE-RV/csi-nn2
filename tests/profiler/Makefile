# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

ROOT_DIR=$(shell cd ../..; pwd)
LIB_DIR=${ROOT_DIR}/install_nn2/x86
INCLUDE += -I${LIB_DIR}/include
LDFLAGS = -L${LIB_DIR}/lib

CC = gcc

CFLAGS += -O2 -g ${INCLUDE} -DSHL_TRACE

LDFLAGS += -lshl -lstdc++ -Wl,--gc-sections

.PHONY: clean all

all: test_shl_trace

test_shl_trace: test_trace.c
	$(CC) $(CFLAGS) -o $@  $^ $(LDFLAGS)

run: test_shl_trace
	./test_shl_trace

run_with_valgrind: test_shl_trace
	valgrind --tool=memcheck --leak-check=full ./test_shl_trace

clean:
	-rm test_shl_trace