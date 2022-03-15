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

# Makefile Example to deploy TVM modules.

ROOT_DIR=$(shell cd ../../..; pwd)
INCLUDE += -I${ROOT_DIR}/include
LDFLAGS = -L${ROOT_DIR}/x86_build -Wl,-unresolved-symbols=ignore-in-shared-libs

CC = gcc

CFLAGS += -O0 -g3 ${INCLUDE}

LDFLAGS += -lstdc++ -lcsi_nn2_hlight_x86 -lm

.PHONY: clean all

all: test_subgraph

test_subgraph: test_subgraph.c
	$(CC) $(CFLAGS) -o $@  $^ $(LDFLAGS)

clean:
	-rm subgraph