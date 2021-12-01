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

INSTALL_PATH = path_to_csinn2
AQROOT := path_to_driver

VIVANTE_SDK_INC := ${AQROOT}/driver/build/sdk/include
OVXLIB_DIR := ${AQROOT}/lib/acuity-ovxlib-dev
VIVANTE_SDK_DIR := ${AQROOT}/driver/build/sdk
VIVANTE_SDK_LIB := ${VIVANTE_SDK_DIR}/drivers

INCLUDE += -I${VIVANTE_SDK_INC} -I${VIVANTE_SDK_INC}/HAL -I${AQROOT}/sdk/inc
INCLUDE += -I./ -I${OVXLIB_DIR}/include/utils -I${OVXLIB_DIR}/include/client
INCLUDE += -I${OVXLIB_DIR}/include/ops -I${OVXLIB_DIR}/include
INCLUDE += -I./include

CFLAGS = -O0 -g3

CXX = csky-abiv2-linux-g++
CFLAGS += -mcpu=ck860fv -mhard-float ${INCLUDE} -I${INSTALL_PATH}/include
LDFLAGS += ${INSTALL_PATH}/lib/libcsi_nn2_openvx.a -lpthread
LDFLAGS += -L${VIVANTE_SDK_LIB} -lNNArchPerf -lArchModelSw -lOpenVX -lOpenVXU -lCLC -lVSC -lm -lGAL -lNNGPUBinary
LDFLAGS += -L${OVXLIB_DIR}/lib -lovxlib -lOvx12VXCBinary


.PHONY: clean all

all: libaie.so

src/aie.o : src/aie.cpp
	${CXX} ${CFLAGS} -c -o $@ $^

model.params.o : model.params
	csky-abiv2-linux-ld -r -bbinary model.params -o model.params.o

model.o : model.c
	csky-abiv2-linux-gcc ${CFLAGS} -c -o $@ $^ -lstdc++

libaie.so : model.params.o model.o src/aie.o
	${CXX} -shared -o $@ $^ ${CFLAGS} ${LDFLAGS}

clean:
	rm src/*.o libaie.so -rf
	rm *.c *.h model.params *.o *.tensor -rf
