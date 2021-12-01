CROSS_COMPILE   ?= csky-abiv2-linux-
CC               = $(CROSS_COMPILE)gcc
AR               = $(CROSS_COMPILE)ar

CFLAGS += -ffunction-sections -fdata-sections -Wl,--gc-sections
INCLUDE = -I../../include -I../../module/acuity-driver/lib/acuity-ovxlib-dev/include/ -I../../module/acuity-driver/driver/build/sdk/include/

all: nn2

include nn2.objs
include ../nn2.objs

nn2: $(objects)
	$(AR) rcs $(INSTALL_DIR)/$(DSP_LIB) $^

nn2_shared: $(objects)
	$(CC) ${CFLAGS} -shared -o $(INSTALL_DIR)/$(DSP_LIB) $^ -L../../module/acuity-driver/driver/build/sdk/drivers -L../../module/acuity-driver/lib/acuity-ovxlib-dev/lib -lArchModelSw -lNNArchPerf -lOpenVX -lOpenVXU -lCLC -lVSC -lGAL -lNNGPUBinary -lovxlib -lOvx12VXCBinary

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
%.o: %.S
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
