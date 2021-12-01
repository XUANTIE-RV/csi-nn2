CROSS_COMPILE   ?= csky-abiv2-elf-
CC        	= $(CROSS_COMPILE)gcc
AR        	= $(CROSS_COMPILE)ar

CFLAGS += -ffunction-sections -fdata-sections -Wl,--gc-sections
INCLUDE = -I../../include/include_xt800/dsp_include/ -I../../include/include_xt800/csi_core_include/ -I../../include/include_xt800/nn_include 
INCLUDE +=  -I../../include

all: nn2_e804

include nn2_dsp2.objs
include ../nn2.objs
include ../nn2_ref/nn2.objs

nn2_e804: $(objects)
	$(AR) rcs $(INSTALL_DIR)/$(DSP_LIB) $^

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
%.o: %.S
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
