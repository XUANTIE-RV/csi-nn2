CROSS_COMPILE   ?= csky-abiv2-linux-
CC               = $(CROSS_COMPILE)gcc
AR               = $(CROSS_COMPILE)ar

CFLAGS += -ffunction-sections -fdata-sections -Wl,--gc-sections
INCLUDE = -I../../include

all: nn2

include nn2.objs
include ../nn2.objs
include ../nn2_ref/nn2.objs

nn2: $(objects)
	$(AR) rcs $(INSTALL_DIR)/$(DSP_LIB) $^

nn2_shared: $(objects)
	$(CC) -shared -o $(INSTALL_DIR)/$(DSP_LIB) $^

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
%.o: %.S
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
