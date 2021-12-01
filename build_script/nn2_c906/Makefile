CROSS_COMPILE   ?= riscv64-unknown-linux-gnu-
CC               = $(CROSS_COMPILE)gcc
AR               = $(CROSS_COMPILE)ar

CFLAGS += -ffunction-sections -fdata-sections -Wl,--gc-sections
INCLUDE = -I../../include

all: nn2_c906

include nn2_c906.objs ../nn2.objs ../nn2_ref/nn2.objs ../nn2_gref/nn2.objs

nn2_c906: $(objects)
	$(AR) rcs $(INSTALL_DIR)/$(DSP_LIB).a $^

nn2_shared: $(objects)
	$(CC) -shared -o $(INSTALL_DIR)/$(DSP_LIB).so $^

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
%.o: %.S
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
