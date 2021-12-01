CROSS_COMPILE   ?= csky-abiv2-linux-
CC               = $(CROSS_COMPILE)gcc
AR               = $(CROSS_COMPILE)ar

CFLAGS += -ffunction-sections -fdata-sections -Wl,--gc-sections -mhard-float
INCLUDE = -I../../include -I../../module/nna_ddk_install/include/

all: nn2

include nn2.objs
include ../nn2.objs

nn2: $(objects)
	$(AR) rcs $(INSTALL_DIR)/$(DSP_LIB).a $^

nn2_shared: $(objects)
	$(CC) -shared -o $(INSTALL_DIR)/$(DSP_LIB).so $^

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
%.o: %.S
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
