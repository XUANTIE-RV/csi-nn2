CC               = $(CROSS_COMPILE)gcc
CXX              = $(CROSS_COMPILE)g++
AR               = $(CROSS_COMPILE)ar

CFLAGS += -ffunction-sections -fdata-sections -Wl,--gc-sections
INCLUDE = -I../../include -I../../module/nna_ddk_install/include/

all: nn2

include nn2.objs
include ../nn2.objs
include ../nn2_ref/nn2.objs

nn2: $(objects)
	$(AR) rcs $(INSTALL_DIR)/$(DSP_LIB) $^

nn2_shared_x86: $(objects)
	$(CC) -shared -o $(INSTALL_DIR)/$(DSP_LIB) $^ -L../../module/nna_ddk_install/x86/ -limgdnn_csim -lnnasession_csim

nn2_shared: $(objects)
	$(CC) -shared -o $(INSTALL_DIR)/$(DSP_LIB) $^ -L../../module/nna_ddk_install/light/ -limgdnn -lnnasession

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
%.o: %.S
	$(CC) -c $(CFLAGS) $(INCLUDE) $(MACRO) $(ENDIAN) $< -o $@
