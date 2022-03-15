SRS             = $(wildcard *.S)
SRC             = $(wildcard *.c)

INCLUDE		= -I./csi_core_include/

CFLAGS		+= $(INCLUDE)

OBJFILE		= $(SRS:%.S=%.o) $(SRC:%.c=%.o)

all: case

case: $(OBJFILE)

%.o: %.c
	@echo "  CC   $@"
	$(Q)$(CC) $(CFLAGS) -c $< -o $@
%.o: %.S
	@echo "  CC   $@"
	$(Q)$(CC) $(CFLAGS) -c $< -o $@
clean:
	 rm -f  *.o *.a *.asm *.elf
