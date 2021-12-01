

ENTRY(__start)

MEMORY
{
	INST   : ORIGIN = 0x00000000 , LENGTH = 0x01000000    /* ROM  16MB*/
	DATA   : ORIGIN = 0x20000000 , LENGTH = 0x00800000    /* RAM  8MB*/
	EEPROM : ORIGIN = 0x60000000 , LENGTH = 0x01000000
}

PROVIDE (__stack =  0x60000000 + 0x01000000 - 0x10);
PROVIDE (__heap_start =  0x20800000);
PROVIDE (__heap_end =  0x20800000 + 0x800000);


SECTIONS
{
    .text : {
		. = ALIGN(0x4) ;
		*crt0.o(.exp_table)
		__stext = . ;
		*(.text)
		*(.text*)
		*(.init)
		*(.fini)

		. = ALIGN(0x10) ;
		__etext = . ;
    } > INST

    .rodata : {
		. = ALIGN(0x4) ;
		__srodata = .;
		*(.rdata)
		*(.rdata*)
		*(.rodata)
		*(.rodata*)
		. = ALIGN(0x10) ;
		__erodata = .;
    } > INST

    .data : {
		. = ALIGN(0x4) ;
		__sdata = . ;
		__data_start = . ;
		data_start = . ;
		*(.got.plt)
		*(.got)
		FILL(0) ;
		. = ALIGN(0x20) ;
		LONG(-1)
		. = ALIGN(0x20) ;
		*(.gnu.linkonce.r*)
		*(.data)
		*(.data*)
		*(.gnu.linkonce.d*)
		*(.gcc_except_table)
		*(.gcc_except_table*)
		*(.sdata)
		*(.sdata.*)
		*(.gnu.linkonce.s.*)
		*(__libc_atexit)
		*(__libc_subinit)
		*(__libc_subfreeres)
		*(.note.ABI-tag)

		. = ALIGN(0x10) ;
		__edata = . ;
	} > DATA

    .bss : {
		. = ALIGN(0x4) ;
		__sbss = ALIGN(0x4) ;
		__bss_start = . ;
		*(.dynsbss)
		*(.sbss)
		*(.sbss.*)
		*(.scommon)
		*(.dynbss)
		*(.bss)
		*(.bss.*)
		*(COMMON)
		. = ALIGN(0x10) ;
		__ebss = . ;
		__end = . ;
		end = . ;
	} > DATA
}
