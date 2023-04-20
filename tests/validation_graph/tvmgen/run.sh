#!/bin/sh -x

qemu-riscv64 -cpu c906fdv ./reg.o.elf
qemu-riscv64 -cpu c906fdv ./callback.o.elf
