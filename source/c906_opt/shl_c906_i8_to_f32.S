
/**************************************************************************************************

    void shl_c906_i8_to_f32(const int8_t *input,
                            float *output,
                            int32_t offset,
                            float *scale,
                            uint32_t length)

    Algorithm works as follows:
        (1)

    register definition:
        a0: input addr
        a1: output addr
        a2: offset
        a3: scale point
        a4: element length

 *************************************************************************************************/
#define SHL_C906_8_TO_F32 shl_c906_i8_to_f32
#define SHL_C906_8_TO_F32_SIGNED

    .file           "shl_c906_i8_to_f32.S"
    .section        .text.SHL_C906_8_TO_F32, "ax", @progbits
    .align          5
    .global         SHL_C906_8_TO_F32
    .type           SHL_C906_8_TO_F32, @function

#include "./shl_c906_u8_to_f32.S"

#undef SHL_C906_8_TO_F32_SIGNED
