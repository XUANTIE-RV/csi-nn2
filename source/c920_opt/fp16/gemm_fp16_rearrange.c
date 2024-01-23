#include "c920/c920.h"

static inline vfloat16m4_t vdeq_vf_f16m4(vint8m2_t _src, __fp16 scale, int vl)
{
    vint16m4_t _i16 = vwadd_vx_i16m4(_src, 0, vl);
    vfloat16m4_t _f16 = vfcvt_f_x_v_f16m4(_i16, vl);
    _f16 = vfmul_vf_f16m4(_f16, scale, vl);
    return _f16;
}

/**
 * 1 the multithread is simply divided by 32, if necessary, it can be redesigned by thread rank.
 */
static inline void gemm_dot_1x1_fp16_q8_rearrange(__fp16 *dst, const __fp16 *sa, const int8_t *sb,
                                                  const __fp16 *scale, int M, int K, int N)
{
    int block_size = 32;
    for (int i = 0; i < M; i++) {
        int vl = vsetvl_e16m4(block_size);
        const __fp16 *a0_ptr = sa + i * K;

        if (shl_multithread_is_enable()) {
#pragma omp parallel for
            for (int j = 0; j < N - 31; j += 32) {
                const __fp16 *s0_ptr = scale + j / 32 * K;
                const int8_t *b0_ptr = sb + j * K;
                __fp16 *dst_ptr = dst + i * N + j;
                vfloat16m4_t _acc00 = vfmv_v_f_f16m4(0.0f, vl);
                for (int k = 0; k < K; k++) {
                    vint8m2_t _b0_i8 = vle8_v_i8m2(b0_ptr, vl);
                    vfloat16m4_t _b0_f32 = vdeq_vf_f16m4(_b0_i8, s0_ptr[0], vl);
                    _acc00 = vfmacc_vf_f16m4(_acc00, a0_ptr[k], _b0_f32, vl);
                    s0_ptr += 1;
                    b0_ptr += block_size;
                }
                vse16_v_f16m4(dst_ptr, _acc00, vl);
            }
        } else {
            for (int j = 0; j < N - 31; j += 32) {
                const __fp16 *s0_ptr = scale + j / 32 * K;
                const int8_t *b0_ptr = sb + j * K;
                __fp16 *dst_ptr = dst + i * N + j;
                vfloat16m4_t _acc00 = vfmv_v_f_f16m4(0.0f, vl);
                for (int k = 0; k < K; k++) {
                    vint8m2_t _b0_i8 = vle8_v_i8m2(b0_ptr, vl);
                    vfloat16m4_t _b0_f32 = vdeq_vf_f16m4(_b0_i8, s0_ptr[0], vl);
                    _acc00 = vfmacc_vf_f16m4(_acc00, a0_ptr[k], _b0_f32, vl);
                    s0_ptr += 1;
                    b0_ptr += block_size;
                }
                vse16_v_f16m4(dst_ptr, _acc00, vl);
            }
        }
    }
}

void shl_c920_gemm_a0nb1_dot_fp16_q8_rearrange(__fp16 *dst, const __fp16 *sa, const int8_t *sb,
                                               __fp16 *bias, int M, int K, int N,
                                               const __fp16 *scale)
{
    // dequantize in gemm
    gemm_dot_1x1_fp16_q8_rearrange(dst, sa, sb, scale, M, K, N);
}

/**
 * 1 for int4 weight in single core, the computational efficiency may be improved.
 * 2 the multithread is simply divided by 32, if necessary, it can be redesigned by thread rank.
 */
static inline void gemm_dot_1x1_fp16_q4_rearrange(__fp16 *dst, const __fp16 *sa, const int8_t *sb,
                                                  const __fp16 *scale, int M, int K, int N)
{
    int block_size = 32;
    int half_block = block_size / 2;  // 16
    for (int i = 0; i < M; i++) {
        int vl = vsetvl_e16m2(half_block);
        const __fp16 *a0_ptr = sa + i * K;

        if (shl_multithread_is_enable()) {
#pragma omp parallel for
            for (int j = 0; j < N - 31; j += 32) {
                const __fp16 *s0_ptr = scale + j / 32 * K;
                const int8_t *b0_ptr = sb + j / 2 * K;
                __fp16 *dst_ptr = dst + i * N + j;

                vfloat16m2_t _acc00 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc01 = vfmv_v_f_f16m2(0.0f, vl);
                for (int k = 0; k < K; k++) {
                    vint8m1_t _b0_i8 = vle8_v_i8m1(b0_ptr, vl);
                    vint8m1_t _low_i8 = vand_vx_i8m1(_b0_i8, 0x0f, vl);
                    vint8m1_t _high_i8 = vsra_vx_i8m1(_b0_i8, 4, vl);
                    _high_i8 = vand_vx_i8m1(_high_i8, 0x0f, vl);
                    vint16m2_t _low_i16 = vwsub_vx_i16m2(_low_i8, 8, vl);
                    vint16m2_t _high_i16 = vwsub_vx_i16m2(_high_i8, 8, vl);
                    vfloat16m2_t _low_f16 = vfcvt_f_x_v_f16m2(_low_i16, vl);
                    vfloat16m2_t _high_f16 = vfcvt_f_x_v_f16m2(_high_i16, vl);
                    _low_f16 = vfmul_vf_f16m2(_low_f16, s0_ptr[0], vl);
                    _high_f16 = vfmul_vf_f16m2(_high_f16, s0_ptr[0], vl);
                    _acc00 = vfmacc_vf_f16m2(_acc00, a0_ptr[k], _low_f16, vl);
                    _acc01 = vfmacc_vf_f16m2(_acc01, a0_ptr[k], _high_f16, vl);
                    s0_ptr += 1;
                    b0_ptr += half_block;
                }
                vse16_v_f16m2(dst_ptr, _acc00, vl);
                vse16_v_f16m2(dst_ptr + half_block, _acc01, vl);
            }
        } else {
            for (int j = 0; j < N - 31; j += 32) {
                const __fp16 *s0_ptr = scale + j / 32 * K;
                const int8_t *b0_ptr = sb + j / 2 * K;
                __fp16 *dst_ptr = dst + i * N + j;

                vfloat16m2_t _acc00 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc01 = vfmv_v_f_f16m2(0.0f, vl);
                for (int k = 0; k < K; k++) {
                    vint8m1_t _b0_i8 = vle8_v_i8m1(b0_ptr, vl);
                    vint8m1_t _low_i8 = vand_vx_i8m1(_b0_i8, 0x0f, vl);
                    vint8m1_t _high_i8 = vsra_vx_i8m1(_b0_i8, 4, vl);
                    _high_i8 = vand_vx_i8m1(_high_i8, 0x0f, vl);
                    vint16m2_t _low_i16 = vwsub_vx_i16m2(_low_i8, 8, vl);
                    vint16m2_t _high_i16 = vwsub_vx_i16m2(_high_i8, 8, vl);
                    vfloat16m2_t _low_f16 = vfcvt_f_x_v_f16m2(_low_i16, vl);
                    vfloat16m2_t _high_f16 = vfcvt_f_x_v_f16m2(_high_i16, vl);
                    _low_f16 = vfmul_vf_f16m2(_low_f16, s0_ptr[0], vl);
                    _high_f16 = vfmul_vf_f16m2(_high_f16, s0_ptr[0], vl);
                    _acc00 = vfmacc_vf_f16m2(_acc00, a0_ptr[k], _low_f16, vl);
                    _acc01 = vfmacc_vf_f16m2(_acc01, a0_ptr[k], _high_f16, vl);
                    s0_ptr += 1;
                    b0_ptr += half_block;
                }
                vse16_v_f16m2(dst_ptr, _acc00, vl);
                vse16_v_f16m2(dst_ptr + half_block, _acc01, vl);
            }
        }
    }
}

void shl_c920_gemm_a0nb1_dot_fp16_q4_rearrange(__fp16 *dst, const __fp16 *sa, const int8_t *sb,
                                               __fp16 *bias, int M, int K, int N,
                                               const __fp16 *scale)
{
    // dequantize in gemm
    gemm_dot_1x1_fp16_q4_rearrange(dst, sa, sb, scale, M, K, N);
}