#ifndef RVV_MATHFUN_FP16_H
#define RVV_MATHFUN_FP16_H

#include <riscv_vector.h>

#define c_exp_hi_f16 10.7421875f
#define c_exp_lo_f16 -10.7421875f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

#define _RVV_FLOAT16_EXP_OP(LMUL, MLEN)                                                        \
    static inline vfloat16m##LMUL##_t exp_ps_vfloat16m##LMUL(vfloat16m##LMUL##_t x, size_t vl) \
    {                                                                                          \
        vfloat16m##LMUL##_t tmp, fx;                                                           \
                                                                                               \
        x = vfmin_vf_f16m##LMUL(x, c_exp_hi_f16, vl);                                          \
        x = vfmax_vf_f16m##LMUL(x, c_exp_lo_f16, vl);                                          \
                                                                                               \
        /* express exp(x) as exp(g + n*log(2)) */                                              \
        fx = vfmacc_vf_f16m##LMUL(vfmv_v_f_f16m##LMUL(0.5f, vl), c_cephes_LOG2EF, x, vl);      \
                                                                                               \
        /* perform a floorf */                                                                 \
        tmp = vfcvt_f_x_v_f16m##LMUL(vfcvt_x_f_v_i16m##LMUL(fx, vl), vl);                      \
                                                                                               \
        /* if greater, substract 1 */                                                          \
        vbool##MLEN##_t mask = vmfgt_vv_f16m##LMUL##_b##MLEN(tmp, fx, vl);                     \
        fx = vfsub_vf_f16m##LMUL##_m(mask, tmp, tmp, 1.f, vl);                                 \
                                                                                               \
        tmp = vfmul_vf_f16m##LMUL(fx, c_cephes_exp_C1, vl);                                    \
        vfloat16m##LMUL##_t z = vfmul_vf_f16m##LMUL(fx, c_cephes_exp_C2, vl);                  \
        x = vfsub_vv_f16m##LMUL(x, tmp, vl);                                                   \
        x = vfsub_vv_f16m##LMUL(x, z, vl);                                                     \
                                                                                               \
        vfloat16m##LMUL##_t y = vfmul_vf_f16m##LMUL(x, c_cephes_exp_p0, vl);                   \
        z = vfmul_vv_f16m##LMUL(x, x, vl);                                                     \
                                                                                               \
        y = vfadd_vf_f16m##LMUL(y, c_cephes_exp_p1, vl);                                       \
        y = vfmul_vv_f16m##LMUL(y, x, vl);                                                     \
        y = vfadd_vf_f16m##LMUL(y, c_cephes_exp_p2, vl);                                       \
        y = vfmul_vv_f16m##LMUL(y, x, vl);                                                     \
        y = vfadd_vf_f16m##LMUL(y, c_cephes_exp_p3, vl);                                       \
        y = vfmul_vv_f16m##LMUL(y, x, vl);                                                     \
        y = vfadd_vf_f16m##LMUL(y, c_cephes_exp_p4, vl);                                       \
        y = vfmul_vv_f16m##LMUL(y, x, vl);                                                     \
        y = vfadd_vf_f16m##LMUL(y, c_cephes_exp_p5, vl);                                       \
                                                                                               \
        y = vfmul_vv_f16m##LMUL(y, z, vl);                                                     \
        y = vfadd_vv_f16m##LMUL(y, x, vl);                                                     \
        y = vfadd_vf_f16m##LMUL(y, 1.f, vl);                                                   \
                                                                                               \
        /* build 2^n */                                                                        \
        vint16m##LMUL##_t mm = vfcvt_x_f_v_i16m##LMUL(fx, vl);                                 \
        mm = vadd_vx_i16m##LMUL(mm, 0xf, vl);                                                  \
        mm = vsll_vx_i16m##LMUL(mm, 10, vl);                                                   \
        vfloat16m##LMUL##_t pow2n = vreinterpret_v_i16m##LMUL##_f16m##LMUL(mm);                \
                                                                                               \
        y = vfmul_vv_f16m##LMUL(y, pow2n, vl);                                                 \
        return y;                                                                              \
    }

_RVV_FLOAT16_EXP_OP(1, 16)
_RVV_FLOAT16_EXP_OP(2, 8)
_RVV_FLOAT16_EXP_OP(4, 4)
_RVV_FLOAT16_EXP_OP(8, 2)

#endif  // RVV_MATHFUN_FP16_H