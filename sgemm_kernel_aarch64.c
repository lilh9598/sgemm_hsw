#include <arm_neon.h>
void kernel_12x8(long k, const float *a, const float *b, float *c, long ldc);

/*
    row-major                   col-major
    c : m x 12                  c : 12 x m
    a : m x k       =========>  a : 12 x k
    b : k x 12                  b : k x m

    c  = a  * b
    cT = bT * aT
*/
void sgemm_kernel_x64_fma_m8n12(float *b_pack, float *a, float *c, int m, int k) {
    for (int im = 0; im < m; im += 8) {
        kernel_12x8(k, a, b_pack + im * k, c + im * 12, 12);
    }
}


void pack_(float *dst, float *src, int m, int k) {
    for (int im = 0; im < m; im += 8) {
        for (int ik = 0; ik < k; ik += 1) {
            for (int mm = 0; mm < 8; mm += 1) {
                dst[im * k + ik * 8 + mm] = src[im * k + mm * k + ik];
            }
        }
    }
}

#define ST_C(x)                           \
    vst1q_f32(&c[x * ldc + 0], vc##x##0); \
    vst1q_f32(&c[x * ldc + 4], vc##x##1); \
    vst1q_f32(&c[x * ldc + 8], vc##x##2);

#define INIT_C(x)                                      \
    float32x4_t vc##x##0 = vld1q_f32(&c[x * ldc + 0]); \
    float32x4_t vc##x##1 = vld1q_f32(&c[x * ldc + 4]); \
    float32x4_t vc##x##2 = vld1q_f32(&c[x * ldc + 8]);

#define UNROLL8(x) \
    x(0);          \
    x(1);          \
    x(2);          \
    x(3);          \
    x(4);          \
    x(5);          \
    x(6);          \
    x(7);

#define FMA(x, vb)                                            \
    do {                                                      \
        vc##x##0 = vfmaq_laneq_f32(vc##x##0, va0, vb, x % 4); \
        vc##x##1 = vfmaq_laneq_f32(vc##x##1, va1, vb, x % 4); \
        vc##x##2 = vfmaq_laneq_f32(vc##x##2, va2, vb, x % 4); \
    } while (0)

inline void kernel_12x8(long k, const float *a, const float *b, float *c, long ldc) {
    UNROLL8(INIT_C);
    for (int p = 0; p < k; p += 1) {
        float32x4_t va0 = vld1q_f32(a);
        float32x4_t va1 = vld1q_f32(a + 4);
        float32x4_t va2 = vld1q_f32(a + 8);
        float32x4_t vb0 = vld1q_f32(b + 0);
        float32x4_t vb1 = vld1q_f32(b + 4);

        FMA(0, vb0);
        FMA(1, vb0);
        FMA(2, vb0);
        FMA(3, vb0);
        FMA(4, vb1);
        FMA(5, vb1);
        FMA(6, vb1);
        FMA(7, vb1);

        a += 12;
        b += 8;
    }
    UNROLL8(ST_C);
}

