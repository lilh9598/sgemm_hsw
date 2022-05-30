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