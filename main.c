#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#define ABS(x, y) ((x) > (y) ? (x) - (y) : (y) - (x))

#ifdef __cplusplus
extern "C" {
#endif

void sgemm_kernel_x64_fma_m8n12(float *a,
    float *b,
    float *c,
    int m,
    int k);

void pack_(float *dst,
    float *src,
    int m,
    int k);

#ifdef __cplusplus
}
#endif

static double get_time(struct timespec *start, struct timespec *end)
{
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

static void thread_bind(int cpu)
{
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu, &cpu_set);
    if (pthread_setaffinity_np(pthread_self(),
            sizeof(cpu_set_t), &cpu_set) != 0)
    {
        fprintf(stderr, "Error: cpu[%d] bind failed.\n", cpu);
        exit(0);
    }
}

static void *page_alloc(size_t size)
{
    void *data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    if (data == (void*)-1)
    {
        fprintf(stderr, "Error(MemData::Construction): mmap failed.\n");
        exit(0);
    }
    return data;
}

static void page_free(void *mem, size_t size)
{
    munmap(mem, size);
}

void sgemm_naive(float *a, float *b, float *c, int m, int n, int k)
{
    int i, j, kk;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (kk = 0; kk < k; kk++)
            {
                c[i * n + j] += a[i * k + kk] * b[kk * n + j];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int i;

    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s m k\n", argv[0]);
        return 0;
    }

    int m = atoi(argv[1]);
    if (m % 8 != 0) {
        printf("m must be a multiple of 8.");
        return -1;
    }
    int k = atoi(argv[2]);
    long comp = 2L * m * k * 12L;
    int loop_time = (int)(2e11 / comp);
    printf("sgemm_kernel loop_time : %d, m=%d n=%d k=%d\n", loop_time, m, 12, k);

    struct timespec start, end;
    double t, gflops;

    thread_bind(0);

    float *a = (float*)page_alloc(m * k * sizeof(float));
    float *a_pack = (float*)page_alloc(m * k * sizeof(float));
    float *b = (float*)page_alloc(k * 12 * sizeof(float));
    float *c1 = (float*)page_alloc(m * 12 * sizeof(float));
    float *c2 = (float*)page_alloc(m * 12 * sizeof(float));

    srand(time(NULL));

    for (i = 0; i < m * k; i++)
    {
        a[i] = (float)rand() / (float)RAND_MAX;
    }
    for (i = 0; i < k * 12; i++)
    {
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    pack_(a_pack, a, m, k);
    // fma-tuned version
    // warm up
    for (i = 0; i < loop_time; i++)
    {
        sgemm_kernel_x64_fma_m8n12(a_pack, b, c2, m, k);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (i = 0; i < loop_time; i++)
    {
        sgemm_kernel_x64_fma_m8n12(a_pack, b, c2, m, k);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    t = get_time(&start, &end) / loop_time;
    gflops = (double)comp / t * 1e-9;

    printf("sgemm_kernel_x64_fma(%d, %d, %d): time = %lf us, perf = %lf GFLOPS.\n", m, 12, k, t * 1e6, gflops);

    memset(c1, 0, m * 12 * sizeof(float));
    memset(c2, 0, m * 12 * sizeof(float));
    sgemm_naive(a, b, c1, m, 12, k);
    sgemm_kernel_x64_fma_m8n12(a_pack, b, c2, m, k);

    int check_pack = 1;
    double v = 0;
    int ind = 0;
    for (int i = 0; i < m * 12; i++) {
        v = ABS(c1[i], c2[i]);
        if (v > 1e-4) {
            ind = i;
            check_pack = 0;
            break;
        }
    }

    if (check_pack != 1) {
        printf("Check fail. at(%d) v=%lf\n", ind, v);
    } else {
        printf("Check pass.\n");
    }

    page_free(a, m * k * sizeof(float));
    page_free(a_pack, m * k * sizeof(float));
    page_free(b, k * 12 * sizeof(float));
    page_free(c1, m * 12 * sizeof(float));
    page_free(c2, m * 12 * sizeof(float));

    return 0;
}

