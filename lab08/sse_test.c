#include <emmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2
#define NANO_PER_SEC 10e9;

static void mul_sse(const double *A, const double *B, double *C) {
    for (int r = 0; r < N; ++r) {
        __m128d b1 = _mm_load_pd(B);
        __m128d b2 = _mm_load_pd(B + N);
        __m128d ax1 = _mm_set1_pd(A[r * N]);
        __m128d ax2 = _mm_set1_pd(A[r * N + 1]);
        __m128d sum = _mm_add_pd(_mm_mul_pd(ax1, b1), _mm_mul_pd(ax2, b2));
        __m128d cx = _mm_load_pd(C + r * N);
        cx = _mm_add_pd(sum, cx);
        _mm_store_pd(C + r * N, cx);
    }
}

static void mul(const double *A, const double *B, double *C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

static void print_matrix(const double *C) {
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            printf("%f ", C[r * N + c]);
        }
        printf("\n");
    }
}

static double diff_timespec(struct timespec *end, struct timespec *begin) {
    double seconds = (double) end->tv_sec - begin->tv_sec;
    long nano_seconds = end->tv_nsec - begin->tv_nsec;
    return seconds + (double) nano_seconds / NANO_PER_SEC;
}

int main(int argc, char **argv) {
    double A[N * N] = {
        1.0, 2.0,
        3.0, 4.0
    };
    double B[N * N] = {
        3.0, 6.0,
        9.0, 2.0
    };
    double C1[N * N] = {
        0.0, 0.0,
        0.0, 0.0
    };
    double C2[N *N] = {
        0.0, 0.0,
        0.0, 0.0
    };
    long times = atoi(argv[1]);

    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_REALTIME, &start);
    for (long i = 0; i < times; ++i) {
        mul(A, B, C1);
    }
    clock_gettime(CLOCK_REALTIME, &end);
    double mul_time = diff_timespec(&end, &start);
    printf("Calling mul %ld times costs %.9f seconds\n", times, mul_time);
    print_matrix(C1);

    clock_gettime(CLOCK_REALTIME, &start);
    for (long i = 0; i < times; ++i) {
        mul_sse(A, B, C2);
    }
    clock_gettime(CLOCK_REALTIME, &end);
    double mul_sse_time = diff_timespec(&end, &start);
    printf("Calling mul_sse %ld times costs %.9f seconds\n", times, mul_sse_time);
    print_matrix(C2);

    printf("Using sse2 gives %.2f%% improvement\n", 100.0 * (mul_time - mul_sse_time) / mul_sse_time);
    return EXIT_SUCCESS;
}

