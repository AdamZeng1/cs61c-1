/*
 * Project 2: Performance Optimization
 */

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "calc_depth_naive.h"
#include "calc_depth_optimized.h"
#include "utils.h"

static float distance(const float *left, const float *right, int height, int width,
                      int left_x, int left_y, int right_x, int right_y,
                      int box_height, int box_width);

inline static float displacement(int dx, int dy) {
    return sqrt(dx * dx + dy * dy);
}

inline static long square_sum (int x, int y) {
    return (long) x * x + (long) y * y;
}

inline static bool out_of_bound (int x, int y, int box_width, int box_height,
                                 int width, int height) {
    return x < 0 || x + box_width > width || y < 0 || y + box_height > height;
}

void calc_depth_optimized(float *depth, float *left, float *right,
                          int image_width, int image_height, int feature_width,
                          int feature_height, int maximum_displacement) {
    const int box_width = (feature_width << 1) + 1;
    const int box_height = (feature_height << 1) + 1;

    # pragma omp parallel for collapse(1)
    for (int y = 0; y < image_height; y++) {
        for (int x = 0; x < image_width; x++) {
            int left_box_x = x - feature_width;
            int left_box_y = y - feature_width;
            if (out_of_bound(left_box_x, left_box_y, box_width, box_height,
                             image_width, image_height)) {
                depth[y * image_width + x] = 0.0f;
                continue;
            }

            float min_diff = -1.0f;
            int min_dy = 0;
            int min_dx = 0;
            for (int dy = -maximum_displacement; dy <= maximum_displacement; dy++) {
                for (int dx = -maximum_displacement; dx <= maximum_displacement; dx++) {
                    int right_box_x = left_box_x + dx;
                    int right_box_y = left_box_y + dy;

                    if (out_of_bound(right_box_x, right_box_y, box_width, box_height,
                                     image_width, image_height)) {
                        continue;
                    }

                    float squared_diff = distance(left, right, image_width, image_height,
                        left_box_x, left_box_y, right_box_x, right_box_y, box_width, box_height);

                    if (min_diff == -1.0f || min_diff > squared_diff
                            || (min_diff == squared_diff
                                && square_sum(dx, dy) < square_sum(min_dx, min_dy))) {
                        min_diff = squared_diff;
                        min_dx = dx;
                        min_dy = dy;
                    }
                }
            }
            if (min_diff != -1.0f) {
                if (maximum_displacement == 0) {
                    depth[y * image_width + x] = 0.0f;
                } else {
                    depth[y * image_width + x] = displacement(min_dx, min_dy);
                }
            } else {
                depth[y * image_width + x] = 0.0f;
            }
        }
    }
}

#define BLOCK_SIZE (128 / sizeof(float) / CHAR_BIT)

float distance(const float *left, const float *right, int width, int height,
               int left_x, int left_y, int right_x, int right_y,
               int box_width, int box_height) {
    float dist = 0.0f;
    float partial_distance[BLOCK_SIZE];
    __m128 partial = _mm_set1_ps(0.0f);
    const int num_blocks = box_width / BLOCK_SIZE;

    for (int i = 0; i < box_height; ++i) {
        const float *left_x_start = left + (left_y + i) * width + left_x;
        const float *right_x_start = right + (right_y + i) * width + right_x;

        for (int block = 0; block < num_blocks; ++block) {
            __m128 l = _mm_loadu_ps(left_x_start + block * BLOCK_SIZE);
            __m128 r = _mm_loadu_ps(right_x_start + block * BLOCK_SIZE);
            __m128 diff = _mm_sub_ps(l, r);
            partial = _mm_add_ps(partial, _mm_mul_ps(diff, diff));
        }

        for (int j = num_blocks * BLOCK_SIZE; j < box_width; ++j) {
            float diff = *(left_x_start + j) - *(right_x_start + j);
            dist += diff * diff;
        }
    }

    _mm_storeu_ps(partial_distance, partial);
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        dist += partial_distance[i];
    }

    return dist;
}
