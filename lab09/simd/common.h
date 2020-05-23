#ifndef COMMON_H
#define COMMON_H

#include <x86intrin.h>

#define NUM_ELEMS ((1 << 16) + 10)
#define OUTER_ITERATIONS (1 << 16)

long long int sum(unsigned int vals[NUM_ELEMS]) {
	clock_t start = clock();

	long long int sum = 0;
	for(unsigned int w = 0; w < OUTER_ITERATIONS; w++) {
		for(unsigned int i = 0; i < NUM_ELEMS; i++) {
			if(vals[i] >= 128) {
				sum += vals[i];
			}
		}
	}
	clock_t end = clock();
	printf("Time taken: %Lf s\n", (long double)(end - start) / CLOCKS_PER_SEC);
	return sum;
}

long long int sum_unrolled(unsigned int vals[NUM_ELEMS]) {
	clock_t start = clock();
	long long int sum = 0;

	for(unsigned int w = 0; w < OUTER_ITERATIONS; w++) { 
		for(unsigned int i = 0; i < NUM_ELEMS / 4 * 4; i += 4) {
			if(vals[i] >= 128) sum += vals[i];
			if(vals[i + 1] >= 128) sum += vals[i + 1];
			if(vals[i + 2] >= 128) sum += vals[i + 2];
			if(vals[i + 3] >= 128) sum += vals[i + 3];
		}

		//This is what we call the TAIL CASE
		//For when NUM_ELEMS isn't a multiple of 4
		//NONTRIVIAL FACT: NUM_ELEMS / 4 * 4 is the largest multiple of 4 less than NUM_ELEMS
		for(unsigned int i = NUM_ELEMS / 4 * 4; i < NUM_ELEMS; i++) {
			if (vals[i] >= 128) {
				sum += vals[i];
			}
		}
	}
	clock_t end = clock();
	printf("Time taken: %Lf s\n", (long double)(end - start) / CLOCKS_PER_SEC);
	return sum;
}

long long int sum_simd(unsigned int vals[NUM_ELEMS]) {
	clock_t start = clock();
	__m128i _127 = _mm_set1_epi32(127);		// This is a vector with 127s in it... Why might you need this?
	long long int result = 0;				// This is where you should put your final result!
											// DO NOT DO NOT DO NOT DO NOT WRITE ANYTHING ABOVE THIS LINE.
	for(unsigned int w = 0; w < OUTER_ITERATIONS; w++) {
		/* YOUR CODE GOES HERE */
		int sums[4];
		__m128i partial = _mm_set1_epi32(0);
		const unsigned int block_size = 4;
		unsigned int num_blocks = NUM_ELEMS / block_size;

		for (unsigned int block = 0; block < num_blocks; ++block) {
			unsigned int *block_start = vals + (block << 2);
			__m128i data = _mm_loadu_si128((__m128i *) block_start);
			__m128i mark = _mm_cmpgt_epi32(data, _127);
			partial = _mm_add_epi32(partial, _mm_and_si128(data, mark));
		}

		_mm_storeu_si128((__m128i *) sums, partial);
		for (unsigned int i = 0; i < 4; ++i) {
			result += sums[i];
		}

		for (unsigned int i = (NUM_ELEMS >> 2) << 2; i < NUM_ELEMS; ++i) {
			unsigned int num = vals[i];
			if (num > 127) {
				result += num;
			}
		}
		/* You'll need a tail case. */

	}
	clock_t end = clock();
	printf("Time taken: %Lf s\n", (long double)(end - start) / CLOCKS_PER_SEC);
	return result;
}

long long int sum_simd_unrolled(unsigned int vals[NUM_ELEMS]) {
	clock_t start = clock();
	__m128i _127 = _mm_set1_epi32(127);
	long long int result = 0;
	for(unsigned int w = 0; w < OUTER_ITERATIONS; w++) {
		/* COPY AND PASTE YOUR sum_simd() HERE */
		/* MODIFY IT BY UNROLLING IT */
		const unsigned int BLOCK_SIZE = 32;
		int sums[4];
		__m128i partial = _mm_set1_epi32(0);
		unsigned int num_blocks = NUM_ELEMS / BLOCK_SIZE;

		for (unsigned block = 0; block < num_blocks; ++block) {
			unsigned int *block_start = vals + block * BLOCK_SIZE;
			__m128i sub_block1 = _mm_loadu_si128((__m128i *) block_start);
			__m128i mark1 = _mm_cmpgt_epi32(sub_block1, _127);
			__m128i sub_block2 = _mm_loadu_si128((__m128i *) (block_start + 4));
			__m128i mark2 = _mm_cmpgt_epi32(sub_block2, _127);
			__m128i sub_block3 = _mm_loadu_si128((__m128i *) (block_start + 8));
			__m128i mark3 = _mm_cmpgt_epi32(sub_block3, _127);
			__m128i sub_block4 = _mm_loadu_si128((__m128i *) (block_start + 12));
			__m128i mark4 = _mm_cmpgt_epi32(sub_block4, _127);
			__m128i sub_block5 = _mm_loadu_si128((__m128i *) (block_start + 16));
			__m128i mark5 = _mm_cmpgt_epi32(sub_block5, _127);
			__m128i sub_block6 = _mm_loadu_si128((__m128i *) (block_start + 20));
			__m128i mark6 = _mm_cmpgt_epi32(sub_block6, _127);
			__m128i sub_block7 = _mm_loadu_si128((__m128i *) (block_start + 24));
			__m128i mark7 = _mm_cmpgt_epi32(sub_block7, _127);
			__m128i sub_block8 = _mm_loadu_si128((__m128i *) (block_start + 28));
			__m128i mark8 = _mm_cmpgt_epi32(sub_block8, _127);

			partial = _mm_add_epi32(partial, _mm_and_si128(sub_block1, mark1));
			partial = _mm_add_epi32(partial, _mm_and_si128(sub_block2, mark2));
			partial = _mm_add_epi32(partial, _mm_and_si128(sub_block3, mark3));
			partial = _mm_add_epi32(partial, _mm_and_si128(sub_block4, mark4));
			partial = _mm_add_epi32(partial, _mm_and_si128(sub_block5, mark5));
			partial = _mm_add_epi32(partial, _mm_and_si128(sub_block6, mark6));
			partial = _mm_add_epi32(partial, _mm_and_si128(sub_block7, mark7));
			partial = _mm_add_epi32(partial, _mm_and_si128(sub_block8, mark8));
		}

		_mm_storeu_si128((__m128i *) sums, partial);
		for (unsigned int i = 0; i < 4; ++i) {
			result += sums[i];
		}

		for (unsigned int start = NUM_ELEMS / BLOCK_SIZE * BLOCK_SIZE; start < NUM_ELEMS; ++start) {
			unsigned int num = vals[start];
			if (num > 127) {
				result += vals[start];
			}
		}
		/* You'll need 1 or maybe 2 tail cases here. */

	}
	clock_t end = clock();
	printf("Time taken: %Lf s\n", (long double)(end - start) / CLOCKS_PER_SEC);
	return result;
}

/* DON'T TOUCH THIS FUNCTION */
int int_comparator(const void* a, const void* b) {
	if(*(unsigned int*)a == *(unsigned int*)b) return 0;
	else if(*(unsigned int*)a < *(unsigned int*)b) return -1;
	else return 1;
}

#endif
