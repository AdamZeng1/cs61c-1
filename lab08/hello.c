#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("hello world from thread %d\n", thread_id);
    }

    return EXIT_SUCCESS;
}

