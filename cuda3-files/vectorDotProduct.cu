/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024
#define VECTOR_DIM 1024*1024*32

// CUDA API error checking macro
static void handleError(cudaError_t err,
                        const char *file,
                        int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line );
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))

/**
 * Vector dot product 
  */
__global__ void dotProductCuda(float *C, float *A, float *B)
{
	
	__shared__ float sums[BLOCK_SIZE];
	int lid = threadIdx.x;
	int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	sums[lid] = A[gid] * B[gid];
	__syncthreads();	

        // reduction
	int range = 1;

        while (range < BLOCK_SIZE) {
                // todo zmienic modulo na sprawdzanie bitu
                if (lid % (2*range) == 0) {
                        sums[lid] = sums[lid] + sums[lid + range];
                }
                range *= 2;
                __syncthreads();
        }

        if ( threadIdx.x == 0 ) {
                C[blockIdx.x] = sums[0];
        }
}


__device__ void doReductionCuda(float *A, float *B)
{
	
	__shared__ float sums[BLOCK_SIZE];
        int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        int lid = threadIdx.x;
	int range = 1;
	
	sums[lid] = A[gid];
	syncthreads();
	

	while (range < BLOCK_SIZE) {
		// todo zmienic modulo na sprawdzanie bitu
		if (lid % (2*range) == 0) {
			sums[lid] = sums[lid] + sums[lid + range];
		}
		range *= 2;
		__syncthreads();
	}
	
	if ( threadIdx.x == 0 ) {
		B[blockIdx.x] = sums[0];
	}
}



void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = ((double)rand()/(double)RAND_MAX);
    }
}

double computeVectorDotProductCpu(float* A, float* B) {
  double sum = 0.0;
  for (int i = 0; i < VECTOR_DIM; i++) {
    sum += A[i] * B[i];
  }
  return sum;
}

/**
 * Run a simple test of vector dot product using CUDA
 */
int main()
{
    // Allocate host memory for vectors A, B and C
    size_t vector_mem_size = sizeof(float) * VECTOR_DIM;
    // TODO check if OK
    size_t product_block_num = (VECTOR_DIM + BLOCK_SIZE - 1)/BLOCK_SIZE;
    size_t result_mem_size = sizeof(float) * product_block_num;

    float *h_A = (float *)malloc(vector_mem_size);
    float *h_B = (float *)malloc(vector_mem_size);
    float *h_C = (float *)malloc(result_mem_size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host memory
    randomInit(h_A, VECTOR_DIM);
    randomInit(h_B, VECTOR_DIM);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    cudaCheck(cudaMalloc((void **) &d_A, vector_mem_size));
    cudaCheck(cudaMalloc((void **) &d_B, vector_mem_size));
    cudaCheck(cudaMalloc((void **) &d_C, result_mem_size));

    // copy host memory to device
    cudaCheck(cudaMemcpy(d_A, h_A, vector_mem_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, vector_mem_size, cudaMemcpyHostToDevice));

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Perform 1st phase of a dot product 
    dotProductCuda<<<product_block_num, BLOCK_SIZE>>>(d_C, d_A, d_B);
    
    cudaCheck(cudaPeekAtLastError());
    
    double result = 0;
   
    cudaCheck(cudaMemcpy(h_C, d_C, result_mem_size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < product_block_num; i++) {
	result += h_C[i];
    }


    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps

    double eps = 1.e-1; 

    double expected = computeVectorDotProductCpu(h_A, h_B);

    if (abs(result - expected) > eps) {
      printf("ERROR: %f != %f\n", result, expected);
      correct = false;
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
