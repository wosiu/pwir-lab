#include <stdio.h>
#include <time.h>

// We assume that NUM_ELEMENTS is divisible by BLOCK_SIZE
#define RADIUS        3
#define BLOCK_SIZE    256
//#define NUM_ELEMENTS  (4096*2)*20
#define NUM_ELEMENTS 1e7

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

__global__ void stencil_1d(int *in, int *out) 
{
	// blockDim is 3-dimensional vector storing block grid dimensions

	// index of a thread across all threads + RADIUS
    int gindex = threadIdx.x + (blockIdx.x * blockDim.x) + RADIUS;
    
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
        result += in[gindex + offset];

    // Store the result
    out[gindex - RADIUS] = result;
}

int main()
{
  unsigned int i;

  // vectors stored in the CPU memory - can be used from host code only
  // int h_in[NUM_ELEMENTS + 2 * RADIUS], h_out[NUM_ELEMENTS];
  //size_t size = sizeof(int) * (NUM_ELEMENTS + 2 * RADIUS );
  int *h_in = (int*)malloc( sizeof(int) * (NUM_ELEMENTS + 2 * RADIUS ) );
  int *h_out = (int*)malloc( sizeof(int) * NUM_ELEMENTS );

  // vectors that will be stored in the device memory - can be dereferenced
  // only in kernel code
  int *d_in, *d_out;

  // Initialize host data
  for( i = 0; i < (NUM_ELEMENTS + 2*RADIUS); ++i )
    h_in[i] = 1; // With a value of 1 and RADIUS of 3, all output values should be 7

  // Allocate space on the device
  // cudaMalloc is equivalent of malloc
  cudaCheck( cudaMalloc( &d_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int)) );
  cudaCheck( cudaMalloc( &d_out, NUM_ELEMENTS * sizeof(int)) );

  // Copy input data to device
  cudaCheck( cudaMemcpy( d_in, h_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int), 
  	cudaMemcpyHostToDevice) );

  // Call kernels
  clock_t start = clock(); 
  stencil_1d<<< (NUM_ELEMENTS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE >>> (d_in, d_out);
  printf("CZAS: %d", clock() - start);

  // Check errors from launching the kernel
  cudaCheck(cudaPeekAtLastError());

  // Copy results from device memory to host
  cudaCheck( cudaMemcpy( h_out, d_out, NUM_ELEMENTS * sizeof(int), 
  	cudaMemcpyDeviceToHost) );

  // Verify every out value is 7
  for( i = 0; i < NUM_ELEMENTS; ++i )
    if (h_out[i] != 7)
    {
      printf("Element h_out[%d] == %d != 7\n", i, h_out[i]);
      break;
    }

  if (i == NUM_ELEMENTS)
    printf("SUCCESS!\n");

  free(h_in);
  free(h_out);
  // Free out memory
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}

