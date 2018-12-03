#include <stdio.h>
#include <vector>

#include "add_op_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void AddForward(const int N, const float* input1_data, const float* input2_data, float* output_data)
{
    CUDA_1D_KERNEL_LOOP(index, N)
    {
    	output_data[index] = input1_data[index] + input2_data[index];
    }
}

int AddForwardLauncher(const int N, const float* input1_data, const float* input2_data, float* output_data, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    cudaError_t err;

    printf("N: %d\n", N);

    AddForward<<<(N + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      	N, input1_data, input2_data, output_data);

    // std::vector<float> v(N);
    // cudaMemcpy(&v[0], input1_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; ++i)
    // {
    // 	printf("%d) %.2f\n", i, v[i]);
    // }

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

