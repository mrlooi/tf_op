// kernel_example.cu.cc
// #ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
// #include "example.h"

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/util/cuda_kernel_helper.h>

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

// Define the CUDA kernel.
template <typename T>
__global__ void MatMulKernel(const int rows, const int cols, const T *tensor1, const T *tensor2, T *tensor_out)
{
    int size = rows * cols;

    CUDA_1D_KERNEL_LOOP(tid, size)
    {
        int r = tid / cols;
        int c = tid % cols;
        if (r >= rows)
            return;
        for (int i = 0; i < rows; ++i)
            atomicAdd(&tensor_out[r * rows + i], tensor1[r * cols + c] * tensor2[c * rows + i]);
    }
}

void DotProductGPU(int rows, int cols, const float *tensor1, const float *tensor2, float *tensor_out, const GPUDevice &d)
{
    int maxThreadsPerBlock = 1024;
    int size = rows * cols;
    dim3 threadsPerBlock(maxThreadsPerBlock, 1, 1);
    int grid_size = std::min((size + maxThreadsPerBlock - 1) / maxThreadsPerBlock, 2147483647 ); // checked from deviceQuery
    dim3 blocksPerGrid(grid_size, 1, 1);

    MatMulKernel<float><<<blocksPerGrid, threadsPerBlock, 0, d.stream()>>>(rows, cols, tensor1, tensor2, tensor_out);
}
