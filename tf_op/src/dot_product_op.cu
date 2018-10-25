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

static inline int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

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
        {
            atomicAdd(&tensor_out[r * rows + i], tensor1[r * cols + c] * tensor2[c * rows + i]);
            // tensor_out[r * rows + i] += tensor1[r * cols + c] * tensor2[c * rows + i];
        }
    }
}

template <typename T>
__global__ void MatMulBackwardKernel(const int rows, const int cols, const T *tensor1, const T *tensor2, const T *gradients, T *grad_tensor1, T* grad_tensor2)
{
    int size = rows * cols;

    CUDA_1D_KERNEL_LOOP(tid, size)
    {
        int r = tid / cols;
        int c = tid % cols;
        if (r >= rows)
            return;

        grad_tensor1[r * cols + c] = 0;
        grad_tensor2[c * rows + r] = 0;
        for (int i = 0; i < rows; ++i)
        {
            grad_tensor1[r * cols + c] += tensor2[c * rows + i];
            grad_tensor2[c * rows + r] += tensor1[i * cols + c];
        }
    }
}

void DotProductGPU(int rows, int cols, const float *tensor1, const float *tensor2, float *tensor_out, const GPUDevice &d)
{
    int maxThreadsPerBlock = 1024;
    int size = rows * cols;
    dim3 threadsPerBlock(maxThreadsPerBlock, 1, 1);
    int grid_size = std::min(divUp(size, maxThreadsPerBlock), 2147483647); // checked from deviceQuery    
    dim3 blocksPerGrid(grid_size, 1, 1);

    MatMulKernel<float><<<blocksPerGrid, threadsPerBlock, 0, d.stream()>>>(rows, cols, tensor1, tensor2, tensor_out);
}

void DotProductGradGPU(int rows, int cols, const float *tensor1, const float *tensor2, const float *gradients, 
    float *grad_tensor1, float *grad_tensor2, const GPUDevice &d)
{
    int maxThreadsPerBlock = 1024;
    int size = rows * cols;
    dim3 threadsPerBlock(maxThreadsPerBlock, 1, 1);

    int grid_size = std::min(divUp(size, maxThreadsPerBlock), 2147483647); // checked from deviceQuery
    dim3 blocksPerGrid(grid_size, 1, 1);

    MatMulBackwardKernel<float><<<blocksPerGrid, threadsPerBlock, 0, d.stream()>>>(rows, cols, tensor1, tensor2, gradients, grad_tensor1, grad_tensor2);
}