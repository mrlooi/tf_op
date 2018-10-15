// kernel_example.cu.cc
// #ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
// #include "example.h"

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/util/cuda_kernel_helper.h>

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


// Define the CUDA kernel.
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T *in, T *out)
{
    CUDA_1D_KERNEL_LOOP(i, size)
    {
        out[i] = 3 * in[i];
    }
}

// Define the GPU implementation that launches the CUDA kernel.
// template <typename T>
// void ExampleFunctor<GPUDevice, T>::operator()(
//     const GPUDevice &d, int size, const T *in, T *out)
// {
//     // Launch the cuda kernel.
//     //
//     // See core/util/cuda_kernel_helper.h for example of computing
//     // block count and thread_per_block count.
//     int block_count = 1024;
//     int thread_per_block = 20;
//     ExampleCudaKernel<T>
//         <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
// }

// template <typename T>
typedef float T;
void ExampleFunctorGPU(OpKernelContext *context, int size, const T *in, T *out, const GPUDevice &d)
{
    int maxThreadsPerBlock = 1024;
    dim3 threadsPerBlock(maxThreadsPerBlock, 1, 1);
    dim3 blocksPerGrid((size + maxThreadsPerBlock - 1) / maxThreadsPerBlock, 1, 1);

    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    ExampleCudaKernel<T><<<blocksPerGrid, threadsPerBlock, 0, d.stream()>>>(size, in, out);
}


// // // Explicitly instantiate functors for the types of OpKernels registered.
// template struct ExampleFunctor<GPUDevice, float>;
// template struct ExampleFunctor<GPUDevice, int32>;

// #endif // GOOGLE_CUDA