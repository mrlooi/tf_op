#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <iostream>

#define PRINT(a) std::cout << #a << ": " << a << std::endl;

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__global__ void add_op_forward_kernel(
    const int nthreads,
    const T* input1_data,
    const T* input2_data,
    T* top_data) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
  	top_data[index] = input1_data[index] + input2_data[index];
  }
}

at::Tensor add_op_forward_cuda(const at::Tensor& input1,
                                 const at::Tensor& input2) 
{

	std::cout << "RUNNING ADD FORWARD ON GPGPU!\n";

  AT_ASSERTM(input1.type().is_cuda(), "input1 must be a CUDA tensor");
  AT_ASSERTM(input2.type().is_cuda(), "input2 must be a CUDA tensor");

  auto rows = input1.size(0);
  auto cols = input1.size(1);
  // AT_ASSERTM();   

  // auto output = at::empty({rows, cols}, input1.options());
  auto output = at::zeros_like(input1);
  auto output_size = rows * cols;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input1.type(), "add_op_forward", [&] {
    add_op_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input1.contiguous().data<scalar_t>(),
         input2.contiguous().data<scalar_t>(),
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}

at::Tensor add_op_backward_cuda(const at::Tensor& grad)
{
	// PRINT(grad.size(0))
	// PRINT(grad.size(1))
	// auto output = at::zeros_like(grad);

  return grad;
}