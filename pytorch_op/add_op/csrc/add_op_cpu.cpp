#include "add_op_cpu.h"

#include <iostream>


template <typename T>
void add_op_forward_cpu_kernel(
    const int nthreads,
    const T* input1_data,
    const T* input2_data,
    T* top_data)
{
  for (int n = 0; n < nthreads; ++n)
  {
    top_data[n] = input1_data[n] + input2_data[n];    
  }
}

at::Tensor add_op_forward_cpu(const at::Tensor& input1,
                                const at::Tensor& input2) {

  std::cout << "RUNNING ADD FORWARD ON CPU!\n";

  AT_ASSERTM(!input1.type().is_cuda(), "input must be a CPU tensor");
  AT_ASSERTM(!input2.type().is_cuda(), "rois must be a CPU tensor");

  auto rows = input1.size(0);
  auto cols = input1.size(1);
  // AT_ASSERTM();   

  // auto output = at::empty({rows, cols}, input1.options());
  auto output = at::zeros_like(input1);
  auto output_size = rows * cols;

  if (output.numel() == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input1.type(), "add_op_forward", [&] {
    add_op_forward_cpu_kernel<scalar_t>(
         output_size,
         input1.data<scalar_t>(),
         input2.data<scalar_t>(),
         output.data<scalar_t>());
  });
  return output;
}


at::Tensor add_op_backward_cpu(const at::Tensor& grad)
{
  return grad;
}