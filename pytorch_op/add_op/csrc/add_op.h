#pragma once

#include "add_op_cpu.h"

#ifdef WITH_CUDA
#include "add_op_cuda.h"
#endif

// Interface for Python
at::Tensor add_op_forward(const at::Tensor& input1,
                            const at::Tensor& input2) {
  if (input1.type().is_cuda()) {
#ifdef WITH_CUDA
    return add_op_forward_cuda(input1, input2);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return add_op_forward_cpu(input1, input2);
}

at::Tensor add_op_backward(const at::Tensor& grad) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return add_op_backward_cuda(grad);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  // AT_ERROR("Not implemented on the CPU");
  return add_op_backward_cpu(grad);
}
