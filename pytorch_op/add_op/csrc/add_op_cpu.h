#pragma once
#include <torch/extension.h>


at::Tensor add_op_forward_cpu(const at::Tensor& input1,
                                const at::Tensor& input2);

at::Tensor add_op_backward_cpu(const at::Tensor& grad);