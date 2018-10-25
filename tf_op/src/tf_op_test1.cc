#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

using namespace tensorflow;

#define PRINT(a) std::cout << #a << ": " << a << std::endl;

REGISTER_OP("TfOpTest")
    .Input("to_zero: float32")
    .Output("zeroed: float32");
    // .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    //     // c->set_output(0, c->input(0));
    //     return Status::OK();
    // });


class ZeroOutOp : public OpKernel
{
  public:
    explicit ZeroOutOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<float>();

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto output_flat = output_tensor->flat<float>();

        const TensorShape &in_shape = input_tensor.shape();
        PRINT(in_shape.dims())
        PRINT(output_flat.size())
        PRINT(input.size())

        // Set all but the first element of the output tensor to 0.
        const int N = input.size();
        for (int i = 1; i < N; i++)
        {
            output_flat(i) = 0;
        }

        // Preserve the first input value if possible.
        if (N > 0)
            output_flat(0) = input(0);
    }
};

REGISTER_KERNEL_BUILDER(Name("TfOpTest").Device(DEVICE_CPU), ZeroOutOp);
