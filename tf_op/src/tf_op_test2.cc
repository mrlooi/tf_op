#include <iostream>

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

// #include "example.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define PRINT(a) std::cout << #a << ": " << a << std::endl;

static const std::string ATTR_STR = "T";
static const char* OP_NAME = "TfOpTest";

REGISTER_OP(OP_NAME) // Op must be camel case
    .Attr(ATTR_STR + ": {float, int32}")
    .Input("to_zero: " + ATTR_STR)
    .Output("zeroed: " + ATTR_STR);
// .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
//     // c->set_output(0, c->input(0));
//     return Status::OK();
// });

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class ExampleOp : public OpKernel
{
  public:
    explicit ExampleOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));

        // Do the computation.
        PRINT(input_tensor.NumElements())
        // PRINT(tensorflow::kint32max)
        OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));

        int size = static_cast<int>(input_tensor.NumElements()); 
        const T* in = input_tensor.flat<T>().data();
        T* out = output_tensor->flat<T>().data();
        std::cout << "RUNNING OP ON CPU!\n";
        for (int i = 0; i < size; ++i)
        {
            out[i] = 4 * in[i];
        }
    }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                  \
    REGISTER_KERNEL_BUILDER(                                              \
        Name(OP_NAME).Device(DEVICE_CPU).TypeConstraint<T>(ATTR_STR.c_str()),  \
        ExampleOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

void ExampleFunctorGPU(OpKernelContext *context, int size, const float *in, float *out, const GPUDevice &d);
// extern void ExampleFunctorGPU(OpKernelContext *context, int size, const int32 *in, int32 *out, const GPUDevice &d);

template <class T>
class ExampleOp<GPUDevice, T> : public OpKernel
{
  public:
    explicit ExampleOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));

        // Do the computation.
        PRINT(input_tensor.NumElements())
        // PRINT(tensorflow::kint32max)
        OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));

        int size = static_cast<int>(input_tensor.NumElements());
        const T *in = input_tensor.flat<T>().data();
        T *out = output_tensor->flat<T>().data();
        std::cout << "RUNNING OP ON GPU!\n";
        ExampleFunctorGPU(context, size, in, out, context->eigen_device<GPUDevice>());
    }
};

// Register the GPU kernels.
// #define GOOGLE_CUDA
// #ifdef GOOGLE_CUDA

/* Declare explicit instantiations in kernel_example.cu.cc. */
#define REGISTER_GPU(T)                                            \
    REGISTER_KERNEL_BUILDER(                                       \
        Name(OP_NAME).Device(DEVICE_GPU).TypeConstraint<T>(ATTR_STR.c_str()), \
        ExampleOp<GPUDevice, T>);
REGISTER_GPU(float);
// REGISTER_GPU(int32);
// #endif // GOOGLE_CUDA

// REGISTER_KERNEL_BUILDER(Name(OP_NAME).Device(DEVICE_GPU).TypeConstraint<float>(ATTR_STR.c_str()), ExampleOp<GPUDevice, float>);
// REGISTER_KERNEL_BUILDER(Name(OP_NAME).Device(DEVICE_GPU).TypeConstraint<int32>(ATTR_STR.c_str()), ExampleOp<GPUDevice, int32>);
