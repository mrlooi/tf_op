#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

static const char *OP_NAME = "DotProductGradOp";

REGISTER_OP(OP_NAME)
    .Input("grad: float32")
    .Input("input1: float32")
    .Input("input2: float32")
    .Output("grad_input1: float32")
    .Output("grad_input2: float32");

void check_dot_product_grad_op_input(OpKernelContext *context)
{
    // output and grad is provided as input
    DCHECK_EQ(3, context->num_inputs());

    // get the gradient tensor
    const Tensor &grad = context->input(0);

    // get the original input tensor
    const Tensor &input1 = context->input(1);
    const Tensor &input2 = context->input(2);

    // create input shape (inferred from the additional attribute `n`)
    const TensorShape& input1_shape = input1.shape();
    const TensorShape& input2_shape = input2.shape();
    const TensorShape& grad_shape = grad.shape();

    DCHECK_EQ(input1_shape.dims(), 2);
    DCHECK_EQ(input2_shape.dims(), 2);
    DCHECK_EQ(grad_shape.dims(), 2);
    DCHECK_EQ(grad_shape.dim_size(0), grad_shape.dim_size(1));
    DCHECK_EQ(input1_shape.dim_size(0), input2_shape.dim_size(1));
    DCHECK_EQ(input1_shape.dim_size(1), input2_shape.dim_size(0));
    DCHECK_EQ(input1_shape.dim_size(0), grad_shape.dim_size(0));
}

// compute gradient
template <class Device, class T>
class DotProductGradOp : public OpKernel
{
  public:
    explicit DotProductGradOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }

    void Compute(OpKernelContext *context) override
    {
        check_dot_product_grad_op_input(context);

        // get the gradient tensor
        const Tensor &grad = context->input(0);

        // get the original input tensor
        const Tensor &input1 = context->input(1);
        const Tensor &input2 = context->input(2);

        // create input shape (inferred from the additional attribute `n`)
        const TensorShape &input1_shape = input1.shape();
        const TensorShape &input2_shape = input2.shape();
        const TensorShape &grad_shape = grad.shape();

        // create output tensors
        Tensor *grad_input1 = NULL;
        Tensor *grad_input2 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input1_shape, &grad_input1));
        OP_REQUIRES_OK(context, context->allocate_output(1, input2_shape, &grad_input2));

        // get the Eigen tensors for data access
        auto grad_tensor = grad.matrix<T>();
        auto input1_tensor = input1.matrix<T>();
        auto input2_tensor = input2.matrix<T>();
        auto grad_input1_tensor = grad_input1->matrix<T>();
        auto grad_input2_tensor = grad_input2->matrix<T>();

        int rows = input1_shape.dim_size(0);
        int cols = input1_shape.dim_size(1);

        std::cout << "RUNNING GRAD OP ON CPU!\n";

        #pragma omp parallel for
        for (int c = 0; c < cols; ++c)
        {
            for (int r = 0; r < rows; ++r)
            {
                for (int i = 0; i < rows; ++i)
                {
                    grad_input1_tensor(r, c) += input2_tensor(c, i);
                    grad_input2_tensor(c, r) += input1_tensor(i, c);
                }
            }
        }
    }

  private:
};

REGISTER_KERNEL_BUILDER(Name(OP_NAME).Device(DEVICE_CPU), DotProductGradOp<CPUDevice, float>);

void DotProductGradGPU(int rows, int cols, const float *tensor1, const float *tensor2, const float *gradients,
                       float *grad_tensor1, float *grad_tensor2, const GPUDevice &d);

template <class T>
class DotProductGradOp<GPUDevice, T> : public OpKernel
{
  public:
    explicit DotProductGradOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }

    void Compute(OpKernelContext *context) override
    {
        check_dot_product_grad_op_input(context);

        // get the gradient tensor
        const Tensor &grad = context->input(0);

        // get the original input tensor
        const Tensor &input1 = context->input(1);
        const Tensor &input2 = context->input(2);

        // create input shape (inferred from the additional attribute `n`)
        const TensorShape &input1_shape = input1.shape();
        const TensorShape &input2_shape = input2.shape();
        const TensorShape &grad_shape = grad.shape();

        // create output tensors
        Tensor *grad_input1 = NULL;
        Tensor *grad_input2 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input1_shape, &grad_input1));
        OP_REQUIRES_OK(context, context->allocate_output(1, input2_shape, &grad_input2));

        int rows = input1_shape.dim_size(0);
        int cols = input1_shape.dim_size(1);

        std::cout << "RUNNING GRAD OP ON GPU!\n";
        DotProductGradGPU(rows, cols, input1.flat<T>().data(), input2.flat<T>().data(), grad.flat<T>().data(),
                          grad_input1->flat<T>().data(), grad_input2->flat<T>().data(), context->eigen_device<GPUDevice>());
    }
};

REGISTER_KERNEL_BUILDER(Name(OP_NAME).Device(DEVICE_GPU), DotProductGradOp<GPUDevice, float>);
