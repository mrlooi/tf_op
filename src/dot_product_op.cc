#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

static const char* OP_NAME = "DotProductOp";

REGISTER_OP(OP_NAME)
    .Input("input: float")
    .Input("weights: float")
    .Output("inner_product: float");

void check_dot_product_op_input(OpKernelContext *context)
{
    // some checks to be sure ...
    // DCHECK_EQ(2, context->num_inputs());
    OP_REQUIRES(context, context->num_inputs() == 2,
                errors::InvalidArgument("context must have 2 inputs only"));

    // get the input tensor
    const Tensor &input1 = context->input(0);
    const Tensor &input2 = context->input(1);

    // check shapes of input
    const TensorShape &input1_shape = input1.shape();
    const TensorShape &input2_shape = input2.shape();

    // check input dims
    DCHECK_EQ(input1_shape.dims(), 2);
    DCHECK_EQ(input2_shape.dims(), 2);
    DCHECK_EQ(input1_shape.dim_size(0), input2_shape.dim_size(1));
    DCHECK_EQ(input1_shape.dim_size(1), input2_shape.dim_size(0));
}

template <typename Device, typename T>
class DotProductOp : public OpKernel
{
  public:
    /// \brief Constructor.
    /// \param context
    explicit DotProductOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }

    /// \brief Compute the inner product.
    /// \param context
    void Compute(OpKernelContext *context) override
    {
        check_dot_product_op_input(context);

        // get the input tensor
        const Tensor &input1 = context->input(0);
        const Tensor &input2 = context->input(1);

        // check shapes of input
        const TensorShape &input1_shape = input1.shape();
        const TensorShape &input2_shape = input2.shape();

        // create output shape
        int rows = input1_shape.dim_size(0);
        int cols = input1_shape.dim_size(1);
        TensorShape output_shape({rows, rows});

        // create output tensor
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        // get the corresponding Eigen tensors for data access
        auto input1_tensor = input1.matrix<T>();
        auto input2_tensor = input2.matrix<T>();
        auto output_tensor = output->matrix<T>();

#pragma omp parallel for
        for (int r = 0; r < rows; ++r)
        {
            for (int i = 0; i < rows; ++i) // since rows = input2.cols
            {
                T tmp = 0;
                for (int c = 0; c < cols; ++c)
                {
                    tmp += input1_tensor(r, c) * input2_tensor(c, i);
                }
                output_tensor(r, i) = tmp;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name(OP_NAME).Device(DEVICE_CPU), DotProductOp<CPUDevice, float>);


void DotProductGPU(int rows, int cols, const float *tensor1, const float *tensor2, float *tensor_out, const GPUDevice &d);

template <class T>
class DotProductOp<GPUDevice, T> : public OpKernel
{
  public:
    /// \brief Constructor.
    /// \param context
    explicit DotProductOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }

    /// \brief Compute the inner product.
    /// \param context
    void Compute(OpKernelContext *context) override
    {
        check_dot_product_op_input(context);

        // get the input tensor
        const Tensor &input1 = context->input(0);
        const Tensor &input2 = context->input(1);

        // check shapes of input
        const TensorShape &input1_shape = input1.shape();
        const TensorShape &input2_shape = input2.shape();

        // create output shape
        int rows = input1_shape.dim_size(0);
        int cols = input1_shape.dim_size(1);
        TensorShape output_shape({rows, rows});

        // create output tensor
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        std::cout << "RUNNING OP ON GPU!\n";
        DotProductGPU(rows, cols, input1.flat<T>().data(), input2.flat<T>().data(), output->flat<T>().data(), context->eigen_device<GPUDevice>());
    }
};

REGISTER_KERNEL_BUILDER(Name(OP_NAME).Device(DEVICE_GPU), DotProductOp<GPUDevice, float>);