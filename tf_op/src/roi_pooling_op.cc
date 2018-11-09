// #include <stdio.h>
#include <cfloat>
#include <iostream>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

#define PRINT(a) std::cout << #a << ": " << a << std::endl;


REGISTER_OP("RoiPool")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Attr("pool_channel: int")
    .Input("bottom_data: T")
    .Input("bottom_rois: T");
    // .Output("top_data: T")
    // .Output("argmax: int32");


template <typename Device, typename T>
class RoiPoolOp : public OpKernel {
 public:
  explicit RoiPoolOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));

    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));

    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));

    // Get the pool channel flag
    OP_REQUIRES_OK(context,
                   context->GetAttr("pool_channel", &pool_channel_));

    PRINT(pooled_height_)
    PRINT(pooled_width_)
    PRINT(spatial_scale_)
    PRINT(pool_channel_)
  }

  void Compute(OpKernelContext* context) override 
  {
    // // Grab the input tensor
    // const Tensor& bottom_data = context->input(0);
    // const Tensor& bottom_rois = context->input(1);
    // auto bottom_data_flat = bottom_data.flat<T>();
    // auto bottom_rois_flat = bottom_rois.flat<T>();

    }

 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
  int pool_channel_;
};

REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_CPU).TypeConstraint<float>("T"), RoiPoolOp<CPUDevice, float>);
