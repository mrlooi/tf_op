#include <THC/THC.h>
#include <math.h>

#include <stdio.h>
#include "add_op_kernel.h"

extern THCState *state;

int add_forward_cuda(const int N, THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output)
{
    printf("RUNNING ADD FORWARD ON CUDA\n");
    // Grab the input tensor
    float * input1_flat = THCudaTensor_data(state, input1);
    float * input2_flat = THCudaTensor_data(state, input2);

    // THCudaTensor_resizeAs(state, output, input1);
    float * output_flat = THCudaTensor_data(state, output);

    cudaStream_t stream = THCState_getCurrentStream(state);
    // THCudaTensor_(state, output);
    // THCudaStorage

    // int N = sizeof(input1_flat) / sizeof(float);

    // int sz = THCudaTensor_size(state, input1, 0);
    AddForwardLauncher(N, input1_flat, input2_flat, output_flat, stream);

    return 1;
}

int add_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input)
{

    return 1;
}
