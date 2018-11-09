#include <TH/TH.h>

#include <stdio.h>

#include <math.h>

int pow_forward(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output)
{
    printf("RUNNING POW FORWARD ON CPU!\n");

    if (!THFloatTensor_isSameSizeAs(input1, input2))
        return 0;
    // THFloatTensor_resizeAs(output, input1);
    // THFloatTensor_cadd(output, input1, 1.0, input2);

    int N = THFloatTensor_size(input1, 0);
    const float* input1_flat = THFloatTensor_data(input1);
    const float* input2_flat = THFloatTensor_data(input2);
    float* output_flat = THFloatTensor_data(output);

    for (int i = 0; i < N; ++i)
    {
        output_flat[i] = (float) pow(input1_flat[i], input2_flat[i]);
    }

    return 1;
}

int pow_backward(THFloatTensor* grad_output, THFloatTensor* input1, THFloatTensor* input2, THFloatTensor *grad_input1, THFloatTensor *grad_input2)
{
    // THFloatTensor_resizeAs(grad_input, grad_output);
    // THFloatTensor_fill(grad_input, 1);
    if (!THFloatTensor_isSameSizeAs(input1, grad_input1))
    {
        return 0;
    }
    if (!THFloatTensor_isSameSizeAs(grad_input1, grad_input2))
        return 0;
    int N = THFloatTensor_size(grad_input1, 0);

    const float* grad_output_flat = THFloatTensor_data(grad_output);
    const float* input1_flat = THFloatTensor_data(input1);
    const float* input2_flat = THFloatTensor_data(input2);
    float* grad_input1_flat = THFloatTensor_data(grad_input1);
    float* grad_input2_flat = THFloatTensor_data(grad_input2);

    for (int i = 0; i < N; ++i)
    {
        float v1 = input1_flat[i];
        float v2 = input2_flat[i];
        float g_o = grad_output_flat[i];
        grad_input1_flat[i] = g_o * v2 * pow(v1, v2-1);
        grad_input2_flat[i] = g_o * pow(v1, v2) * log(v1);
    }

    return 1;
}

