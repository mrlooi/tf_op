#include <TH/TH.h>

#include <stdio.h>

#include <math.h>

int pow_forward(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output)
{
	printf("RUNNING POW FORWARD ON CPU!\n");

    if (!THFloatTensor_isSameSizeAs(input1, input2))
        return 0;
    THFloatTensor_resizeAs(output, input1);
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

int pow_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return 1;
}