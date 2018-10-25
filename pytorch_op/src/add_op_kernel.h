#ifndef _ADD_OP_KERNEL
#define _ADD_OP_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int AddForwardLauncher(const int N, const float* input1_data, const float* input2_data, float* output_data, cudaStream_t stream);

// int AddBackwardLauncher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
//     const int height, const int width, const int channels, const int pooled_height,
//     const int pooled_width, const float* bottom_rois,
//     float* bottom_diff, const int* argmax_data, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

