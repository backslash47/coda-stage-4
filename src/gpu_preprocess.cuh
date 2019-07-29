#pragma once

#include <cstdio>
#include <cstdint>

#include "gpu_constants.cuh"
#include "gpu_g1.cuh"
#include "gpu_g2.cuh"

void create_gpu_preprocessed(long size);
void free_gpu_preprocessed();
void upload_gpu_preprocessed(uint8_t *src, long size);
void download_gpu_preprocessed(uint8_t *dst, long size);

extern __device__ void getPreprocessIndex(int &rx, int &ry, int &rz, int window_idx);
extern __device__ void getPreprocessIndexMnt4G2(int &rxc0, int &rxc1, int &ryc0, int &ryc1, int &rzc0, int &rzc1, int window_idx);
extern __device__ void getPreprocessIndexMnt6G2(int &rxc0, int &rxc1, int &rxc2, int &ryc0, int &ryc1, int &ryc2, int &rzc0, int &rzc1, int &rzc2, int window_idx);

extern __device__ GpuG1 load_preprocessed_G1(int i);
extern __device__ GpuMnt4G2 load_preprocessed_mnt4_G2(int i);
extern __device__ GpuMnt6G2 load_preprocessed_mnt6_G2(int i);