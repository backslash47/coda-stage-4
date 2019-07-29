#pragma once

#include "gpu_constants.cuh"

class GpuFq
{
public:
  fixnum data;

public:
  __device__ __forceinline__
  GpuFq(const fixnum &data) : data(data) {}

  __device__ __forceinline__
  GpuFq() {}

  __device__ static GpuFq zero();

  __device__ static GpuFq one();

  __device__ static GpuFq load(const fixnum &data);

  __device__ void save(fixnum &result);

  __device__ GpuFq operator*(const GpuFq &other) const;

  __device__ GpuFq operator+(const GpuFq &other) const;

  __device__ GpuFq operator-(const GpuFq &other) const;

  __device__ GpuFq operator-() const;

  __device__ bool operator==(const GpuFq &other) const;

  __device__ void operator=(const GpuFq &other);

  __device__ GpuFq squared() const;

  __device__ bool is_zero() const;
};
