#pragma once

#include "gpu_fq.cuh"

class GpuFq2
{
public:
  GpuFq c0, c1;

  __device__
  GpuFq2(const GpuFq &c0, const GpuFq &c1) : c0(c0), c1(c1) {}

  __device__
  GpuFq2() {}

  __device__ static GpuFq2 zero();

  __device__ static GpuFq2 one();

  __device__ void save(fixnum &c0, fixnum &c1);

  __device__ GpuFq2 operator*(const GpuFq2 &other) const;

  __device__ GpuFq2 operator+(const GpuFq2 &other) const;

  __device__ GpuFq2 operator-(const GpuFq2 &other) const;
  __device__ GpuFq2 operator-() const;

  __device__ bool operator==(const GpuFq2 &other) const;

  __device__ GpuFq2 squared() const;

  __device__ bool is_zero() const;
};
