#pragma once

#include "gpu_fq.cuh"

class GpuFq3
{
public:
  GpuFq c0, c1, c2;

public:
  __device__
  GpuFq3(const GpuFq &c0, const GpuFq &c1, const GpuFq &c2) : c0(c0), c1(c1), c2(c2) {}

  __device__
  GpuFq3() {}

  __device__ static GpuFq3 zero();

  __device__ static GpuFq3 one();

  __device__ void save(fixnum &c0, fixnum &c1, fixnum &c2);

  __device__ GpuFq3 operator*(const GpuFq3 &other) const;

  __device__ GpuFq3 operator+(const GpuFq3 &other) const;

  __device__ GpuFq3 operator-(const GpuFq3 &other) const;
  __device__ GpuFq3 operator-() const;

  __device__ bool is_zero() const;

  __device__ bool operator==(const GpuFq3 &other) const;

  __device__ GpuFq3 squared() const;
};
