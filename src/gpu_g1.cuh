#pragma once

#include "gpu_fq.cuh"

class GpuG1
{
public:
  GpuFq X, Y, Z;

public:
  __device__
  GpuG1(const GpuFq &X, const GpuFq &Y, const GpuFq &Z) : X(X), Y(Y), Z(Z) {}

  __device__ GpuG1(const GpuG1 &g) : X(g.X), Y(g.Y), Z(g.Z) {}

  __device__ GpuG1() {}

  __device__ static GpuG1 zero();

  __device__ static GpuG1 load_from_array(fixnum *array, long index);
  //__device__ static GpuG1 load_from_array_affine(fixnum *array, long index);
  __device__ void save_to_array(fixnum *array, long index);

  __device__ void save(fixnum &x, fixnum &y, fixnum &z);

  __device__ bool is_zero() const;

  __device__ GpuG1 operator+(const GpuG1 &other) const;
  __device__ GpuG1 operator-() const;
  __device__ GpuG1 operator-(const GpuG1 &other) const;
  __device__ GpuG1 mixed_add(const GpuG1 &other) const;

  __device__ GpuG1 operator*(const fixnum &scalar);

  __device__ void operator=(const GpuG1 &g);

  __device__ GpuG1 dbl() const;
};