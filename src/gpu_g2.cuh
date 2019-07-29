#pragma once

#include "gpu_fq.cuh"
#include "gpu_fq2.cuh"
#include "gpu_fq3.cuh"

class GpuMnt4G2
{
public:
  GpuFq2 X, Y, Z;

  __device__
  GpuMnt4G2(const GpuFq2 &X, const GpuFq2 &Y, const GpuFq2 &Z) : X(X), Y(Y), Z(Z) {}

  __device__
  GpuMnt4G2() {}

  __device__ static GpuMnt4G2 zero();

  __device__ static GpuMnt4G2 load_from_array(fixnum *array, long index);
  //__device__ static GpuMnt4G2 load_from_array_affine(fixnum *array, long index);
  __device__ void save_to_array(fixnum *array, long index);

  __device__ void save(fixnum &xc0, fixnum &xc1, fixnum &yc0, fixnum &yc1, fixnum &zc0, fixnum &zc1);

  __device__ bool is_zero() const;

  __device__ GpuMnt4G2 operator+(const GpuMnt4G2 &other) const;
  __device__ GpuMnt4G2 operator-() const;
  __device__ GpuMnt4G2 operator-(const GpuMnt4G2 &other) const;
  __device__ GpuMnt4G2 mixed_add(const GpuMnt4G2 &other) const;

  __device__ GpuMnt4G2 operator*(const fixnum &scalar);

  __device__ GpuMnt4G2 dbl() const;

  __device__ static GpuFq2 mul_by_a(const GpuFq2 &elt);
};

class GpuMnt6G2
{
public:
  GpuFq3 X, Y, Z;

  __device__
  GpuMnt6G2(const GpuFq3 &X, const GpuFq3 &Y, const GpuFq3 &Z) : X(X), Y(Y), Z(Z) {}

  __device__
  GpuMnt6G2() {}

  __device__ static GpuMnt6G2 zero();

  __device__ void save(fixnum &xc0, fixnum &xc1, fixnum &xc2, fixnum &yc0, fixnum &yc1, fixnum &yc2, fixnum &zc0, fixnum &zc1, fixnum &zc2);

  __device__ bool is_zero() const;

  __device__ static GpuMnt6G2 load_from_array(fixnum *array, long index);
  //__device__ static GpuMnt6G2 load_from_array_affine(fixnum *array, long index);
  __device__ void save_to_array(fixnum *array, long index);

  __device__ GpuMnt6G2 operator+(const GpuMnt6G2 &other) const;
  __device__ GpuMnt6G2 operator-() const;
  __device__ GpuMnt6G2 operator-(const GpuMnt6G2 &other) const;
  __device__ GpuMnt6G2 mixed_add(const GpuMnt6G2 &other) const;

  __device__ GpuMnt6G2 dbl() const;

  __device__ static GpuFq3 mul_by_a(const GpuFq3 &elt);
};
