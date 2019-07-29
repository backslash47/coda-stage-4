#include "gpu_fq2.cuh"
#include "gpu_fq.cuh"
#include "gpu_params.cuh"

extern __device__ GpuParams gpu_params;

__device__ GpuFq non_residue()
{
  return gpu_params.get_mnt_non_residue();
}

__device__ GpuFq2 GpuFq2::zero()
{
  return GpuFq2(GpuFq::zero(), GpuFq::zero());
}

__device__ GpuFq2 GpuFq2::one()
{
  return GpuFq2(GpuFq::one(), GpuFq::zero());
}

__device__ void GpuFq2::save(fixnum &c0, fixnum &c1)
{
  this->c0.save(c0);
  this->c1.save(c1);
}

__device__ GpuFq2 GpuFq2::operator*(const GpuFq2 &other) const
{
  GpuFq a0_b0 = this->c0 * other.c0;
  GpuFq a1_b1 = this->c1 * other.c1;

  GpuFq a0_plus_a1 = this->c0 + this->c1;
  GpuFq b0_plus_b1 = other.c0 + other.c1;

  GpuFq c = a0_plus_a1 * b0_plus_b1;

  return GpuFq2(a0_b0 + a1_b1 * non_residue(), c - a0_b0 - a1_b1);
}

__device__ GpuFq2 GpuFq2::operator+(const GpuFq2 &other) const
{
  return GpuFq2(this->c0 + other.c0, this->c1 + other.c1);
}

__device__ GpuFq2 GpuFq2::operator-(const GpuFq2 &other) const
{
  return GpuFq2(this->c0 - other.c0, this->c1 - other.c1);
}

__device__ GpuFq2 GpuFq2::operator-() const
{
  return GpuFq2(-this->c0, -this->c1);
}

__device__ bool GpuFq2::operator==(const GpuFq2 &other) const
{
  return (this->c0 == other.c0 && this->c1 == other.c1);
}

__device__ GpuFq2 GpuFq2::squared() const
{
  /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Complex squaring) */
  const GpuFq &a = this->c0, &b = this->c1;
  const GpuFq ab = a * b;

  return GpuFq2((a + b) * (a + non_residue() * b) - ab - non_residue() * ab, ab + ab);
}

__device__ bool GpuFq2::is_zero() const { return this->c0.is_zero() && this->c1.is_zero(); }