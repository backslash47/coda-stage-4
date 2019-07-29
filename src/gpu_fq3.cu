#include "gpu_fq3.cuh"
#include "gpu_fq.cuh"
#include "gpu_params.cuh"

extern __device__ GpuParams gpu_params;

__device__ GpuFq non_residue3()
{
  return gpu_params.get_mnt_non_residue();
}

__device__ void GpuFq3::save(fixnum &c0, fixnum &c1, fixnum &c2)
{
  this->c0.save(c0);
  this->c1.save(c1);
  this->c2.save(c2);
}

__device__ GpuFq3 GpuFq3::zero()
{
  return GpuFq3(GpuFq::zero(), GpuFq::zero(), GpuFq::zero());
}

__device__ GpuFq3 GpuFq3::one()
{
  return GpuFq3(GpuFq::one(), GpuFq::zero(), GpuFq::zero());
}


__device__ GpuFq3 GpuFq3::operator*(const GpuFq3 &other) const
{
  const GpuFq c0_c0 = this->c0 * other.c0;
  const GpuFq c1_c1 = this->c1 * other.c1;
  const GpuFq c2_c2 = this->c2 * other.c2;

  return GpuFq3(c0_c0 + non_residue3() * ((this->c1 + this->c2) * (other.c1 + other.c2) - c1_c1 - c2_c2),
                (this->c0 + this->c1) * (other.c0 + other.c1) - c0_c0 - c1_c1 + non_residue3() * c2_c2,
                (this->c0 + this->c2) * (other.c0 + other.c2) - c0_c0 + c1_c1 - c2_c2);
}

__device__ GpuFq3 GpuFq3::operator+(const GpuFq3 &other) const
{
  return GpuFq3(this->c0 + other.c0, this->c1 + other.c1, this->c2 + other.c2);
}

__device__ GpuFq3 GpuFq3::operator-(const GpuFq3 &other) const
{
  return GpuFq3(this->c0 - other.c0, this->c1 - other.c1, this->c2 - other.c2);
}

__device__ GpuFq3 GpuFq3::operator-() const
{
  return GpuFq3(-this->c0, -this->c1, -this->c2);
}

__device__ bool GpuFq3::is_zero() const { return this->c0.is_zero() && this->c1.is_zero() && this->c2.is_zero(); }

__device__ bool GpuFq3::operator==(const GpuFq3 &other) const
{
  return (this->c0 == other.c0 && this->c1 == other.c1 && this->c2 == other.c2);
}

__device__ GpuFq3 GpuFq3::squared() const
{
  /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 4 (CH-SQR2) */
  const GpuFq
      &a = this->c0,
      &b = this->c1, &c = this->c2;
  const GpuFq s0 = a.squared();
  const GpuFq ab = a * b;
  const GpuFq s1 = ab + ab;
  const GpuFq s2 = (a - b + c).squared();
  const GpuFq bc = b * c;
  const GpuFq s3 = bc + bc;
  const GpuFq s4 = c.squared();

  return GpuFq3(s0 + non_residue3() * s3,
                s1 + non_residue3() * s4,
                s1 + s2 + s3 - s0 - s4);
}