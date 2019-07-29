#include "gpu_constants.cuh"
#include "gpu_params.cuh"
#include "gpu_fq.cuh"

extern __device__ GpuParams gpu_params;

__device__ modnum mod()
{
  return gpu_params.get_mnt_mod();
}

__device__ GpuFq GpuFq::zero()
{
  return GpuFq(mod().zero());
}

__device__ GpuFq GpuFq::one()
{
  return GpuFq(mod().one());
}

__device__ GpuFq GpuFq::load(const fixnum &data)
{
  fixnum result;
  mod().to_modnum(result, data);
  return GpuFq(result);
}

__device__ void GpuFq::save(fixnum &result)
{
  mod().from_modnum(result, this->data);
}

__device__ GpuFq GpuFq::operator*(const GpuFq &other) const
{
  fixnum result;
  mod().mul(result, this->data, other.data);
  return GpuFq(result);
}

__device__ GpuFq GpuFq::operator+(const GpuFq &other) const
{
  fixnum result;
  mod().add(result, this->data, other.data);
  return GpuFq(result);
}

__device__ GpuFq GpuFq::operator-(const GpuFq &other) const
{
  fixnum result;
  mod().sub(result, this->data, other.data);
  return GpuFq(result);
}

__device__ GpuFq GpuFq::operator-() const
{
  if (this->is_zero())
  {
    return (*this);
  }
  else
  {
    fixnum result;
    mod().neg(result, this->data);
    return GpuFq(result);
  }
}

__device__ bool GpuFq::operator==(const GpuFq &other) const
{
  return fixnum::cmp(this->data, other.data) == 0;
}

__device__ void GpuFq::operator=(const GpuFq &other)
{
  this->data = other.data;
}

__device__ GpuFq GpuFq::squared() const
{
  fixnum result;
  mod().sqr(result, this->data);
  return GpuFq(result);
}

__device__ bool GpuFq::is_zero() const
{
  return fixnum::is_zero(this->data);
}
