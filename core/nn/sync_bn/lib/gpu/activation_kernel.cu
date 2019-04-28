#include <vector>
// #include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime_api.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>


namespace {

template<typename T>
inline void leaky_relu_backward_impl(T *z, T *dz, float slope, int64_t count) {
  // Create thrust pointers
  thrust::device_ptr<T> th_z = thrust::device_pointer_cast(z);
  thrust::device_ptr<T> th_dz = thrust::device_pointer_cast(dz);

  thrust::transform_if(th_dz, th_dz + count, th_z, th_dz,
                       [slope] __device__ (const T& dz) { return dz * slope; },
                       [] __device__ (const T& z) { return z < 0; });
  thrust::transform_if(th_z, th_z + count, th_z,
                       [slope] __device__ (const T& z) { return z / slope; },
                       [] __device__ (const T& z) { return z < 0; });
}

}

void LeakyRelu_Forward_CUDA(at::Tensor z, float slope) {
  at::leaky_relu_(z, slope);
}

void LeakyRelu_Backward_CUDA(at::Tensor z, at::Tensor dz, float slope) {
  int64_t count = z.numel();

  AT_DISPATCH_FLOATING_TYPES(z.type(), "LeakyRelu_Backward_CUDA", ([&] {
    leaky_relu_backward_impl<scalar_t>(z.data<scalar_t>(), dz.data<scalar_t>(), slope, count);
  }));
  /*
  // unstable after scaling
  at::leaky_relu_(z, 1.0 / slope);
  at::leaky_relu_backward(dz, z, slope);
  */
}