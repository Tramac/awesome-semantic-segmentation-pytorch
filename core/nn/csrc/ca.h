#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor ca_forward(const at::Tensor& t,
                      const at::Tensor& f) {
    if (t.type().is_cuda()) {
  #ifdef WITH_CUDA
      return ca_forward_cuda(t, f);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    return ca_forward_cpu(t, f);
}

std::tuple<at::Tensor, at::Tensor> ca_backward(const at::Tensor& dw,
                                               const at::Tensor& t,
                                               const at::Tensor& f) {
    if (dw.type().is_cuda()) {
  #ifdef WITH_CUDA
      return ca_backward_cuda(dw, t, f);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    return ca_backward_cpu(dw, t, f);
}

at::Tensor ca_map_forward(const at::Tensor& weight,
                          const at::Tensor& g) {
    if (weight.type().is_cuda()) {
  #ifdef WITH_CUDA
      return ca_map_forward_cuda(weight, g);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    return ca_map_forward_cpu(weight, g);
}

std::tuple<at::Tensor, at::Tensor> ca_map_backward(const at::Tensor& dout,
                                                   const at::Tensor& weight,
                                                   const at::Tensor& g) {
    if (dout.type().is_cuda()) {
  #ifdef WITH_CUDA
      return ca_map_backward_cuda(dout, weight, g);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    return ca_map_backward_cpu(dout, weight, g);
}