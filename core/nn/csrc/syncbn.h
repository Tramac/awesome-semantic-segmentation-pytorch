#pragma once

#include <vector>
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor batchnorm_forward(const at::Tensor input_,
                             const at::Tensor ex_,
                             const at::Tensor exs_,
                             const at::Tensor gamma_,
                             const at::Tensor beta_,
                             float eps) {
    if (input_.type().is_cuda()) {
  #ifdef WITH_CUDA
      return batchnorm_forward_cuda(input_, ex_, exs_, gamma_, beta_, eps);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    return batchnorm_forward_cpu(input_, ex_, exs_, gamma_, beta_, eps);
}

at::Tensor inp_batchnorm_forward(const at::Tensor input_,
                                 const at::Tensor ex_,
                                 const at::Tensor exs_,
                                 const at::Tensor gamma_,
                                 const at::Tensor beta_,
                                 float eps) {
    if (input_.type().is_cuda()) {
  #ifdef WITH_CUDA
      return inp_batchnorm_forward_cuda(input_, ex_, exs_, gamma_, beta_, eps);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> batchnorm_backward(const at::Tensor gradoutput_,
                                           const at::Tensor input_,
                                           const at::Tensor ex_,
                                           const at::Tensor exs_,
                                           const at::Tensor gamma_,
                                           const at::Tensor beta_,
                                           float eps) {
    if (gradoutput_.type().is_cuda()) {
  #ifdef WITH_CUDA
      return batchnorm_backward_cuda(gradoutput_, input_, ex_, exs_, gamma_, beta_, eps);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    return batchnorm_backward_cpu(gradoutput_, input_, ex_, exs_, gamma_, beta_, eps);
}

std::vector<at::Tensor> inp_batchnorm_backward(const at::Tensor gradoutput_,
                                               const at::Tensor input_,
                                               const at::Tensor ex_,
                                               const at::Tensor exs_,
                                               const at::Tensor gamma_,
                                               const at::Tensor beta_,
                                               float eps) {
    if (gradoutput_.type().is_cuda()) {
  #ifdef WITH_CUDA
      return inp_batchnorm_backward_cuda(gradoutput_, input_, ex_, exs_, gamma_, beta_, eps);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> expectation_forward(const at::Tensor input_) {
    if (input_.type().is_cuda()) {
  #ifdef WITH_CUDA
      return expectation_forward_cuda(input_);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

at::Tensor expectation_backward(const at::Tensor input_,
                                const at::Tensor gradEx_,
                                const at::Tensor gradExs_) {
    if (input_.type().is_cuda()) {
  #ifdef WITH_CUDA
      return expectation_backward_cuda(input_, gradEx_, gradExs_);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

at::Tensor inp_expectation_backward(const at::Tensor gradInput_,
                                    const at::Tensor output_,
                                    const at::Tensor gradEx_,
                                    const at::Tensor gradExs_,
                                    const at::Tensor ex_,
                                    const at::Tensor exs_,
                                    const at::Tensor gamma_,
                                    const at::Tensor beta_,
                                    float eps) {
    if (output_.type().is_cuda()) {
  #ifdef WITH_CUDA
      return inp_expectation_backward_cuda(gradInput_, output_, gradEx_, gradExs_, ex_, exs_, gamma_, beta_, eps);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}