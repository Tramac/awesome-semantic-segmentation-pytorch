#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

at::Tensor broadcast_to(at::Tensor v, at::Tensor x) {
  if (x.ndimension() == 2) {
    return v;
  } else {
    std::vector<int64_t> broadcast_size = {1, -1};
    for (int64_t i = 2; i < x.ndimension(); ++i)
      broadcast_size.push_back(1);

    return v.view(broadcast_size);
  }
}

at::Tensor batchnorm_forward_cpu(
    const at::Tensor input_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps) {
  auto output = (input_ - broadcast_to(ex_, input_)) / broadcast_to(exs_, input_);
  output = output * broadcast_to(gamma_, input_) + broadcast_to(beta_, input_);
  return output;
}

// Not implementing CPU backward for now
std::vector<at::Tensor> batchnorm_backward_cpu(
    const at::Tensor gradoutput_,
    const at::Tensor input_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps) {
  /* outputs*/
  at::Tensor gradinput = at::zeros_like(input_);
  at::Tensor gradgamma = at::zeros_like(gamma_);
  at::Tensor gradbeta = at::zeros_like(beta_);
  at::Tensor gradMean = at::zeros_like(ex_);
  at::Tensor gradStd = at::zeros_like(exs_);
  return {gradinput, gradMean, gradStd, gradgamma, gradbeta};
}