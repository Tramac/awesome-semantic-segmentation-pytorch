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

at::Tensor BatchNorm_Forward_CPU(
    const at::Tensor input, 
    const at::Tensor mean,
    const at::Tensor std,
    const at::Tensor gamma,
    const at::Tensor beta) {
  auto output = (input - broadcast_to(mean, input)) / broadcast_to(std, input);
  output = output * broadcast_to(gamma, input) + broadcast_to(beta, input);
  return output;
}

// Not implementing CPU backward for now
std::vector<at::Tensor> BatchNorm_Backward_CPU(
    const at::Tensor gradoutput,
    const at::Tensor input,
    const at::Tensor mean, 
    const at::Tensor std,
    const at::Tensor gamma,
    const at::Tensor beta, 
    bool train) {
  /* outputs*/
  at::Tensor gradinput = at::zeros_like(input);
  at::Tensor gradgamma = at::zeros_like(gamma);
  at::Tensor gradbeta = at::zeros_like(beta);
  at::Tensor gradMean = at::zeros_like(mean);
  at::Tensor gradStd = at::zeros_like(std);
  return {gradinput, gradMean, gradStd, gradgamma, gradbeta};
}

std::vector<at::Tensor> Sum_Square_Forward_CPU(
    const at::Tensor input) {
  /* outputs */
  at::Tensor sum = torch::zeros({input.size(1)}, input.options());
  at::Tensor square = torch::zeros({input.size(1)}, input.options());
  return {sum, square};
}

at::Tensor Sum_Square_Backward_CPU(
    const at::Tensor input,
    const at::Tensor gradSum,
    const at::Tensor gradSquare) {
  /* outputs */
  at::Tensor gradInput = at::zeros_like(input);
  return gradInput;
}