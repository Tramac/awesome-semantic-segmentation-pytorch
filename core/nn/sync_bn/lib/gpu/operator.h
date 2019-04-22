#include <torch/extension.h>
#include <vector>

at::Tensor BatchNorm_Forward_CUDA(
  const at::Tensor input_, 
  const at::Tensor mean_,
  const at::Tensor std_,
  const at::Tensor gamma_,
  const at::Tensor beta_,
  float eps);

at::Tensor BatchNorm_Forward_Inp_CUDA(
    const at::Tensor input_, 
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps);

std::vector<at::Tensor> BatchNorm_Backward_CUDA(
  const at::Tensor gradoutput_,
  const at::Tensor input_,
  const at::Tensor ex_, 
  const at::Tensor exs_,
  const at::Tensor gamma_,
  const at::Tensor beta_,
  float eps);

std::vector<at::Tensor> BatchNorm_Inp_Backward_CUDA(
  const at::Tensor gradoutput_,
  const at::Tensor output_,
  const at::Tensor ex_, 
  const at::Tensor exs_,
  const at::Tensor gamma_,
  const at::Tensor beta_,
  float eps);

std::vector<at::Tensor> Expectation_Forward_CUDA(
  const at::Tensor input_);

at::Tensor Expectation_Backward_CUDA(
  const at::Tensor input_,
  const at::Tensor gradEx_,
  const at::Tensor gradExs_);

at::Tensor Expectation_Inp_Backward_CUDA(
  const at::Tensor gradInput_,
  const at::Tensor output_,
  const at::Tensor gradEx_,
  const at::Tensor gradExs_,
  const at::Tensor ex_,
  const at::Tensor exs_,
  const at::Tensor gamma_,
  const at::Tensor beta_,
  float eps);

void LeakyRelu_Forward_CUDA(at::Tensor z, float slope);

void LeakyRelu_Backward_CUDA(at::Tensor z, at::Tensor dz, float slope);