#pragma once
#include <torch/extension.h>
#include <vector>


at::Tensor ca_forward_cuda(
    const at::Tensor& t,
    const at::Tensor& f);

std::tuple<at::Tensor, at::Tensor> ca_backward_cuda(
    const at::Tensor& dw,
    const at::Tensor& t,
    const at::Tensor& f);

at::Tensor ca_map_forward_cuda(
    const at::Tensor& weight,
    const at::Tensor& g);

std::tuple<at::Tensor, at::Tensor> ca_map_backward_cuda(
    const at::Tensor& dout,
    const at::Tensor& weight,
    const at::Tensor& g);

at::Tensor psa_forward_cuda(
    const at::Tensor& hc,
    const int forward_type);

at::Tensor psa_backward_cuda(
    const at::Tensor& dout,
    const at::Tensor& hc,
    const int forward_type);

at::Tensor batchnorm_forward_cuda(
    const at::Tensor input_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps);

at::Tensor inp_batchnorm_forward_cuda(
    const at::Tensor input_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps);

std::vector<at::Tensor> batchnorm_backward_cuda(
    const at::Tensor gradoutput_,
    const at::Tensor input_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps);

std::vector<at::Tensor> inp_batchnorm_backward_cuda(
    const at::Tensor gradoutput_,
    const at::Tensor output_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps);

std::vector<at::Tensor> expectation_forward_cuda(
    const at::Tensor input_);

at::Tensor expectation_backward_cuda(
    const at::Tensor input_,
    const at::Tensor gradEx_,
    const at::Tensor gradExs_);

at::Tensor inp_expectation_backward_cuda(
    const at::Tensor gradInput_,
    const at::Tensor output_,
    const at::Tensor gradEx_,
    const at::Tensor gradExs_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps);