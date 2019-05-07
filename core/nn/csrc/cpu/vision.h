#pragma once
#include <torch/extension.h>


at::Tensor ca_forward_cpu(
    const at::Tensor& t,
    const at::Tensor& f);

std::tuple<at::Tensor, at::Tensor> ca_backward_cpu(
    const at::Tensor& dw,
    const at::Tensor& t,
    const at::Tensor& f);

at::Tensor ca_map_forward_cpu(
    const at::Tensor& weight,
    const at::Tensor& g);

std::tuple<at::Tensor, at::Tensor> ca_map_backward_cpu(
    const at::Tensor& dout,
    const at::Tensor& weight,
    const at::Tensor& g);

at::Tensor psa_forward_cpu(
    const at::Tensor& hc,
    const int forward_type);

at::Tensor psa_backward_cpu(
    const at::Tensor& dout,
    const at::Tensor& hc,
    const int forward_type);

at::Tensor batchnorm_forward_cpu(
    const at::Tensor input_,
    const at::Tensor mean_,
    const at::Tensor std_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps);

std::vector<at::Tensor> batchnorm_backward_cpu(
    const at::Tensor gradoutput_,
    const at::Tensor input_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps);