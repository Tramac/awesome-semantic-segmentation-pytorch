#pragma once
#include <torch/extension.h>


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