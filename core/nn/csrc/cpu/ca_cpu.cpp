#include "cpu/vision.h"


at::Tensor ca_forward_cpu(
    const torch::Tensor& t,
    const torch::Tensor& f) {
  AT_ERROR("Not implemented on the CPU");}

std::tuple<at::Tensor, at::Tensor> ca_backward_cpu(
    const at::Tensor& dw,
    const at::Tensor& t,
    const at::Tensor& f) {
  AT_ERROR("Not implemented on the CPU");}

at::Tensor ca_map_forward_cpu(
    const at::Tensor& weight,
    const at::Tensor& g) {
  AT_ERROR("Not implemented on the CPU");}

std::tuple<at::Tensor, at::Tensor> ca_map_backward_cpu(
    const at::Tensor& dout,
    const at::Tensor& weight,
    const at::Tensor& g) {
  AT_ERROR("Not implemented on the CPU");}