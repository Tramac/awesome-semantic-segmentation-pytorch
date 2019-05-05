#include "cpu/vision.h"


at::Tensor psa_forward_cpu(
    const torch::Tensor& h,
    const int forward_type) {
  AT_ERROR("Not implemented on the CPU")}

at::Tensor psa_backward_cpu(
    const at::Tensor& dout,
    const at::Tensor& h,
    const int forward_type) {
  AT_ERROR("Not implemented on the CPU")}