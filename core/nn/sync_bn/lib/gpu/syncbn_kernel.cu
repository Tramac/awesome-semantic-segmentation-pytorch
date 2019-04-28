#include <vector>
// #include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "common.h"
#include "device_tensor.h"

namespace {

template <typename DType, typename Acctype, typename DeviceTensor3>
struct GradOp {
  __device__ GradOp(Acctype m, const DeviceTensor3 i, const DeviceTensor3 g)
    : beta(m), output(i), gradOutput(g) {}
  __device__ __forceinline__ Float2<DType, Acctype> operator()(int batch, int plane, int n) {
    DType g = gradOutput[batch][plane][n];
    DType c = ScalarConvert<Acctype, DType>::to(output[batch][plane][n] - beta);
    return Float2<DType, Acctype>(g, g * c);
  }
  const Acctype beta;
  const DeviceTensor3 output;
  const DeviceTensor3 gradOutput;
};

template <typename DType, typename Acctype>
struct SumOp {
  __device__ SumOp(DeviceTensor<DType, 3> i) : input(i){}
  __device__ __forceinline__ Float2<DType, Acctype> operator()(int batch, int plane, int n) {
    DType g = input[batch][plane][n];
    return Float2<DType, Acctype>(g, g * g);
  }
  DType mean;
  DeviceTensor<DType, 3> input;
};

// Sum across (batch, x/y/z) applying Op() pointwise
template<typename T, typename Op, typename DeviceTensor3>
__device__ T reduce(Op op, DeviceTensor3 tensor, int plane) {
  T sum = (T)0;
  for (int batch = 0; batch < tensor.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < tensor.getSize(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T)0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

template <typename DType>
__global__ void BatchNorm_Forward_kernel (
  DeviceTensor<DType, 3> output,
  DeviceTensor<DType, 3> input,
  DeviceTensor<DType, 1> mean,
  DeviceTensor<DType, 1> std,
  DeviceTensor<DType, 1> gamma,
  DeviceTensor<DType, 1> beta) {
  int c = blockIdx.x;
  /* main operation */
  for (int b = 0; b < input.getSize(0); ++b) {
    for (int x = threadIdx.x; x < input.getSize(2); x += blockDim.x) {
      DType inp = input[b][c][x];
      output[b][c][x] = gamma[c] * (inp - mean[c]) /
        std[c] + beta[c];
    }
  }
}

template <typename DType>
__global__ void BatchNorm_Forward_Inp_kernel (
  DeviceTensor<DType, 3> input,
  DeviceTensor<DType, 1> mean,
  DeviceTensor<DType, 1> std,
  DeviceTensor<DType, 1> gamma,
  DeviceTensor<DType, 1> beta) {
  int c = blockIdx.x;
  /* main operation */
  for (int b = 0; b < input.getSize(0); ++b) {
    for (int x = threadIdx.x; x < input.getSize(2); x += blockDim.x) {
      DType inp = input[b][c][x];
      input[b][c][x] = gamma[c] * (inp - mean[c]) /
        std[c] + beta[c];
    }
  }
}

template <typename DType>
__global__ void BatchNorm_Backward_Inp_kernel (
    DeviceTensor<DType, 3> gradoutput,
    DeviceTensor<DType, 3> output,
    DeviceTensor<DType, 3> gradinput,
    DeviceTensor<DType, 1> gradgamma,
    DeviceTensor<DType, 1> gradbeta,
    DeviceTensor<DType, 1> mean,
    DeviceTensor<DType, 1> std,
    DeviceTensor<DType, 1> gamma,
    DeviceTensor<DType, 1> beta,
    DeviceTensor<DType, 1> gradEx,
    DeviceTensor<DType, 1> gradExs) {
  /* declarations of the variables */
  /* Get the index and channels */
  int c = blockIdx.x;
  /* main operation */
  GradOp<DType, DType, DeviceTensor<DType, 3>> g(beta[c], output, gradoutput);
  Float2<DType, DType> res = reduce<Float2<DType, DType>,
    GradOp<DType, DType, DeviceTensor<DType, 3>>,
    DeviceTensor<DType, 3>>(g, gradoutput, c);
  DType gradOutputSum = res.v1;
  DType dotP = res.v2;
  DType invstd = DType(1.0) / std[c];
  DType gradScale = invstd * gamma[c];
  if (threadIdx.x == 0) {
    gradEx[c] = - gradOutputSum * gradScale + mean[c] * invstd * invstd * dotP;
    gradExs[c]  = - 0.5 * invstd * invstd * dotP;
  }
  if (gradinput.numElements() > 0) {
    for (int batch = 0; batch < gradoutput.getSize(0); ++batch) {
      for (int x = threadIdx.x; x < gradoutput.getSize(2); x += blockDim.x) {
        gradinput[batch][c][x] = gradoutput[batch][c][x] * gradScale;
      }
    }
  }
  if (gradgamma.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradgamma[c] += dotP / gamma[c];
    }
  }
  if (gradbeta.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradbeta[c] += gradOutputSum;
    }
  }
}

template <typename DType>
__global__ void BatchNorm_Backward_kernel (
    DeviceTensor<DType, 3> gradoutput,
    DeviceTensor<DType, 3> input,
    DeviceTensor<DType, 3> gradinput,
    DeviceTensor<DType, 1> gradgamma,
    DeviceTensor<DType, 1> gradbeta,
    DeviceTensor<DType, 1> mean,
    DeviceTensor<DType, 1> std,
    DeviceTensor<DType, 1> gamma,
    DeviceTensor<DType, 1> beta,
    DeviceTensor<DType, 1> gradEx,
    DeviceTensor<DType, 1> gradExs) {
  /* declarations of the variables */
  /* Get the index and channels */
  int c = blockIdx.x;
  /* main operation */
  GradOp<DType, DType, DeviceTensor<DType, 3>> g(mean[c], input, gradoutput);
  Float2<DType, DType> res = reduce<Float2<DType, DType>,
    GradOp<DType, DType, DeviceTensor<DType, 3>>,
    DeviceTensor<DType, 3>>(g, gradoutput, c);
  DType gradOutputSum = res.v1;
  DType dotP = res.v2;
  DType invstd = DType(1.0) / std[c];
  DType gradScale = invstd * gamma[c];
  if (threadIdx.x == 0) {
    gradEx[c] = - gradOutputSum * gradScale + mean[c] * invstd * invstd * dotP * gradScale;
    gradExs[c]  = - 0.5 * invstd * invstd * dotP * gradScale;
  }
  if (gradinput.numElements() > 0) {
    for (int batch = 0; batch < gradoutput.getSize(0); ++batch) {
      for (int x = threadIdx.x; x < gradoutput.getSize(2); x += blockDim.x) {
        gradinput[batch][c][x] = gradoutput[batch][c][x] * gradScale;
      }
    }
  }
  if (gradgamma.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradgamma[c] += dotP * invstd;
    }
  }
  if (gradbeta.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradbeta[c] += gradOutputSum;
    }
  }
}


template <typename DType>
__global__ void Expectation_Forward_kernel (
    DeviceTensor<DType, 3> input,
    DeviceTensor<DType, 1> ex,
    DeviceTensor<DType, 1> exs,
    DType norm) {
  int c = blockIdx.x;
  /* main operation */
  SumOp<DType, DType> g(input);
  Float2<DType, DType> res = reduce<Float2<DType, DType>,
    SumOp<DType, DType>, DeviceTensor<DType, 3>>(g, input, c);
  DType xsum = res.v1;
  DType xsquare = res.v2;
  if (threadIdx.x == 0) {
    ex[c] = xsum * norm;
    exs[c] = xsquare * norm;
  }
}

template <typename DType>
__global__ void Expectation_Backward_kernel (
  DeviceTensor<DType, 3> gradInput,
  DeviceTensor<DType, 3> input,
  DeviceTensor<DType, 1> gradEx,
  DeviceTensor<DType, 1> gradExs,
  DType norm) {
  int c = blockIdx.x;
  /* main operation */
  for (int batch = 0; batch < gradInput.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < gradInput.getSize(2); x += blockDim.x) {
      gradInput[batch][c][x] = gradEx[c] * norm + 2 * gradExs[c] *
          input[batch][c][x] * norm;
    }
  }
}

template <typename DType>
__global__ void Expectation_Backward_Inp_kernel (
  DeviceTensor<DType, 3> gradInput,
  DeviceTensor<DType, 3> output,
  DeviceTensor<DType, 1> gradEx,
  DeviceTensor<DType, 1> gradExs,
  DeviceTensor<DType, 1> mean,
  DeviceTensor<DType, 1> std,
  DeviceTensor<DType, 1> gamma,
  DeviceTensor<DType, 1> beta,
  DType norm) {
  int c = blockIdx.x;
  /* main operation */
  for (int batch = 0; batch < gradInput.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < gradInput.getSize(2); x += blockDim.x) {
      gradInput[batch][c][x] += gradEx[c] * norm + 2 * gradExs[c] *
          ((output[batch][c][x] - beta[c]) / gamma[c] * std[c] + mean[c]) * norm;
    }
  }
}

} // namespace

at::Tensor BatchNorm_Forward_CUDA(
    const at::Tensor input_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps) {
  auto output_ = at::zeros_like(input_);
  auto std_ = (exs_ - ex_ * ex_ + eps).sqrt();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(input_.size(1));
  dim3 threads(getNumThreads(input_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(input_.type(), "BatchNorm_Forward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> output = devicetensor<scalar_t, 3>(output_);
    DeviceTensor<scalar_t, 3> input = devicetensor<scalar_t, 3>(input_);
    DeviceTensor<scalar_t, 1> ex = devicetensor<scalar_t, 1>(ex_);
    DeviceTensor<scalar_t, 1> std = devicetensor<scalar_t, 1>(std_);
    DeviceTensor<scalar_t, 1> gamma = devicetensor<scalar_t, 1>(gamma_);
    DeviceTensor<scalar_t, 1> beta = devicetensor<scalar_t, 1>(beta_);
    /* kernel function */
    BatchNorm_Forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        output, input, ex, std, gamma, beta);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return output_;
}

at::Tensor BatchNorm_Forward_Inp_CUDA(
    const at::Tensor input_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps) {
  auto std_ = (exs_ - ex_ * ex_ + eps).sqrt();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(input_.size(1));
  dim3 threads(getNumThreads(input_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(input_.type(), "BatchNorm_Forward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> input = devicetensor<scalar_t, 3>(input_);
    DeviceTensor<scalar_t, 1> ex = devicetensor<scalar_t, 1>(ex_);
    DeviceTensor<scalar_t, 1> std = devicetensor<scalar_t, 1>(std_);
    DeviceTensor<scalar_t, 1> gamma = devicetensor<scalar_t, 1>(gamma_);
    DeviceTensor<scalar_t, 1> beta = devicetensor<scalar_t, 1>(beta_);
    /* kernel function */
    BatchNorm_Forward_Inp_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input, ex, std, gamma, beta);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return input_;
}


std::vector<at::Tensor> BatchNorm_Inp_Backward_CUDA(
    const at::Tensor gradoutput_,
    const at::Tensor output_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps) {
  /* outputs*/
  auto std_ = (exs_ - ex_ * ex_ + eps).sqrt();
  auto gradinput_ = at::zeros_like(output_);
  auto gradgamma_ = at::zeros_like(gamma_);
  auto gradbeta_ = at::zeros_like(beta_);
  auto gradEx_ = at::zeros_like(ex_);
  auto gradExs_ = at::zeros_like(std_);
  /* cuda utils*/
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(output_.size(1));
  dim3 threads(getNumThreads(output_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(output_.type(), "BatchNorm_Inp_Backward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> gradoutput = devicetensor<scalar_t, 3>(gradoutput_);
    DeviceTensor<scalar_t, 3> output = devicetensor<scalar_t, 3>(output_);
    DeviceTensor<scalar_t, 3> gradinput = devicetensor<scalar_t, 3>(gradinput_);
    DeviceTensor<scalar_t, 1> gradgamma = devicetensor<scalar_t, 1>(gradgamma_);
    DeviceTensor<scalar_t, 1> gradbeta = devicetensor<scalar_t, 1>(gradbeta_);
    DeviceTensor<scalar_t, 1> ex = devicetensor<scalar_t, 1>(ex_);
    DeviceTensor<scalar_t, 1> std = devicetensor<scalar_t, 1>(std_);
    DeviceTensor<scalar_t, 1> gamma = devicetensor<scalar_t, 1>(gamma_);
    DeviceTensor<scalar_t, 1> beta = devicetensor<scalar_t, 1>(beta_);
    DeviceTensor<scalar_t, 1> gradEx = devicetensor<scalar_t, 1>(gradEx_);
    DeviceTensor<scalar_t, 1> gradExs = devicetensor<scalar_t, 1>(gradExs_);
    /* kernel function */
    BatchNorm_Backward_Inp_kernel<scalar_t>
      <<<blocks, threads, 0, stream>>>(
      gradoutput, output, gradinput, gradgamma, gradbeta, ex, std,
      gamma, beta, gradEx, gradExs);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return {gradinput_, gradEx_, gradExs_, gradgamma_, gradbeta_};
}


std::vector<at::Tensor> BatchNorm_Backward_CUDA(
    const at::Tensor gradoutput_,
    const at::Tensor input_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps) {
  /* outputs*/
  auto std_ = (exs_ - ex_ * ex_ + eps).sqrt();
  auto gradinput_ = at::zeros_like(input_);
  auto gradgamma_ = at::zeros_like(gamma_);
  auto gradbeta_ = at::zeros_like(beta_);
  auto gradEx_ = at::zeros_like(ex_);
  auto gradExs_ = at::zeros_like(std_);
  /* cuda utils*/
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(input_.size(1));
  dim3 threads(getNumThreads(input_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(input_.type(), "BatchNorm_Inp_Backward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> gradoutput = devicetensor<scalar_t, 3>(gradoutput_);
    DeviceTensor<scalar_t, 3> input = devicetensor<scalar_t, 3>(input_);
    DeviceTensor<scalar_t, 3> gradinput = devicetensor<scalar_t, 3>(gradinput_);
    DeviceTensor<scalar_t, 1> gradgamma = devicetensor<scalar_t, 1>(gradgamma_);
    DeviceTensor<scalar_t, 1> gradbeta = devicetensor<scalar_t, 1>(gradbeta_);
    DeviceTensor<scalar_t, 1> ex = devicetensor<scalar_t, 1>(ex_);
    DeviceTensor<scalar_t, 1> std = devicetensor<scalar_t, 1>(std_);
    DeviceTensor<scalar_t, 1> gamma = devicetensor<scalar_t, 1>(gamma_);
    DeviceTensor<scalar_t, 1> beta = devicetensor<scalar_t, 1>(beta_);
    DeviceTensor<scalar_t, 1> gradEx = devicetensor<scalar_t, 1>(gradEx_);
    DeviceTensor<scalar_t, 1> gradExs = devicetensor<scalar_t, 1>(gradExs_);
    /* kernel function */
    BatchNorm_Backward_kernel<scalar_t>
      <<<blocks, threads, 0, stream>>>(
      gradoutput, input, gradinput, gradgamma, gradbeta, ex, std,
      gamma, beta, gradEx, gradExs);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return {gradinput_, gradEx_, gradExs_, gradgamma_, gradbeta_};
}

std::vector<at::Tensor> Expectation_Forward_CUDA(
    const at::Tensor input_) {
  /* outputs */
  auto ex_ = torch::zeros({input_.size(1)}, input_.options());
  auto exs_ = torch::zeros({input_.size(1)}, input_.options());
  /* cuda utils*/
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(input_.size(1));
  dim3 threads(getNumThreads(input_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(input_.type(), "SumSquare_forward_CUDA", ([&] {
    scalar_t norm = scalar_t(1) / (input_.size(0) * input_.size(2));
    /* Device tensors */
    DeviceTensor<scalar_t, 3> input = devicetensor<scalar_t, 3>(input_);
    DeviceTensor<scalar_t, 1> ex = devicetensor<scalar_t, 1>(ex_);
    DeviceTensor<scalar_t, 1> exs = devicetensor<scalar_t, 1>(exs_);
    /* kernel function */
    Expectation_Forward_kernel<scalar_t>
      <<<blocks, threads, 0, stream>>>(input, ex, exs, norm);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return {ex_, exs_};
}

at::Tensor Expectation_Backward_CUDA(
    const at::Tensor input_,
    const at::Tensor gradEx_,
    const at::Tensor gradExs_) {
  /* outputs */
  at::Tensor gradInput_ = at::zeros_like(input_);
  /* cuda utils*/
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(input_.size(1));
  dim3 threads(getNumThreads(input_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(input_.type(), "SumSquare_Backward_CUDA", ([&] {
    scalar_t norm = scalar_t(1) / (input_.size(0) * input_.size(2));
    /* Device tensors */
    DeviceTensor<scalar_t, 3> gradInput = devicetensor<scalar_t, 3>(gradInput_);
    DeviceTensor<scalar_t, 3> input = devicetensor<scalar_t, 3>(input_);
    DeviceTensor<scalar_t, 1> gradEx = devicetensor<scalar_t, 1>(gradEx_);
    DeviceTensor<scalar_t, 1> gradExs =devicetensor<scalar_t, 1>(gradExs_);
    /* kernel function */
    Expectation_Backward_kernel<scalar_t>
      <<<blocks, threads, 0, stream>>>(gradInput, input, gradEx, gradExs, norm);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return gradInput_;
}

at::Tensor Expectation_Inp_Backward_CUDA(
    const at::Tensor gradInput_,
    const at::Tensor output_,
    const at::Tensor gradEx_,
    const at::Tensor gradExs_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps) {
  /* outputs */
  //auto gradInput_ = at::zeros_like(output_);
  auto std_ = (exs_ - ex_ * ex_ + eps).sqrt();
  /* cuda utils*/
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(output_.size(1));
  dim3 threads(getNumThreads(output_.size(2)));
  AT_DISPATCH_FLOATING_TYPES(output_.type(), "SumSquare_Backward_CUDA", ([&] {
    scalar_t norm = scalar_t(1) / (output_.size(0) * output_.size(2));
    /* Device tensors */
    DeviceTensor<scalar_t, 3> gradInput = devicetensor<scalar_t, 3>(gradInput_);
    DeviceTensor<scalar_t, 3> input = devicetensor<scalar_t, 3>(output_);
    DeviceTensor<scalar_t, 1> gradEx = devicetensor<scalar_t, 1>(gradEx_);
    DeviceTensor<scalar_t, 1> gradExs =devicetensor<scalar_t, 1>(gradExs_);
    DeviceTensor<scalar_t, 1> ex = devicetensor<scalar_t, 1>(ex_);
    DeviceTensor<scalar_t, 1> std = devicetensor<scalar_t, 1>(std_);
    DeviceTensor<scalar_t, 1> gamma = devicetensor<scalar_t, 1>(gamma_);
    DeviceTensor<scalar_t, 1> beta = devicetensor<scalar_t, 1>(beta_);
    /* kernel function */
    Expectation_Backward_Inp_kernel<scalar_t>
      <<<blocks, threads, 0, stream>>>(gradInput, input, gradEx, gradExs,
          ex, std, gamma, beta, norm);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return gradInput_;
}