#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#define PSA_TYPE_COLLECT 1
#define PSA_TYPE_DISTRIBUTE 2

const int CUDA_NUM_THREADS = 512;

inline int GET_BLOCKS(const int N) {
   return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void psa_collect_forward_kernel(const float *h, float *out, int num, int height, int width) {
    const int out_h = 2 * height - 1;
    const int out_w = 2 * width - 1;
    const int half_out_h = (out_h - 1) / 2;
    const int half_out_w = (out_w - 1) / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = num * height * width;

    for (int i = x; i < nthreads; i += blockDim.x * gridDim.x) {
        const int w = i % width;
        const int h = (i / width) % height;
        const int n = i / width / height;

        // effective mask region : [hstart, hend) x [wstart, wend) with out-indexed
        const int hstart = max(0, half_out_height - h);
        const int hend = min(out_h, height + half_out_h - h);
        const int wstart = max(0, half_out_width - w);
        const int wend = min(out_w, width + half_out_w - w);

        // (hidx, widx) with out-indexed
        // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
            for (int widx = wstart; widx < wend; widx++) {
                out[(n * height * width + (hidx + h - half_out_h) * width + (widx + w - half_out_w)) * height * width + h * width + w] =
                    h[((n * out_h * out_w + hidx * out_w + widx) * height + h) * width + w];
            }
        }
    }
}

template <typename T>
__global__ void psa_distribute_forward_kernel(const float *h, float *out, int num, int height, int width) {
    const int out_h = 2 * height - 1;
    const int out_w = 2 * width - 1;
    const int half_out_h = (out_h - 1) / 2;
    const int half_out_w = (out_w - 1) / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = num * height * width;

    for (int i = x; i < nthreads; i += blockDim.x * gridDim.x) {
        const int w = i % width;
        const int h = (i / width) % height;
        const int n = i / width / height;

        // effective mask region : [hstart, hend) x [wstart, wend) with out-indexed
        const int hstart = max(0, half_out_height - h);
        const int hend = min(out_h, height + half_out_h - h);
        const int wstart = max(0, half_out_width - w);
        const int wend = min(out_w, width + half_out_w - w);

        // (hidx, widx) with out-indexed
        // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
            for (int widx = wstart; widx < wend; widx++) {
                out[(n * height * width + h * width + w) * height * width + (hidx + h - half_out_h) * width + (widx + w - half_out_w)] =
                    h[((n * out_h * out_w + hidx * out_w + widx) * height + h) * width + w];
            }
        }
    }
}

template <typename T>
__global__ void psa_collect_backward_kernel(const float *dout, float *dh, int num, int height, int width) {
    const int out_h = 2 * height - 1;
    const int out_w = 2 * width - 1;
    const int half_out_h = (out_h - 1) / 2;
    const int half_out_w = (out_w - 1) / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = num * height * width;

    for (int i = x; i < nthreads; i += blockDim.x * gridDim.x) {
        const int w = i % width;
        const int h = (i / width) % height;
        const int n = i / width / height;

        // effective mask region : [hstart, hend) x [wstart, wend) with out-indexed
        const int hstart = max(0, half_out_height - h);
        const int hend = min(out_h, height + half_out_h - h);
        const int wstart = max(0, half_out_width - w);
        const int wend = min(out_w, width + half_out_w - w);

        // (hidx, widx) with out-indexed
        // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
            for (int widx = wstart; widx < wend; widx++) {
                dh[((h * out_h * out_w + hidx * out_w + widx) * height + h) * width + w] =
                    dout[(n * height * width + (hidx + h - half_out_h) * width + (widx + w - half_out_w)) * height * width + h * width + w];
            }
        }
    }
}

template <typename T>
__global__ void psa_distribute_backward_kernel(const float *dout, float *dh, int num, int height, int width) {
    const int out_h = 2 * height - 1;
    const int out_w = 2 * width - 1;
    const int half_out_h = (out_h - 1) / 2;
    const int half_out_w = (out_w - 1) / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = num * height * width;

    for (int i = x; i < nthreads; i += blockDim.x * gridDim.x) {
        const int w = i % width;
        const int h = (i / width) % height;
        const int n = i / width / height;

        // effective mask region : [hstart, hend) x [wstart, wend) with out-indexed
        const int hstart = max(0, half_out_height - h);
        const int hend = min(out_h, height + half_out_h - h);
        const int wstart = max(0, half_out_width - w);
        const int wend = min(out_w, width + half_out_w - w);

        // (hidx, widx) with out-indexed
        // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
            for (int widx = wstart; widx < wend; widx++) {
                dh[((n * out_h * out_w + hidx * out_w + widx) * height + h) * width + w] =
                    dout[(n * height * width + h * width + w) * height * width + (hidx + h - half_out_h) * width + (widx + w - half_out_w)];
            }
        }
    }
}

at::Tensor psa_forward_cuda(const at::Tensor& h, const int forward_type) {
    AT_ASSERTM(h.type().is_cuda(), "input must be a CUDA tensor");

    auto n = h.size(0);
    auto c = h.size(1);
    auto h = h.size(2);
    auto w = h.size(3);

    at::Tensor out = at::zeros({n, h * w, h * w}, h.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int nthreads = n * h * w;

    switch (forward_type) {
    case PSA_TYPE_COLLECT:
        AT_DISPATCH_FLOATING_TYPES(h.type(), "psa_forward", [&] {
            psa_collect_forward_kernel<scalar_t><<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, stream>>>(
                h.contiguous().data<scalar_t>(),
                out.contiguous().data<scalar_t>(),
                n, h, w);
        });
        break;
    case PSA_TYPE_DISTRIBUTE:
        AT_DISPATCH_FLOATING_TYPES(h.type(), "psa_forward", [&] {
            psa_distribute_forward_kernel<scalar_t><<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, stream>>>(
                h.contiguous().data<scalar_t>(),
                out.contiguous().data<scalar_t>(),
                n, h, w);
        });
        break;
    }
    THCudaCheck(cudaGetLastError());
    return out;
}

at::Tensor psa_backward_cuda(const at::Tensor& dout, const at::Tensor& h, const int forward_type) {
    AT_ASSERTM(dout.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(h.type().is_cuda(), "input must be a CUDA tensor");

    auto n = h.size(0);
    auto c = h.size(1);
    auto h = h.size(2);
    auto w = h.size(3);

    at::Tensor dh = at::zeros_like(h);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int nthreads = n * h * w;

    switch (forward_type) {
    case PSA_TYPE_COLLECT:
        AT_DISPATCH_FLOATING_TYPES(h.type(), "psa_backward", [&] {
            psa_collect_backward_kernel<scalar_t><<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, stream>>>(
                dout.contiguous().data<scalar_t>(),
                dh.contiguous().data<scalar_t>(),
                n, h, w);
        });
        break;
    case PSA_TYPE_DISTRIBUTE:
        AT_DISPATCH_FLOATING_TYPES(h.type(), "psa_backward", [&] {
            psa_distribute_backward_kernel<scalar_t><<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, stream>>>(
                dout.contiguous().data<scalar_t>(),
                dh.contiguous().data<scalar_t>(),
                n, h, w);
        });
        break;
    }
    THCudaCheck(cudaGetLastError());
    return dh;
}