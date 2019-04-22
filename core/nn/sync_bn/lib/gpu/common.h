#include <cuda.h>
#include <cuda_runtime.h>

static const unsigned WARP_SIZE = 32;

// The maximum number of threads in a block
static const unsigned MAX_BLOCK_SIZE = 512U;

template<typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ __forceinline__ Out to(const In v) { return (Out) v; }
};

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ T values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

template <typename DType, typename Acctype>
struct Float2 {
  Acctype v1, v2;
  __device__ Float2() {}
  __device__ Float2(DType v1, DType v2) : v1(ScalarConvert<DType, Acctype>::to(v1)), v2(ScalarConvert<DType, Acctype>::to(v2)) {}
  __device__ Float2(DType v) : v1(ScalarConvert<DType, Acctype>::to(v)), v2(ScalarConvert<DType, Acctype>::to(v)) {}
  __device__ Float2(int v) : v1(ScalarConvert<int, Acctype>::to(v)), v2(ScalarConvert<int, Acctype>::to(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename DType, typename Acctype>
static __device__ __forceinline__ Float2<DType, Acctype> warpSum(Float2<DType, Acctype> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

template<typename T, typename Op>
__device__ T reduceD(
    Op op, int b, int i, int k, int D) {
  T sum = 0;
  for (int x = threadIdx.x; x < D; x += blockDim.x) {
      sum += op(b,i,k,x);
  }
  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];

  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
      if (threadIdx.x / WARP_SIZE < 32) {
              shared[threadIdx.x / WARP_SIZE] = sum;
      }
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
      // zero out the other entries in shared
      shared[threadIdx.x] = (T) 0;
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

template<typename T, typename Op>
__device__ T reduceN(
    Op op, int b, int k, int d, int N) {
  T sum = 0;
  for (int x = threadIdx.x; x < N; x += blockDim.x) {
      sum += op(b,x,k,d);
  }
  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];

  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
      if (threadIdx.x / WARP_SIZE < 32) {
              shared[threadIdx.x / WARP_SIZE] = sum;
      }
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
      // zero out the other entries in shared
      shared[threadIdx.x] = (T) 0;
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

template<typename T, typename Op>
__device__ T reduceK(
    Op op, int b, int i, int d, int K) {
  T sum = 0;
  for (int x = threadIdx.x; x < K; x += blockDim.x) {
    sum += op(b,i,x,d);
  }
  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];

  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    if (threadIdx.x / WARP_SIZE < 32) {
            shared[threadIdx.x / WARP_SIZE] = sum;
    }
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T) 0;
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

template<typename T, typename Op>
__device__ T reduceBN(
    Op op, 
    int k, int d, int B, int N) {
  T sum = 0;
  for (int batch = 0; batch < B; ++batch) {
    for (int x = threadIdx.x; x < N; x += blockDim.x) {
        sum += op(batch,x,k,d);
    }
  }
  // sum over NumThreads within a warp
  sum = warpSum(sum);
  // 'transpose', and reduce within warp again
  __shared__ T shared[32];

  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    if (threadIdx.x / WARP_SIZE < 32) {
            shared[threadIdx.x / WARP_SIZE] = sum;
    }
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T) 0;
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