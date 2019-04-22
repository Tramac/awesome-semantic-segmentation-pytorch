#include <ATen/ATen.h>

template<typename DType, int Dim>
struct DeviceTensor {
 public:
  inline __device__ __host__ DeviceTensor(DType *p, const int *size)
    : dptr_(p) {
    for (int i = 0; i < Dim; ++i) {
      size_[i] = size ? size[i] : 0;
    }
  }

  inline __device__ __host__ unsigned getSize(const int i) const {
    assert(i < Dim);
    return size_[i];
  }

  inline __device__ __host__ int numElements() const {
    int n = 1;
    for (int i = 0; i < Dim; ++i) {
      n *= size_[i];
    }
    return n;
  }

  inline __device__ __host__ DeviceTensor<DType, Dim-1> select(const size_t x) const {
    assert(Dim > 1);
    int offset = x;
    for (int i = 1; i < Dim; ++i) {
      offset *= size_[i];
    }
    DeviceTensor<DType, Dim-1> tensor(dptr_ + offset, nullptr);
    for (int i = 0; i < Dim - 1; ++i) {
      tensor.size_[i] = this->size_[i+1];
    }
    return tensor;
  }

  inline __device__ __host__ DeviceTensor<DType, Dim-1> operator[](const size_t x) const {
    assert(Dim > 1);
    int offset = x;
    for (int i = 1; i < Dim; ++i) {
      offset *= size_[i];
    }
    DeviceTensor<DType, Dim-1> tensor(dptr_ + offset, nullptr);
    for (int i = 0; i < Dim - 1; ++i) {
      tensor.size_[i] = this->size_[i+1];
    }
    return tensor;
  }

  inline __device__ __host__ size_t InnerSize() const {
    assert(Dim >= 3);
    size_t sz = 1;
    for (size_t i = 2; i < Dim; ++i) {
      sz *= size_[i];
    }
    return sz;
  }

  inline __device__ __host__ size_t ChannelCount() const {
    assert(Dim >= 3);
    return size_[1];
  }

  inline __device__ __host__ DType* data_ptr() const {
    return dptr_;
  }

  DType *dptr_;
  int size_[Dim];
};

template<typename DType>
struct DeviceTensor<DType, 1> {
  inline __device__ __host__ DeviceTensor(DType *p, const int *size)
    : dptr_(p) {
    size_[0] = size ? size[0] : 0;
  }

  inline __device__ __host__ unsigned getSize(const int i) const {
    assert(i == 0);
    return size_[0];
  }

  inline __device__ __host__ int numElements() const {
    return size_[0];
  }

  inline __device__ __host__ DType &operator[](const size_t x) const {
      return *(dptr_ + x);
  }

  inline __device__ __host__ DType* data_ptr() const {
    return dptr_;
  }

  DType *dptr_;
  int size_[1];
};

template<typename DType, int Dim>
static DeviceTensor<DType, Dim> devicetensor(const at::Tensor &blob) {
  DType *data = blob.data<DType>();
  DeviceTensor<DType, Dim> tensor(data, nullptr);
  for (int i = 0; i < Dim; ++i) {
    tensor.size_[i] = blob.size(i);
  }
  return tensor;
}