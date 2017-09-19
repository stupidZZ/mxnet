#include "./sigmoid_output-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

#define SigmoidOutput_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace mshadow {
  namespace cuda {

  template<typename DType>
    __global__ void SigmoidForwardKernel(
      const int count,
      const DType* bottom_data,
      DType* top_data) {
      CUDA_KERNEL_LOOP(index, count) {
        top_data[index] = DType(DType(1.0f) / (DType(1.0f) + expf(-bottom_data[index])));
      }
    }

  template<typename DType>
    inline void SigmoidForward(const Tensor<gpu, 2, DType> &out,
      const Tensor<gpu, 2, DType> &data) {
      // LOG(INFO) << "SigmoidForward";
      const DType *bottom_data = data.dptr_;
      DType *top_data = out.dptr_;
      const int count = out.shape_.Size();
      cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
      SigmoidForwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
        kBaseThreadNum, 0, stream >> >(count, bottom_data, top_data);
      SigmoidOutput_CUDA_CHECK(cudaPeekAtLastError());
    }

  template<typename DType>
    __global__ void SigmoidBackwardKernel(
      const int count,
      const DType* out_data,
	  const DType* out_label,
      DType* in_grad,
	  DType ignore_label) {
      CUDA_KERNEL_LOOP(index, count) {
      const int k = static_cast<int>(out_label[index]);
	  if(k == ignore_label)
		in_grad[index] = DType(0.0f);
	  else
		in_grad[index] = out_data[index] - out_label[index];
      }
    }
		
  template<typename DType>
    inline void SigmoidBackward(Tensor<gpu, 3, DType> &grad,
      const Tensor<gpu, 3, DType> &out,
      const Tensor<gpu, 2, DType> &label,
      const DType &ignore_label) {
      // LOG(INFO) << "SigmoidBackward";
      const DType *out_data = out.dptr_;
      const DType *out_label = label.dptr_;
	  DType *in_grad = grad.dptr_;
      const int count = out.shape_.Size();
      cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
      SigmoidBackwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
        kBaseThreadNum, 0, stream >> >(
          count, out_data, out_label, in_grad, ignore_label);
      SigmoidOutput_CUDA_CHECK(cudaPeekAtLastError());
    }
	
  template<typename DType>
	__global__ void SigmoidBackwardKernel(
      const int count,
      const DType* out_data,
	  const DType* out_label,
      DType* in_grad) {
      CUDA_KERNEL_LOOP(index, count) {
		in_grad[index] = out_data[index] - out_label[index];
      }
    }
		
  template<typename DType>
    inline void SigmoidBackward(Tensor<gpu, 3, DType> &grad,
      const Tensor<gpu, 3, DType> &out,
      const Tensor<gpu, 2, DType> &label) {
      // LOG(INFO) << "SigmoidBackward";
      const DType *out_data = out.dptr_;
      const DType *out_label = label.dptr_;
	  DType *in_grad = grad.dptr_;
      const int count = out.shape_.Size();
      cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
      SigmoidBackwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
        kBaseThreadNum, 0, stream >> >(
          count, out_data, out_label, in_grad);
      SigmoidOutput_CUDA_CHECK(cudaPeekAtLastError());
    }

  }  // namespace cuda

  template<typename DType>
  inline void SigmoidForward(const Tensor<gpu, 2, DType> &out,
      const Tensor<gpu, 2, DType> &data) {
    cuda::SigmoidForward(out, data);
  }
  
  template<typename DType>
  inline void SigmoidBackward(Tensor<gpu, 3, DType> &grad,
      const Tensor<gpu, 3, DType> &out,
      const Tensor<gpu, 2, DType> &label,
      const DType &ignore_label) {
    cuda::SigmoidBackward(grad, out, label, ignore_label);
  }
  
  template<typename DType>
  inline void SigmoidBackward(Tensor<gpu, 3, DType> &grad,
      const Tensor<gpu, 3, DType> &out,
      const Tensor<gpu, 2, DType> &label) {
    cuda::SigmoidBackward(grad, out, label);
  }

}  // namespace mshadow


namespace mxnet {
  namespace op {

    template<>
    Operator* CreateOp<gpu>(SigmoidOutputParam param, int dtype) {
      Operator* op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new SigmoidOutputOp<gpu, DType>(param);
      });
      return op;
    }

  }  // namespace op
}  // namespace mxnet