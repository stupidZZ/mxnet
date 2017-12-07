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
    __global__ void SigmoidForwardKernel(const int count,
                                         DType* out_data,
                                         DType* loss_data,
                                         const DType* bottom_data,
                                         const DType* bottom_label,
                                         const DType* dynamic_normalizer,
                                         const int ignore_label) {
      CUDA_KERNEL_LOOP(index, count) {
        out_data[index] = DType(DType(1.0f) / (DType(1.0f) + expf(-bottom_data[index])));
        const int target_value = static_cast<int>(bottom_label[index]);
        if (ignore_label == target_value) {
          loss_data[index] = 0.0f;
        } else {
          const DType loss_val = bottom_data[index] * (bottom_label[index] - (bottom_data[index] >= 0)) - 
                                 log(1 + exp(bottom_data[index] - 2 * bottom_data[index] * (bottom_data[index] >= 0)));
          loss_data[index] = -1 * loss_val / dynamic_normalizer[0];
        }
      }
    }

    /* 
     * sigmoid forward with focal loss support
     */
    template<typename DType>
    __global__ void SigmoidForwardKernel(const int count,
                                         DType* out_data,
                                         DType* loss_data,
                                         const DType* bottom_data,
                                         const DType* bottom_label,
                                         const DType* dynamic_normalizer,
                                         const float alpha,
                                         const float gamma,
                                         const int ignore_label) {
      CUDA_KERNEL_LOOP(index, count) {
        // out data remains the same under focal loss setting
        out_data[index] = DType(DType(1.0f) / (DType(1.0f) + expf(-bottom_data[index])));
        const int target_value = static_cast<int>(bottom_label[index]);
        if (ignore_label == target_value) {
          loss_data[index] = 0.0f;
        } else {
          const DType p = DType(DType(1.0f) / (DType(1.0f) + expf(-bottom_data[index])));
          const DType pt = target_value == 0 ? 1.0f - p : p;
          const DType alpha_t = target_value == 0 ? 1.0f - alpha : alpha;
          const DType loss_val = alpha_t * pow(1 - pt, gamma) * log(pt);
          loss_data[index] = -1 * loss_val / dynamic_normalizer[0];
        }
      }
    }

    /*
     * sigmoid backward kernel with focal loss support
     * note the gradient normalization doesn't happen in backward
     */
    template<typename DType>
    __global__ void SigmoidBackwardKernel(const int count,
                                          const DType* out_data,
                                          const DType* out_label,
                                          DType* in_grad,
                                          const float alpha,
                                          const float gamma,
                                          const int ignore_label) {
      CUDA_KERNEL_LOOP(index, count) {
        const int target_label = static_cast<int>(out_label[index]);
        if (ignore_label == target_label) {
          in_grad[index] = static_cast<DType>(0.0f);
        } else {
          const DType p = out_data[index];
          const DType pt = target_label == 0 ? 1.0f - p : p;
          const DType alpha_t = target_label == 0 ? 1.0f - alpha : alpha;
          const DType label_scalar = target_label == 0 ? -1.0f : 1.0f;
          in_grad[index] = label_scalar * alpha_t * pow(1.0f - pt, gamma) * (gamma * pt * log(pt) + pt - 1);
        }
      }
    }


    template<typename DType>
    __global__ void SigmoidBackwardKernel(const int count,
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

  template <typename DType>
  inline void SigmoidForward(const Tensor<gpu, 2, DType> &out,
                             const Tensor<gpu, 2, DType> &loss,
                             const Tensor<gpu, 2, DType> &data,
                             const Tensor<gpu, 2, DType> &label,
                             const Tensor<gpu, 1, DType> &dynamic_normalizer,
                             const int &ignore_label) {
    DType* out_data = out.dptr_;
    DType* loss_data = loss.dptr_;
    const DType* bottom_data = data.dptr_;
    const DType* bottom_label = label.dptr_;
    const DType* dy_norm = dynamic_normalizer.dptr_;
    const int count = out.shape_.Size();
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    cuda::SigmoidForwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
      cuda::kBaseThreadNum, 0, stream >> >(count, out_data, loss_data, bottom_data, bottom_label, 
        dy_norm, ignore_label);
    SigmoidOutput_CUDA_CHECK(cudaPeekAtLastError());
  }

  /*
   * wrapper for cuda kernel: sigmoid forward with focal loss support
   */
  template <typename DType>
  inline void SigmoidForward(const Tensor<gpu, 2, DType> &out,
                             const Tensor<gpu, 2, DType> &loss,
                             const Tensor<gpu, 2, DType> &data,
                             const Tensor<gpu, 2, DType> &label,
                             const Tensor<gpu, 1, DType> &dynamic_normalizer,
                             const float alpha,
                             const float gamma,
                             const int &ignore_label) {
    DType* out_data = out.dptr_;
    DType* loss_data = loss.dptr_;
    const DType* bottom_data = data.dptr_;
    const DType* bottom_label = label.dptr_;
    const DType* dy_norm = dynamic_normalizer.dptr_;
    const int count = out.shape_.Size();
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    cuda::SigmoidForwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
      cuda::kBaseThreadNum, 0, stream >> >(count, out_data, loss_data, bottom_data, bottom_label, 
        dy_norm, alpha, gamma, ignore_label);
    SigmoidOutput_CUDA_CHECK(cudaPeekAtLastError());
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

  template <typename DType>
  inline void SigmoidBackward(const Tensor<gpu, 3, DType> &grad,
                              const Tensor<gpu, 3, DType> &out,
                              const Tensor<gpu, 2, DType> &label,
                              const float alpha,
                              const float gamma,
                              const int &ignore_label) {
    const DType* out_data = out.dptr_;
    const DType* out_label = label.dptr_;
    DType *in_grad = grad.dptr_;
    const int count = out.shape_.Size();
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    cuda::SigmoidBackwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
      cuda::kBaseThreadNum, 0, stream >> >(count, out_data, out_label, in_grad, alpha, gamma, ignore_label);
    SigmoidOutput_CUDA_CHECK(cudaPeekAtLastError());
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