/*
 * Written by Haozhi Qi
 */

#include "./label_converter-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

namespace mshadow {
  namespace cuda {
    template <typename DType>
    __global__ void LabelConverterForwardKernel(const int count,
                                                const DType* bottom_data,
                                                DType* top_data) {
      CUDA_KERNEL_LOOP(index, count) {
        // assume top is (n x 80), bottom is (n x 1)
        const int top_c = index % 80;
        const int top_n = index / 80;
        const int bottom_idx = top_n;
        const int source_label = static_cast<int>(bottom_data[bottom_idx]);
        // printf("%d\n", source_label);
        if (source_label == -1) {
          top_data[index] = -1;
        } else {
          if ((source_label - 1) == top_c) {
            top_data[index] = 1.0f;
          } else {
            top_data[index] = 0.0f;
          }
        }
      }
    }
  }  // namspace cuda


  template <typename DType>
  inline void LabelConverterForward(const Tensor<gpu, 2, DType> &data, 
                                    const Tensor<gpu, 2, DType> &out) {
    const DType* bottom_data = data.dptr_;
    DType* top_data = out.dptr_;
    const int count = out.shape_.Size();
    cudaStream_t stream = Stream<gpu>::GetStream(data.stream_);
    cuda::LabelConverterForwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
      cuda::kBaseThreadNum, 0, stream >> > (count, bottom_data, top_data); 
  }

}  // namespace mshadow


namespace mxnet {
  namespace op {
    template<>
    Operator* CreateOp<gpu>(LabelConverterParam param, int dtype) {
      Operator* op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new LabelConverter<gpu, DType>(param);
      });
      return op;
    }
  }  // namespace op
}  // namespace mxnet
