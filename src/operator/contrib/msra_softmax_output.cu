/*!
 * Copyright (c) 2015 by Contributors
 * \file msra_softmax_output.cu
 * \brief
 * \author Zheng Zhang, Xizhou Zhu
*/

#include "./msra_softmax_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MsraVcSoftmaxOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MsraVcSoftmaxOutputOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

