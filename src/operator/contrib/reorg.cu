/*!
 * Copyright (c) 2017 by Contributors
 * \file reorg.cu
 * \brief reorg layer
 * \author Bowen Cheng
 */
#include "./reorg-inl.h"

namespace mxnet {
  namespace op {

    template<>
    Operator* CreateOp<gpu>(ReorgParam param, int dtype) {
      Operator *op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new ReorgOp<gpu, DType>(param);
      })
      return op;
    }

  }  // namespace op
}  // namespace mxnet
