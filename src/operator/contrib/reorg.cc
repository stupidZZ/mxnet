/*!
 * Copyright (c) 2017 by Contributors
 * \file reorg.cc
 * \brief reorg op
 * \author Bowen Cheng
 */
#include "./reorg-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(ReorgParam param, int dtype) {
      Operator *op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new ReorgOp<cpu, DType>(param);
        });
      return op;
    }

    Operator *ReorgProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
        std::vector<int> *in_type) const {
      std::vector<TShape> out_shape, aux_shape;
      std::vector<int> out_type, aux_type;
      CHECK(InferType(in_type, &out_type, &aux_type));
      CHECK(InferShape(in_shape, &out_shape, &aux_shape));
      DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
    }

    DMLC_REGISTER_PARAMETER(ReorgParam);

    MXNET_REGISTER_OP_PROPERTY(_contrib_Reorg, ReorgProp)
      .describe("Reorg layer. Reorganize a tensor of shape (NxHxWxA, C, k, k) to shape (N, AxCxkxk, H, W)")
      .add_argument("data", "NDArray-or-Symbol", "Input data to the ReorgOp.")
      .add_arguments(ReorgParam::__FIELDS__());
  }  // namespace op
}  // namespace mxnet
