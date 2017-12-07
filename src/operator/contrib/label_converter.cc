/*
 * Written by Haozhi Qi
 */

#include "./label_converter-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

namespace mshadow {

  template<typename DType>
  inline void LabelConverterForward(const Tensor<cpu, 2, DType> &data,
                                    const Tensor<cpu, 2, DType> &out) {
    // NOT_IMPLEMENTED;
    return;
  }
}  // namespace mshadow


namespace mxnet {
  namespace op {

    template<>
    Operator *CreateOp<cpu>(LabelConverterParam param, int dtype) {
      Operator* op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new LabelConverter<cpu, DType>(param);
      });
      return op;
    }

    Operator *LabelConverterProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const {
      std::vector<TShape> out_shape, aux_shape;
      std::vector<int> out_type, aux_type;
      CHECK(InferType(in_type, &out_type, &aux_type));
      CHECK(InferShape(in_shape, &out_shape, &aux_shape));
      DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
    }

    DMLC_REGISTER_PARAMETER(LabelConverterParam);

    MXNET_REGISTER_OP_PROPERTY(_contrib_LabelConverter, LabelConverterProp)
      .describe("label converter ")
      .add_argument("data", "NDArray-or-Symbol", "Input data.")
      .add_arguments(LabelConverterParam::__FIELDS__());
  }  // namespace op
}  // namespace mxnet