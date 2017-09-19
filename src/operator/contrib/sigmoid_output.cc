#include "./sigmoid_output-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {

  template<typename DType>
  inline void SigmoidForward(const Tensor<cpu, 2, DType> &out,
      const Tensor<cpu, 2, DType> &data) {
    // NOT_IMPLEMENTED;
    return;
  }
  
  template<typename DType>
  inline void SigmoidBackward(Tensor<cpu, 3, DType> &grad,
      const Tensor<cpu, 3, DType> &out,
      const Tensor<cpu, 2, DType> &label,
      const DType &ignore_label) {
    // NOT_IMPLEMENTED;
    return;
  }
  
  template<typename DType>
  inline void SigmoidBackward(Tensor<cpu, 3, DType> &grad,
      const Tensor<cpu, 3, DType> &out,
      const Tensor<cpu, 2, DType> &label) {
    // NOT_IMPLEMENTED;
    return;
  }
}  // namespace mshadow

namespace mxnet {
  namespace op {

    template<>
    Operator *CreateOp<cpu>(SigmoidOutputParam param, int dtype) {
      Operator* op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new SigmoidOutputOp<cpu, DType>(param);
      });
      return op;
    }

    Operator *SigmoidOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const {
      std::vector<TShape> out_shape, aux_shape;
      std::vector<int> out_type, aux_type;
      CHECK(InferType(in_type, &out_type, &aux_type));
      CHECK(InferShape(in_shape, &out_shape, &aux_shape));
      DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
    }

    DMLC_REGISTER_PARAMETER(SigmoidOutputParam);

    MXNET_REGISTER_OP_PROPERTY(_contrib_SigmoidOutput, SigmoidOutputProp)
      .describe("Sigmoid with binary entropy loss, supporting ignore label. ")
      .add_argument("data", "NDArray-or-Symbol", "Input data.")
	  .add_argument("label", "NDArray-or-Symbol", "Ground truth label.")
      .add_arguments(SigmoidOutputParam::__FIELDS__());
  }  // namespace op
}  // namespace mxnet