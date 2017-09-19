#include "./bilinear_resize-inl.h"
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
  template <typename DType>
  inline void BilinearResizeForward(const Tensor<cpu, 4, DType> &out,
    const Tensor<cpu, 4, DType> &data) {
    return;
  }


  template <typename DType>
  inline void BilinearResizeBackward(const Tensor<cpu, 4, DType> &in_grad,
    const Tensor<cpu, 4, DType> &data,
    const Tensor<cpu, 4, DType> &out_grad) {
    return;
  }

}


namespace mxnet {
  namespace op {

    template<>
    Operator *CreateOp<cpu>(BilinearResizeParam param, int dtype) {
      Operator* op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new BilinearResizeOp<cpu, DType>(param);
      });
      return op;
    }

    Operator *BilinearResizeProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const {
      std::vector<TShape> out_shape, aux_shape;
      std::vector<int> out_type, aux_type;
      CHECK(InferType(in_type, &out_type, &aux_type));
      CHECK(InferShape(in_shape, &out_shape, &aux_shape));
      DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
    }

    DMLC_REGISTER_PARAMETER(BilinearResizeParam);

    MXNET_REGISTER_OP_PROPERTY(_contrib_BilinearResize, BilinearResizeProp)
      .describe("Bilinear Resizeing")
      .add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
      .add_argument("data_ref", "Symbol", "Reference data")
      .add_arguments(BilinearResizeParam::__FIELDS__());
  }  // namespace op
}  // namespace mxnet
