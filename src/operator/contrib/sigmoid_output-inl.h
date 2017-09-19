
#ifndef MXNET_OPERATOR_SIGMOID_OUTPUT_INL_H_
#define MXNET_OPERATOR_SIGMOID_OUTPUT_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"

namespace mxnet {
namespace op {

namespace sigmoidout_enum {
enum SigmoidOutputOpInputs {kData, kLabel};
enum SigmoidOutputOpOutputs {kOut};
enum SigmoidOutputNormType {kNull, kBatch, kValid};
enum SigmoidOutputOpResource {kTempSpace};
}  // namespace sigmoidout_enum

struct SigmoidOutputParam : public dmlc::Parameter<SigmoidOutputParam> {
  float grad_scale;
  float ignore_label;
  bool use_ignore;
  int normalization;
  bool out_grad;
  DMLC_DECLARE_PARAMETER(SigmoidOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("the labels with value equals to ``ignore_label`` will be ignored "
              "during backward (only works if "
              "use_ignore is set to be true).");
    DMLC_DECLARE_FIELD(use_ignore).set_default(false)
    .describe("If set to true, the ignore_label value will not contribute "
      "to the backward gradient");
    DMLC_DECLARE_FIELD(normalization)
    .add_enum("null", sigmoidout_enum::kNull)
    .add_enum("batch", sigmoidout_enum::kBatch)
    .add_enum("valid", sigmoidout_enum::kValid)
    .set_default(sigmoidout_enum::kNull)
    .describe("Normalize the gradient");
    DMLC_DECLARE_FIELD(out_grad)
    .set_default(false)
    .describe("Apply weighting from output gradient");
  };
};

template<typename xpu, typename DType>
class SigmoidOutputOp : public Operator {
 public:
  explicit SigmoidOutputOp(SigmoidOutputParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U) << "SigmoidOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 1U) << "SigmoidOutput Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[sigmoidout_enum::kData].size(0);
    int k = in_data[sigmoidout_enum::kData].Size()/n;
    Shape<2> s2 = Shape2(n, k);
    Tensor<xpu, 2, DType> data =
        in_data[sigmoidout_enum::kData].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 2, DType> out =
        out_data[sigmoidout_enum::kOut].get_with_shape<xpu, 2, DType>(s2, s);
    SigmoidForward(out, data);
	}

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_GE(in_grad.size(), 1U);
    CHECK_GE(req.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
	int n = out_data[sigmoidout_enum::kOut].size(0);
	int k = out_data[sigmoidout_enum::kOut].size(1);
	Shape<3> s3 = Shape3(n, k, static_cast<int>(out_data[sigmoidout_enum::kOut].Size()/n/k));
	Shape<2> s2 = Shape2(s3[0], s3[2]);
	Tensor<xpu, 2, DType> label = in_data[sigmoidout_enum::kLabel].get_with_shape<xpu, 2, DType>(s2, s);
	Tensor<xpu, 3, DType> out = out_data[sigmoidout_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> grad = in_grad[sigmoidout_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);
    
	index_t valid_cnt = label.shape_.Size();
      if (param_.use_ignore) {
          SigmoidBackward(grad, out, label, static_cast<DType>(param_.ignore_label));
      } else {
          SigmoidBackward(grad, out, label);
      }
      if (param_.normalization == sigmoidout_enum::kBatch) {
        valid_cnt = label.size(0);
      } else if (param_.normalization == sigmoidout_enum::kValid) {
        int i_label = static_cast<int>(param_.ignore_label);
        Tensor<cpu, 2, DType> workspace = ctx.requested[sigmoidout_enum::kTempSpace].get_host_space_typed<2, DType>(
          label.shape_);
        Copy(workspace, label, label.stream_);
        for (index_t i = 0; i < workspace.size(0); ++i) {
          for (index_t j = 0; j < workspace.size(1); ++j) {
            if (static_cast<int>(workspace[i][j]) == i_label) {
              valid_cnt--;
            }
          }
        }
        valid_cnt = valid_cnt == 0 ? 1 : valid_cnt;
      } else {
        valid_cnt = 1;
      }
      grad *= DType(param_.grad_scale /
                    (param_.normalization == sigmoidout_enum::kValid ? 1 : s3[2]) /
                    valid_cnt);
      if (param_.out_grad) {
        Tensor<xpu, 3, DType> ograd =
          out_grad[sigmoidout_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
        grad *= ograd;
      } 
  }

 private:
  SigmoidOutputParam param_;
};  // class SigmoidOutputOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SigmoidOutputParam param, int dtype);

#if DMLC_USE_CXX11
class SigmoidOutputProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;

    // label.shape == data.shape: use probability as label
    if (dshape != (*in_shape)[sigmoidout_enum::kLabel]) {
      LOG(FATAL) << "Data and label have different shapes!" << dshape << " v.s. " << (*in_shape)[sigmoidout_enum::kLabel];
	  return false;
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SigmoidOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_SigmoidOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.out_grad) {
      return {in_data[sigmoidout_enum::kLabel], out_data[sigmoidout_enum::kOut],
              out_grad[sigmoidout_enum::kOut]};
    } else {
      return {in_data[sigmoidout_enum::kLabel], out_data[sigmoidout_enum::kOut]};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[sigmoidout_enum::kOut], in_grad[sigmoidout_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[sigmoidout_enum::kData], out_data[sigmoidout_enum::kOut]}};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 protected:
  SigmoidOutputParam param_;
};  // class SigmoidOutputProp


#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SIGMOID_OUTPUT_INL_H_
