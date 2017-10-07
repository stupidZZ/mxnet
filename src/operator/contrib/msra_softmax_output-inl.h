/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output-inl.h
 * \brief
 * \author Zheng Zhang, Xizhou zhu
*/
#ifndef MXNET_OPERATOR_MSRAVC_SOFTMAX_OUTPUT_INL_H_
#define MXNET_OPERATOR_MSRAVC_SOFTMAX_OUTPUT_INL_H_

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

namespace msravc_softmaxout_enum {
enum MsraVcMsraVcSoftmaxOutputOpInputs {kData, kLabel};
enum MsraVcMsraVcSoftmaxOutputOpOutputs {kOut};
enum MsraVcSoftmaxOutputNormType {kAbsoluteNull, kNull, kBatch, kValid};
enum MsraVcMsraVcSoftmaxOutputOpResource {kTempSpace};
}  // namespace msravc_softmaxout_enum

struct MsraVcSoftmaxOutputParam : public dmlc::Parameter<MsraVcSoftmaxOutputParam> {
  float grad_scale;
  float ignore_label;
  bool multi_output;
  bool use_ignore;
  bool preserve_shape;
  int normalization;
  bool out_grad;
  DMLC_DECLARE_PARAMETER(MsraVcSoftmaxOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("the labels with value equals to ``ignore_label`` will be ignored "
              "during backward (only works if "
              "use_ignore is set to be true).");
    DMLC_DECLARE_FIELD(multi_output).set_default(false)
    .describe("If set to true, softmax will applied on axis 1");
    DMLC_DECLARE_FIELD(use_ignore).set_default(false)
    .describe("If set to true, the ignore_label value will not contribute "
      "to the backward gradient");
    DMLC_DECLARE_FIELD(preserve_shape).set_default(false)
    .describe("If true, softmax will applied on the last axis");
    DMLC_DECLARE_FIELD(normalization)
    .add_enum("absolute_null", msravc_softmaxout_enum::kAbsoluteNull)
    .add_enum("null", msravc_softmaxout_enum::kNull)
    .add_enum("batch", msravc_softmaxout_enum::kBatch)
    .add_enum("valid", msravc_softmaxout_enum::kValid)
    .set_default(msravc_softmaxout_enum::kNull)
    .describe("Normalize the gradient");
    DMLC_DECLARE_FIELD(out_grad)
    .set_default(false)
    .describe("Apply weighting from output gradient");
  };
};

template<typename xpu, typename DType>
class MsraVcSoftmaxOutputOp : public Operator {
 public:
  explicit MsraVcSoftmaxOutputOp(MsraVcSoftmaxOutputParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U) << "MsraVcSoftmaxOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 1U) << "MsraVcSoftmaxOutput Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.multi_output) {
      int n = in_data[msravc_softmaxout_enum::kData].size(0);
      int k = in_data[msravc_softmaxout_enum::kData].size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[msravc_softmaxout_enum::kData].Size()/n/k));
      Tensor<xpu, 3, DType> data =
          in_data[msravc_softmaxout_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);
      Tensor<xpu, 3, DType> out =
          out_data[msravc_softmaxout_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
      Softmax(out, data);
    } else {
      if (param_.preserve_shape) {
        Tensor<xpu, 2, DType> data = in_data[msravc_softmaxout_enum::kData].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> out = out_data[msravc_softmaxout_enum::kOut].FlatTo2D<xpu, DType>(s);
        Softmax(out, data);
      } else {
        int n = in_data[msravc_softmaxout_enum::kData].size(0);
        int k = in_data[msravc_softmaxout_enum::kData].Size()/n;
        Shape<2> s2 = Shape2(n, k);
        Tensor<xpu, 2, DType> data =
            in_data[msravc_softmaxout_enum::kData].get_with_shape<xpu, 2, DType>(s2, s);
        Tensor<xpu, 2, DType> out =
            out_data[msravc_softmaxout_enum::kOut].get_with_shape<xpu, 2, DType>(s2, s);
        Softmax(out, data);
      }
    }
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

    if (out_data[msravc_softmaxout_enum::kOut].shape_ ==
        in_data[msravc_softmaxout_enum::kLabel].shape_) {
      // use probability as label
      Tensor<xpu, 2, DType> label = in_data[msravc_softmaxout_enum::kLabel].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> out = out_data[msravc_softmaxout_enum::kOut].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> grad = in_grad[msravc_softmaxout_enum::kData].FlatTo2D<xpu, DType>(s);
      if (param_.out_grad) {
        Tensor<xpu, 2, DType> ograd = out_grad[msravc_softmaxout_enum::kOut].FlatTo2D<xpu, DType>(s);
        grad = scalar<DType>(param_.grad_scale) * (out - label) * ograd;
      } else {
        grad = (out - label) * scalar<DType>(param_.grad_scale);
      }
    } else if (param_.multi_output) {
      int n = out_data[msravc_softmaxout_enum::kOut].size(0);
      int k = out_data[msravc_softmaxout_enum::kOut].size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<int>(out_data[msravc_softmaxout_enum::kOut].Size()/n/k));
      Shape<2> s2 = Shape2(s3[0], s3[2]);
      Tensor<xpu, 2, DType> label =
          in_data[msravc_softmaxout_enum::kLabel].get_with_shape<xpu, 2, DType>(s2, s);
      Tensor<xpu, 3, DType> out =
          out_data[msravc_softmaxout_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
      Tensor<xpu, 3, DType> grad =
          in_grad[msravc_softmaxout_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);

      index_t valid_cnt = label.shape_.Size();
      if (param_.use_ignore) {
          SoftmaxGrad(grad, out, label, static_cast<DType>(param_.ignore_label));
      } else {
          SoftmaxGrad(grad, out, label);
      }
      if (param_.normalization == msravc_softmaxout_enum::kBatch) {
        valid_cnt = label.size(0);
      } else if (param_.normalization == msravc_softmaxout_enum::kValid) {
        int i_label = static_cast<int>(param_.ignore_label);
        Tensor<cpu, 2, DType> workspace =
          ctx.requested[msravc_softmaxout_enum::kTempSpace].get_host_space_typed<2, DType>(
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
                    ((param_.normalization == msravc_softmaxout_enum::kAbsoluteNull ||
                      param_.normalization == msravc_softmaxout_enum::kValid) ? 1 : s3[2]) /
                    valid_cnt);
      if (param_.out_grad) {
        Tensor<xpu, 3, DType> ograd =
          out_grad[msravc_softmaxout_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
        grad *= ograd;
      }
    } else {
      Shape<1> label_shape = Shape1(in_data[msravc_softmaxout_enum::kLabel].Size());
      Shape<2> data_shape;
      if (param_.preserve_shape) {
        data_shape = out_data[msravc_softmaxout_enum::kOut].shape_.FlatTo2D();
//        Tensor<xpu, 1, DType> label = in_data[msravc_softmaxout_enum::kLabel].FlatTo1D<xpu, DType>(s);
//        Tensor<xpu, 2, DType> out = out_data[msravc_softmaxout_enum::kOut].FlatTo2D<xpu, DType>(s);
//        Tensor<xpu, 2, DType> grad = in_grad[msravc_softmaxout_enum::kData].FlatTo2D<xpu, DType>(s);
      } else {
        int n = out_data[msravc_softmaxout_enum::kOut].size(0);
        data_shape = Shape2(n, out_data[msravc_softmaxout_enum::kOut].Size()/n);
      }
      Tensor<xpu, 1, DType> label = in_data[msravc_softmaxout_enum::kLabel].get_with_shape<xpu, 1, DType>(
          label_shape, s);
      Tensor<xpu, 2, DType> out =
          out_data[msravc_softmaxout_enum::kOut].get_with_shape<xpu, 2, DType>(data_shape, s);
      Tensor<xpu, 2, DType> grad =
          in_grad[msravc_softmaxout_enum::kData].get_with_shape<xpu, 2, DType>(data_shape, s);
      index_t valid_cnt = label.shape_.Size();
      if (param_.use_ignore) {
        SoftmaxGrad(grad, out, label, static_cast<DType>(param_.ignore_label));
      } else {
        SoftmaxGrad(grad, out, label);
      }
      if (param_.normalization == msravc_softmaxout_enum::kBatch) {
        valid_cnt = label.size(0);
      } else if (param_.normalization == msravc_softmaxout_enum::kValid) {
        int i_label = static_cast<int>(param_.ignore_label);
        Tensor<cpu, 1, DType> workspace =
          ctx.requested[msravc_softmaxout_enum::kTempSpace].get_host_space_typed<1, DType>(
          label.shape_);
        Copy(workspace, label, label.stream_);
        for (index_t i = 0; i < label.size(0); ++i) {
          if (static_cast<int>(workspace[i]) == i_label) {
            valid_cnt--;
          }
        }
        valid_cnt = valid_cnt == 0 ? 1 : valid_cnt;
      } else {
        valid_cnt = 1;
      }
      grad *= DType(param_.grad_scale / valid_cnt);
      if (param_.out_grad) {
        Tensor<xpu, 2, DType> ograd =
          out_grad[msravc_softmaxout_enum::kOut].get_with_shape<xpu, 2, DType>(data_shape, s);
        grad *= ograd;
      }
    }
  }

 private:
  MsraVcSoftmaxOutputParam param_;
};  // class MsraVcSoftmaxOutputOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(MsraVcSoftmaxOutputParam param, int dtype);

#if DMLC_USE_CXX11
class MsraVcSoftmaxOutputProp : public OperatorProperty {
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
    if (dshape != (*in_shape)[msravc_softmaxout_enum::kLabel]) {
      if (param_.multi_output) {
        TShape lshape1 = Shape2(dshape[0], dshape.Size()/dshape[0]/dshape[1]);
        TShape lshape2(dshape.ndim() - 1);
        lshape2[0] = dshape[0];
        for (index_t i = 2; i < dshape.ndim(); ++i)
          lshape2[i-1] = dshape[i];
        TShape lshape3 = dshape;
        lshape3[1] = 1;
        if (in_shape->at(msravc_softmaxout_enum::kLabel).ndim() == 0) {
          in_shape->at(msravc_softmaxout_enum::kLabel) = lshape1;
        } else if (in_shape->at(msravc_softmaxout_enum::kLabel) == lshape1) {
        } else if (in_shape->at(msravc_softmaxout_enum::kLabel) == lshape2) {
        } else if (in_shape->at(msravc_softmaxout_enum::kLabel) == lshape3) {
        } else {
          std::ostringstream os;
          os << "Expecting " << lshape1 << " or " << lshape2
             << ". But got " << in_shape->at(msravc_softmaxout_enum::kLabel);
          throw InferShapeError(os.str(), msravc_softmaxout_enum::kLabel);
        }
      } else {
        TShape label_shape(dshape.ndim() - 1);
        for (index_t i = 0; i + 1 < dshape.ndim(); ++i)
          label_shape[i] = dshape[i];
        SHAPE_ASSIGN_CHECK(*in_shape, msravc_softmaxout_enum::kLabel, label_shape);
      }
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
    auto ptr = new MsraVcSoftmaxOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_MsraVcSoftmaxOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.out_grad) {
      return {in_data[msravc_softmaxout_enum::kLabel], out_data[msravc_softmaxout_enum::kOut],
              out_grad[msravc_softmaxout_enum::kOut]};
    } else {
      return {in_data[msravc_softmaxout_enum::kLabel], out_data[msravc_softmaxout_enum::kOut]};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[msravc_softmaxout_enum::kOut], in_grad[msravc_softmaxout_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[msravc_softmaxout_enum::kData], out_data[msravc_softmaxout_enum::kOut]}};
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
  MsraVcSoftmaxOutputParam param_;
};  // class MsraVcSoftmaxOutputProp

/*
class DeprecatedSoftmaxProp : public MsraVcSoftmaxOutputProp {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    LOG(INFO) << "Softmax symbol is renamed to SoftmaxOutput. "
      << "This API will be deprecated in Dec, 2015";
    MsraVcSoftmaxOutputProp::param_.Init(kwargs);
  }

  std::string TypeString() const override {
    return "_contrib_MsraVcSoftmaxOutput";
  }
};
*/
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MSRAVC_SOFTMAX_OUTPUT_INL_H_
