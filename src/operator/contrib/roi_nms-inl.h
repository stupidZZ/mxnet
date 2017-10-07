/*!
 * Copyright (c) 2015 by Contributors
 * \file roi_nms-inl.h
 * \brief RoiNms Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo
*/
#ifndef MXNET_OPERATOR_CONTRIB_ROINMS_INL_H_
#define MXNET_OPERATOR_CONTRIB_ROINMS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace roi_nms {
enum RoiNmsOpInputs {kClsProb, kBBoxPred};
enum RoiNmsOpOutputs {kOut, kScore};
enum RoiNmsForwardResource {kTempResource};
}  // roi_nms

struct RoiNmsParam : public dmlc::Parameter<RoiNmsParam> {
  int pre_nms_top_n;
  int post_nms_top_n;
  float threshold;
  bool output_score;
  DMLC_DECLARE_PARAMETER(RoiNmsParam) {
    float tmp[] = {0, 0, 0, 0};
    DMLC_DECLARE_FIELD(pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(post_nms_top_n).set_default(300)
    .describe("Overlap threshold used for non-maximum"
              "suppresion(suppress boxes with IoU >= this threshold");
    DMLC_DECLARE_FIELD(threshold).set_default(0.7)
    .describe("NMS value, below which to suppress.");
    DMLC_DECLARE_FIELD(output_score).set_default(false)
    .describe("Add score to outputs");
  }
};

template<typename xpu>
Operator *CreateOp(RoiNmsParam param);

#if DMLC_USE_CXX11
class RoiNmsProp : public OperatorProperty {
 public:
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[cls_prob, bbox_pred, im_info]";
    const TShape &dshape = in_shape->at(roi_nms::kClsProb);
    if (dshape.ndim() == 0) return false;
    Shape<2> bbox_pred_shape;
    bbox_pred_shape = Shape2(dshape[0], 5);
    SHAPE_ASSIGN_CHECK(*in_shape, roi_nms::kBBoxPred,
                       bbox_pred_shape);
    out_shape->clear();
    // output
    out_shape->push_back(Shape2(param_.post_nms_top_n, 5));
    // score
    out_shape->push_back(Shape2(param_.post_nms_top_n, 1));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new RoiNmsProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_RoiNms";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  int NumVisibleOutputs() const override {
    if (param_.output_score) {
      return 2;
    } else {
      return 1;
    }
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"cls_prob", "bbox_pred"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "score"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  RoiNmsParam param_;
};  // class ProposalProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_PROPOSAL_INL_H_
