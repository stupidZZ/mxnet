/*!
 * Copyright (c) 2015 by Contributors
 * \file roi_nms.cc
 * \brief
 * \author Piotr Teterwak, Bing Xu, Jian Guo
*/

#include "./roi_nms-inl.h"

//=====================
// NMS Utils for R
//=====================
namespace mxnet {
namespace op {
namespace roi_nms_utils {

struct ReverseArgsortCompl {
  const float *val_;
  explicit ReverseArgsortCompl(float *val)
    : val_(val) {}
  bool operator() (float i, float j) {
    return (val_[static_cast<index_t>(i)] >
            val_[static_cast<index_t>(j)]);
  }
};

// copy score and init order
inline void CopyScore(const mshadow::Tensor<cpu, 2>& dets,
                      mshadow::Tensor<cpu, 1> *score,
                      mshadow::Tensor<cpu, 1> *order) {
  for (index_t i = 0; i < dets.size(0); i++) {
    (*score)[i] = dets[i][4];
    (*order)[i] = i;
  }
}

// sort order array according to score
inline void ReverseArgsort(const mshadow::Tensor<cpu, 1>& score,
                           mshadow::Tensor<cpu, 1> *order) {
  ReverseArgsortCompl cmpl(score.dptr_);
  std::sort(order->dptr_, order->dptr_ + score.size(0), cmpl);
}

// reorder proposals according to order and keep the pre_nms_top_n proposals
// dets.size(0) == pre_nms_top_n
inline void ReorderProposals(const mshadow::Tensor<cpu, 2>& prev_dets,
                             const mshadow::Tensor<cpu, 1>& order,
                             const index_t pre_nms_top_n,
                             mshadow::Tensor<cpu, 2> *dets) {
  CHECK_EQ(dets->size(0), pre_nms_top_n);
  for (index_t i = 0; i < dets->size(0); i++) {
    const index_t index = order[i];
    for (index_t j = 0; j < dets->size(1); j++) {
      (*dets)[i][j] = prev_dets[index][j];
    }
  }
}

// greedily keep the max detections (already sorted)
inline void NonMaximumSuppression(const mshadow::Tensor<cpu, 2>& dets,
                                  const float thresh,
                                  const index_t post_nms_top_n,
                                  mshadow::Tensor<cpu, 1> *area,
                                  mshadow::Tensor<cpu, 1> *suppressed,
                                  mshadow::Tensor<cpu, 1> *keep,
                                  index_t *out_size) {
  CHECK_EQ(dets.shape_[1], 5) << "dets: [x1, y1, x2, y2, score]";
  CHECK_GT(dets.shape_[0], 0);
  CHECK_EQ(dets.CheckContiguous(), true);
  CHECK_EQ(area->CheckContiguous(), true);
  CHECK_EQ(suppressed->CheckContiguous(), true);
  CHECK_EQ(keep->CheckContiguous(), true);
  // calculate area
  for (index_t i = 0; i < dets.size(0); ++i) {
    (*area)[i] = (dets[i][2] - dets[i][0] + 1) *
                 (dets[i][3] - dets[i][1] + 1);
  }

  // calculate nms
  *out_size = 0;
  for (index_t i = 0; i < dets.size(0) && (*out_size) < post_nms_top_n; ++i) {
    float ix1 = dets[i][0];
    float iy1 = dets[i][1];
    float ix2 = dets[i][2];
    float iy2 = dets[i][3];
    float iarea = (*area)[i];

    if ((*suppressed)[i] > 0.0f) {
      continue;
    }

    (*keep)[(*out_size)++] = i;
    for (index_t j = i + 1; j < dets.size(0); j ++) {
      if ((*suppressed)[j] > 0.0f) {
        continue;
      }
      float xx1 = std::max(ix1, dets[j][0]);
      float yy1 = std::max(iy1, dets[j][1]);
      float xx2 = std::min(ix2, dets[j][2]);
      float yy2 = std::min(iy2, dets[j][3]);
      float w = std::max(0.0f, xx2 - xx1 + 1.0f);
      float h = std::max(0.0f, yy2 - yy1 + 1.0f);
      float inter = w * h;
      float ovr = inter / (iarea + (*area)[j] - inter);
      if (ovr > thresh) {
        (*suppressed)[j] = 1.0f;
      }
    }
  }
}

}  // namespace roi_nms_utils
}  // namespace op
}  // namespace mxnet


namespace mxnet {
namespace op {

template<typename xpu>
class RoiNmsOp : public Operator{
 public:
  explicit RoiNmsOp(RoiNmsParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[roi_nms::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<cpu, 2> scores = in_data[roi_nms::kClsProb].get<cpu, 2, real_t>(s);
	Tensor<cpu, 2> proposals = in_data[roi_nms::kBBoxPred].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> out = out_data[roi_nms::kOut].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> out_score = out_data[roi_nms::kScore].get<cpu, 2, real_t>(s);
	
	int count = scores.shape_[0];
    int pre_nms_top_n = (param_.pre_nms_top_n > 0) ? param_.pre_nms_top_n : count;
    pre_nms_top_n = std::min(pre_nms_top_n, count);
    int post_nms_top_n = std::min(param_.post_nms_top_n, pre_nms_top_n);
	
	int workspace_size = count * 5 + 2 * count + pre_nms_top_n * 5 + 3 * pre_nms_top_n;
    Tensor<cpu, 1> workspace = ctx.requested[roi_nms::kTempResource].get_space<cpu>(
      Shape1(workspace_size), s);
	  
    int start = 0;
    Tensor<cpu, 2> workspace_proposals(workspace.dptr_ + start, Shape2(count, 5));
    start += count * 5;
    Tensor<cpu, 2> workspace_pre_nms(workspace.dptr_ + start, Shape2(2, count));
    start += 2 * count;
    Tensor<cpu, 2> workspace_ordered_proposals(workspace.dptr_ + start,
                                               Shape2(pre_nms_top_n, 5));
    start += pre_nms_top_n * 5;
    Tensor<cpu, 2> workspace_nms(workspace.dptr_ + start, Shape2(3, pre_nms_top_n));
    start += 3 * pre_nms_top_n;
    CHECK_EQ(workspace_size, start) << workspace_size << " " << start << std::endl;
   
    // Enumerate all shifted anchors
    for (index_t i = 0; i < count; ++i) {
	  workspace_proposals[i][0] = proposals[i][0];
	  workspace_proposals[i][1] = proposals[i][1];
	  workspace_proposals[i][2] = proposals[i][2];
	  workspace_proposals[i][3] = proposals[i][3];
	  workspace_proposals[i][4] = scores[i][0];
    }
	
	Tensor<cpu, 1> score = workspace_pre_nms[0];
    Tensor<cpu, 1> order = workspace_pre_nms[1];

    roi_nms_utils::CopyScore(workspace_proposals,
                     &score,
                     &order);
    roi_nms_utils::ReverseArgsort(score,
                          &order);
    roi_nms_utils::ReorderProposals(workspace_proposals,
                            order,
                            pre_nms_top_n,
                            &workspace_ordered_proposals);
	
	index_t out_size = 0;
    Tensor<cpu, 1> area = workspace_nms[0];
    Tensor<cpu, 1> suppressed = workspace_nms[1];
    Tensor<cpu, 1> keep = workspace_nms[2];
    suppressed = 0;  // surprised!
	
    roi_nms_utils::NonMaximumSuppression(workspace_ordered_proposals,
                                 param_.threshold,
                                 post_nms_top_n,
                                 &area,
                                 &suppressed,
                                 &keep,
                                 &out_size);
    
    // fill in output rois
    for (index_t i = 0; i < out.size(0); ++i) {
      // batch index 0
      out[i][0] = 0;
      if (i < out_size) {
        index_t index = keep[i];
        for (index_t j = 0; j < 4; ++j) {
          out[i][j + 1] =  workspace_ordered_proposals[index][j];
        }
      } else {
        index_t index = keep[i % out_size];
        for (index_t j = 0; j < 4; ++j) {
          out[i][j + 1] = workspace_ordered_proposals[index][j];
        }
      }
    }

    // fill in output score
    for (index_t i = 0; i < out_score.size(0); i++) {
      if (i < out_size) {
        index_t index = keep[i];
        out_score[i][0] = workspace_ordered_proposals[index][4];
      } else {
        index_t index = keep[i % out_size];
        out_score[i][0] = workspace_ordered_proposals[index][4];
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 3);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[roi_nms::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[roi_nms::kBBoxPred].get<xpu, 4, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[roi_nms::kClsProb], 0);
    Assign(gbbox, req[roi_nms::kBBoxPred], 0);
  }

 private:
  RoiNmsParam param_;
};  // class ProposalOp

template<>
Operator *CreateOp<cpu>(RoiNmsParam param) {
  return new RoiNmsOp<cpu>(param);
}

Operator* RoiNmsProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(RoiNmsParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_RoiNms, RoiNmsProp)
.describe("Generate region proposals via RPN")
.add_argument("cls_score", "NDArray-or-Symbol", "Score of how likely proposal is object.")
.add_argument("bbox_pred", "NDArray-or-Symbol", "BBox Predicted deltas from anchors for proposals")
.add_arguments(RoiNmsParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
