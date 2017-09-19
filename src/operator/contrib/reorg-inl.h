/*!
 * Copyright (c) 2017 by Contributors
 * \file reorg-inl.h
 * \brief reorgnize psroi pooling output layer
 * \author Bowen Cheng
*/

#ifndef MXNET_OPERATOR_CONTRIB_REORG_INL_H_
#define MXNET_OPERATOR_CONTRIB_REORG_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/base.h>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {
namespace reorg_enum {
enum ReorgInputs {kData, kFeat};
enum ReorgOutputs {kOut};
} // namespace reorg_enum

struct ReorgParam : public dmlc::Parameter<ReorgParam> {
  int num_anchors;
  int channel;
  int pooled_size;
  DMLC_DECLARE_PARAMETER(ReorgParam)  {
    DMLC_DECLARE_FIELD(num_anchors).set_default(9)
    .describe("Number of anchors.");
    DMLC_DECLARE_FIELD(channel).set_default(256)
    .describe("Feature map channels. Used to get height and width of feature map.");
    DMLC_DECLARE_FIELD(pooled_size).set_default(7)
    .describe("ROI/PSROI Pooling size.");
  }
}; // struct ReorgParam

/*
 * Assume batch size (N) = 1 for now.
 * data shape: (H*W*A, C, k, k)
 * out shape: (N=1, A*C*k*k, H, W)
*/
struct ReorgForward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* data, DType* out,
    const int num_batch, const int num_anchors, const int channel, const int k, const int height, const int width) {
    // individual index for out (N, A, C, k, k, H, W)
    int w_idx = i % width;
    int h_idx = i / width % height;
    int kw_idx = i / width / height % k;
    int kh_idx = i / width / height / k % k;
    int channel_idx = i / width / height / k / k % channel;
    int anchor_idx = i / width / height / k / k / channel % num_anchors;
    // batch_idx not used for now
    int batch_idx = i / width / height / k / k / channel / num_anchors;
    // data_idx (N=1, H, W, A, C, k, k)
    int data_idx = h_idx * width * num_anchors * channel * k * k + \
                   w_idx * num_anchors * channel * k * k + \
                   anchor_idx * channel * k * k + \
                   channel_idx * k * k +\
                   kh_idx * k + \
                   kw_idx;
    out[i] = data[data_idx];
  }
}; // struct ReorgForward

/*
 * Assume batch size (N) = 1 for now.
 * in_grad shape: (H*W*A, C, k, k)
 * out_grad shape: (N=1, A*C*k*k, H, W)
*/
struct ReorgBackward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad, DType* out_grad,
    const int num_batch, const int num_anchors, const int channel, const int k, const int height, const int width) {
    // individual index for in_grad (H, W, A, C, k, k)
    int kw_idx = i % k;
    int kh_idx = i / k % k;
    int channel_idx = i / k / k % channel;
    int anchor_idx = i / k / k / channel % num_anchors;
    int w_idx = i / k / k / channel / num_anchors % width;
    int h_idx = i / k / k / channel / num_anchors / width;
    // batch_idx not used for now

    // out_grad_idx (N=1, A, C, k, k, H, W)
    int out_grad_idx = anchor_idx * channel * k * k * height * width + \
                       channel_idx * k * k * height * width + \
                       kh_idx * k * height * width + \
                       kw_idx * height * width + \
                       h_idx * width + \
                       w_idx;
    in_grad[i] = out_grad[out_grad_idx];
  }
}; // struct ReorgBackward

template<typename xpu, typename DType>
class ReorgOp : public Operator {
  public:
    explicit ReorgOp(ReorgParam param) {
      this->param_ = param;
    }

    virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      using namespace mshadow::expr;
      using namespace mxnet_op;
      CHECK_EQ(in_data.size(), 2U) << "Reorg Input: [data, feature]";
      CHECK_EQ(out_data.size(), 1U) << "Reorg Ouput: [out]";
      Stream<xpu> *s = ctx.get_stream<xpu>();
      Tensor<xpu, 4, DType> data = in_data[reorg_enum::kData].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> feature = in_data[reorg_enum::kFeat].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = out_data[reorg_enum::kOut].get<xpu, 4, DType>(s);

      int num_batch = feature.shape_[0];
      if (num_batch > 1) {
        LOG(FATAL) << "Not implemented for batch size > 1";
        return;
      }
      int num_anchors = param_.num_anchors;
      int channel = param_.channel;
      int k = param_.pooled_size;
      int height = feature.shape_[2];
      int width = feature.shape_[3];

      CHECK_EQ(data.shape_[0], height*width*num_anchors);
      CHECK_EQ(data.shape_[1], channel);
      CHECK_EQ(data.shape_[2], k);
      CHECK_EQ(data.shape_[3], k);
      // CHECK_EQ(feature.shape_[1], k*k*channel);
      CHECK_EQ(out.shape_[0], num_batch);
      CHECK_EQ(out.shape_[1], num_anchors*k*k*channel);
      CHECK_EQ(out.shape_[2], height);
      CHECK_EQ(out.shape_[3], width);

      int index = out.shape_.Size();
      Kernel<ReorgForward, xpu>::Launch(s, index,
        data.dptr_, out.dptr_,
        num_batch, num_anchors, channel, k, height, width);
    } // Forward

    virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      using namespace mshadow::expr;
      using namespace mxnet_op;
      CHECK_EQ(in_data.size(), 2U) << "Reorg Input: [data, feature]";
      CHECK_EQ(out_data.size(), 1U) << "Reorg Ouput: [out]";
      Stream<xpu> *s = ctx.get_stream<xpu>();
      Tensor<xpu, 4, DType> data = in_grad[reorg_enum::kData].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> feature = in_grad[reorg_enum::kFeat].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = out_grad[reorg_enum::kOut].get<xpu, 4, DType>(s);

      // feature should have zero gradient
      ASSIGN_DISPATCH(feature, req[reorg_enum::kFeat], 0);

      int num_batch = feature.shape_[0];
      if (num_batch > 1) {
        LOG(FATAL) << "Not implemented for batch size > 1";
        return;
      }
      int num_anchors = param_.num_anchors;
      int channel = param_.channel;
      int k = param_.pooled_size;
      int height = feature.shape_[2];
      int width = feature.shape_[3];

      CHECK_EQ(data.shape_[0], height*width*num_anchors);
      CHECK_EQ(data.shape_[1], channel);
      CHECK_EQ(data.shape_[2], k);
      CHECK_EQ(data.shape_[3], k);
      // CHECK_EQ(feature.shape_[1], k*k*channel);
      CHECK_EQ(out.shape_[0], num_batch);
      CHECK_EQ(out.shape_[1], num_anchors*k*k*channel);
      CHECK_EQ(out.shape_[2], height);
      CHECK_EQ(out.shape_[3], width);

      int index = out.shape_.Size();
      Kernel<ReorgBackward, xpu>::Launch(s, index,
        data.dptr_, out.dptr_,
        num_batch, num_anchors, channel, k, height, width);
    } // Backward

 private:
  ReorgParam param_;
}; // class ReorgOp

template<typename xpu>
Operator* CreateOp(ReorgParam param, int dtype);

#if DMLC_USE_CXX11
class ReorgProp : public OperatorProperty {
 public:
  virtual void Init(const std::vector<std::pair<std::string, std::string>>& kwargs) override {
    param_.Init(kwargs);
  }

  virtual std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
      return {"data", "feature"};
    }

  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape,
                          std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, feature]";

    // not implemented for N > 1
    // input data shape (H*W*A, C, k, k)
    TShape dshape = in_shape->at(reorg_enum::kData);
    TShape fshape = in_shape->at(reorg_enum::kFeat);
    if (dshape.ndim() == 0) return false;
    if (fshape.ndim() == 0) return false;
    CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";
    CHECK_EQ(fshape.ndim(), 4) << "feature should be a 4D tensor";

    int batch = fshape[0];
    CHECK_EQ(batch, 1) << "Not implemented for batch size > 1";
    int num_anchors = param_.num_anchors;
    int channel = param_.channel;
    int k = param_.pooled_size;
    int height = fshape[2];
    int width = fshape[3];

    // output shape (N=1, A*C*k*k, H, W)
    TShape oshape = Shape4(batch, num_anchors*channel*k*k, height, width);
    out_shape->clear();
    out_shape->push_back(oshape);

    return true;
  }

  OperatorProperty* Copy() const override {
    ReorgProp* reorg_sym = new ReorgProp();
    reorg_sym->param_ = this->param_;
    return reorg_sym;
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
      return NULL;
    }

    Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

    std::string TypeString() const override {
      return "_contrib_Reorg";
    }

    std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
      return {out_grad[reorg_enum::kOut], in_data[reorg_enum::kFeat]};
    }

 private:
  ReorgParam param_;
};  // class PSROIPoolingProp
#endif // DMLC_USE_CXX11

} // namespace op
} // namespace mxnet


#endif // MXNET_OPERATOR_CONTRIB_REORG_INL_H_