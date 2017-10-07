/*!
* Copyright (c) 2017 by Contributors
* \file psroi_pooling_align-inl.h
* \brief psroi pooling operator and symbol
* \author Yi Li, Guodong Zhang
*/
#ifndef MXNET_OPERATOR_PSROI_POOLING_ALIGN_INL_H_
#define MXNET_OPERATOR_PSROI_POOLING_ALIGN_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"


namespace mxnet {
  namespace op {

    // Declare enumeration of input order to make code more intuitive.
    // These enums are only visible within this header
    namespace psroipoolalign {
      enum PSROIPoolingAlignOpInputs { kData, kBox };
      enum PSROIPoolingAlignOpOutputs { kOut, kTopCount };
    }  // psroipoolalign

    struct PSROIPoolingAlignParam : public dmlc::Parameter<PSROIPoolingAlignParam> {
      // TShape pooled_size;
      float spatial_scale;
      int output_dim;
      int pooled_size;
      int group_size;
      int sample_per_part;
      DMLC_DECLARE_PARAMETER(PSROIPoolingAlignParam) {
        DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
          .describe("Ratio of input feature map height (or w) to raw image height (or w). "
            "Equals the reciprocal of total stride in convolutional layers");
        DMLC_DECLARE_FIELD(output_dim).describe("fix output dim");
        DMLC_DECLARE_FIELD(pooled_size).describe("fix pooled size");
        DMLC_DECLARE_FIELD(group_size).set_default(0).describe("fix group size");
        DMLC_DECLARE_FIELD(sample_per_part).describe("the number of samples each part");
      }
    };

    template<typename xpu, typename DType>
    class PSROIPoolingAlignOp : public Operator {
    public:
      explicit PSROIPoolingAlignOp(PSROIPoolingAlignParam p) {
        this->param_ = p;
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        size_t expected = 2;
        CHECK_EQ(in_data.size(), expected);
        CHECK_EQ(out_data.size(), expected);
        CHECK_EQ(out_data[psroipoolalign::kOut].shape_[0], in_data[psroipoolalign::kBox].shape_[0]);
        CHECK_EQ(out_data[psroipoolalign::kTopCount].shape_[0], in_data[psroipoolalign::kBox].shape_[0]);
        Stream<xpu> *s = ctx.get_stream<xpu>();

        Tensor<xpu, 4, DType> data = in_data[psroipoolalign::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 2, DType> bbox = in_data[psroipoolalign::kBox].get<xpu, 2, DType>(s);
        Tensor<xpu, 4, DType> out = out_data[psroipoolalign::kOut].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> top_count = out_data[psroipoolalign::kTopCount].get<xpu, 4, DType>(s);
        CHECK_EQ(data.CheckContiguous(), true);
        CHECK_EQ(bbox.CheckContiguous(), true);
        CHECK_EQ(out.CheckContiguous(), true);
        CHECK_EQ(top_count.CheckContiguous(), true);
        out = -FLT_MAX;
        top_count = 0.0f;
        PSROIPoolAlignForward(out, data, bbox, top_count, param_.spatial_scale, param_.output_dim, 
          param_.group_size, param_.sample_per_part);
      }

      virtual void Backward(const OpContext &ctx,
        const std::vector<TBlob> &out_grad,
        const std::vector<TBlob> &in_data,
        const std::vector<TBlob> &out_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &in_grad,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        size_t expected = 2;
        CHECK_EQ(in_data.size(), expected);
        CHECK_EQ(out_data.size(), expected);
        CHECK_EQ(out_grad[psroipoolalign::kOut].shape_[0], in_data[psroipoolalign::kBox].shape_[0]);
        CHECK_EQ(out_data[psroipoolalign::kTopCount].shape_[0], in_data[psroipoolalign::kBox].shape_[0]);
        CHECK_NE(req[psroipoolalign::kData], kWriteInplace) <<
          "ROIPooling: Backward doesn't support kWriteInplace.";
        CHECK_NE(req[psroipoolalign::kBox], kWriteInplace) <<
          "ROIPooling: Backward doesn't support kWriteInplace.";
        Stream<xpu> *s = ctx.get_stream<xpu>();

        Tensor<xpu, 4, DType> grad_out = out_grad[psroipoolalign::kOut].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> data = in_data[psroipoolalign::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 2, DType> bbox = in_data[psroipoolalign::kBox].get<xpu, 2, DType>(s);
        Tensor<xpu, 4, DType> top_count = out_data[psroipoolalign::kTopCount].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> grad_in = in_grad[psroipoolalign::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 2, DType> grad_roi = in_grad[psroipoolalign::kBox].get<xpu, 2, DType>(s);

        CHECK_EQ(grad_out.CheckContiguous(), true);
        CHECK_EQ(bbox.CheckContiguous(), true);
        CHECK_EQ(top_count.CheckContiguous(), true);
        CHECK_EQ(grad_in.CheckContiguous(), true);

        if (kAddTo == req[psroipoolalign::kData] || kWriteTo == req[psroipoolalign::kData]) {
          if (kWriteTo == req[psroipoolalign::kData]) {
            grad_in = 0.0f;
          }
          PSROIPoolAlignBackwardAcc(grad_in, grad_out, data, bbox, top_count, param_.spatial_scale, 
            param_.output_dim, param_.group_size, param_.sample_per_part);
        }
        if (kWriteTo == req[psroipoolalign::kBox]) {
          grad_roi = 0.0f;
        }

      }

    private:
      PSROIPoolingAlignParam param_;
    };  // class PSROIPoolingAlignOp

      // Decalre Factory function, used for dispatch specialization
    template<typename xpu>
    Operator* CreateOp(PSROIPoolingAlignParam param, int dtype);

#if DMLC_USE_CXX11
    class PSROIPoolingAlignProp : public OperatorProperty {
    public:
      std::vector<std::string> ListArguments() const override {
        return{ "data", "rois" };
      }

      std::vector<std::string> ListOutputs() const override {
        return{ "output", "top_count" };
      }

      int NumOutputs() const override {
        return 2;
      }

      int NumVisibleOutputs() const override {
        return 1;
      }

      void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
        param_.Init(kwargs);
        if (param_.group_size == 0) {
          param_.group_size = param_.pooled_size;
        }
      }

      std::map<std::string, std::string> GetParams() const override {
        return param_.__DICT__();
      }

      bool InferShape(std::vector<TShape> *in_shape,
        std::vector<TShape> *out_shape,
        std::vector<TShape> *aux_shape) const override {
        using namespace mshadow;
        CHECK_EQ(in_shape->size(), 2) << "Input:[data, rois]";

        // data: [batch_size, c, h, w]
        TShape dshape = in_shape->at(psroipoolalign::kData);
        CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";

        // bbox: [num_rois, 5]
        TShape bshape = in_shape->at(psroipoolalign::kBox);
        CHECK_EQ(bshape.ndim(), 2) << "bbox should be a 2D tensor of shape [batch, 5]";
        CHECK_EQ(bshape[1], 5) << "bbox should be a 2D tensor of shape [batch, 5]";

        // out: [num_rois, c, pooled_h, pooled_w]
        // mapping_channel: [num_rois, c, pooled_h, pooled_w]
        out_shape->clear();
        out_shape->push_back(
          Shape4(bshape[0], param_.output_dim, param_.pooled_size, param_.pooled_size));
        out_shape->push_back(
          Shape4(bshape[0], param_.output_dim, param_.pooled_size, param_.pooled_size));
        return true;
      }

      bool InferType(std::vector<int> *in_type,
        std::vector<int> *out_type,
        std::vector<int> *aux_type) const override {
        CHECK_EQ(in_type->size(), 2);
        int dtype = (*in_type)[0];
        CHECK_EQ(dtype, (*in_type)[1]);
        CHECK_NE(dtype, -1) << "Input must have specified type";

        out_type->clear();
        out_type->push_back(dtype);
        out_type->push_back(dtype);
        return true;
      }

      OperatorProperty* Copy() const override {
        PSROIPoolingAlignProp* psroi_pooling_align_sym = new PSROIPoolingAlignProp();
        psroi_pooling_align_sym->param_ = this->param_;
        return psroi_pooling_align_sym;
      }

      std::string TypeString() const override {
        return "_contrib_PSROIPoolingAlign";
      }

      // decalre dependency and inplace optimization options
      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        return{ out_grad[psroipoolalign::kOut], in_data[psroipoolalign::kData],
          in_data[psroipoolalign::kBox], out_data[psroipoolalign::kTopCount] };
      }


      Operator* CreateOperator(Context ctx) const override {
        LOG(FATAL) << "Not Implemented.";
        return NULL;
      }

      Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
        std::vector<int> *in_type) const override;


    private:
      PSROIPoolingAlignParam param_;
    };  // class PSROIPoolingAlignProp
#endif
  }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_PSROI_POOLING_ALIGN_INL_H_
