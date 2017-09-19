/*!
* Copyright (c) 2017 Microsoft
* Licensed under The Apache-2.0 License [see LICENSE for details]
* \file deformable_psroi_pooling-inl.h
* \brief deformable psroi pooling operator and symbol
* \author Xizhou Zhu, Haozhi Qi
*/

#ifndef MXNET_OPERATOR_BILINEAR_RESIZE_INL_H_
#define MXNET_OPERATOR_BILINEAR_RESIZE_INL_H_

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
    namespace BilinearResize {
      enum BilinearResizeOpInputs { kData, kRef };
      enum BilinearResizeOpOutputs { kOut };
    }  // 

    struct BilinearResizeParam : public dmlc::Parameter<BilinearResizeParam> {
      float resize_ratio;
      bool use_ref_data;
      DMLC_DECLARE_PARAMETER(BilinearResizeParam) {
        DMLC_DECLARE_FIELD(use_ref_data).set_default(false).describe("use reference data array");
        DMLC_DECLARE_FIELD(resize_ratio).set_default(1.0).describe("resizing ratio");
      }
    };

    template <typename xpu, typename DType>
    class BilinearResizeOp : public Operator {
    public:
      explicit BilinearResizeOp(BilinearResizeParam p) {
        this->param_ = p;
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        size_t in_expected = param_.use_ref_data ? 2 : 1;
        size_t out_expected = 1;
        CHECK_EQ(in_data.size(), in_expected);
        CHECK_EQ(out_data.size(), out_expected);
        Stream<xpu> *s = ctx.get_stream<xpu>();

        Tensor<xpu, 4, DType> data = in_data[BilinearResize::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> out = out_data[BilinearResize::kOut].get<xpu, 4, DType>(s);
        BilinearResizeForward(out, data);

      }

      virtual void Backward(const OpContext &ctx,
        const std::vector<TBlob> &out_grad,
        const std::vector<TBlob> &in_data,
        const std::vector<TBlob> &out_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &in_grad,
        const std::vector<TBlob> &aux_args) {

        using namespace mshadow;
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4, DType> top_grad = out_grad[BilinearResize::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> data = in_data[BilinearResize::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> data_grad = in_grad[BilinearResize::kData].get<xpu, 4, DType>(s);

        Assign(data_grad, req[BilinearResize::kData], 0.0);
        if (param_.use_ref_data) {
          Tensor<xpu, 4, DType> data_ref_grad = in_grad[BilinearResize::kRef].get<xpu, 4, DType>(s);
          Assign(data_ref_grad, req[BilinearResize::kRef], 0.0);
        }

        BilinearResizeBackward(data_grad, data, top_grad);
      }


    private:
      BilinearResizeParam param_;
    };  // class BilinearResizeOp

    template <typename xpu>
    Operator* CreateOp(BilinearResizeParam param, int dtype);

#if DMLC_USE_CXX11
    class BilinearResizeProp : public OperatorProperty {
    public:

      void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
        param_.Init(kwargs);
      }

      std::vector<std::string> ListArguments() const override {
        if (param_.use_ref_data) {
          return { "data", "data_ref" };
        } else {
          return { "data" };
        }
      }

      std::vector<std::string> ListOutputs() const override {
        return{ "output" };
      }

      int NumOutputs() const override { return 2; }
      int NumVisibleOutputs() const override { return 1; }

      std::map<std::string, std::string> GetParams() const override {
        return param_.__DICT__();
      }

      bool InferShape(std::vector<TShape> *in_shape,
        std::vector<TShape> *out_shape,
        std::vector<TShape> *aux_shape) const override {
        using namespace mshadow;
        if (param_.use_ref_data) {
          CHECK_EQ(in_shape->size(), 2) << "Input:[data, data_ref]";
          TShape resized_shape = in_shape->at(BilinearResize::kRef);
          out_shape->clear();
          out_shape->push_back(resized_shape);
        } else {
          CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
          TShape resized_shape = in_shape->at(BilinearResize::kData);
          resized_shape[2] = static_cast<int>(round(resized_shape[2] * param_.resize_ratio));
          resized_shape[3] = static_cast<int>(round(resized_shape[3] * param_.resize_ratio));
          out_shape->clear();
          out_shape->push_back(resized_shape);
        }
        return true;
      }

      bool InferType(std::vector<int> *in_type,
        std::vector<int> *out_type,
        std::vector<int> *aux_type) const override {
        CHECK_GE(in_type->size(), 1);
        int dtype = (*in_type)[0];
        CHECK_NE(dtype, -1) << "Input must have specified type";

        out_type->clear();
        out_type->push_back(dtype);
        return true;
      }

      OperatorProperty* Copy() const override {
        BilinearResizeProp* bilinear_resize_sym = new BilinearResizeProp();
        bilinear_resize_sym->param_ = this->param_;
        return bilinear_resize_sym;
      }

      std::string TypeString() const override {
        return "_contrib_BilinearResize";
      }

      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        return{ out_grad[BilinearResize::kOut], in_data[BilinearResize::kData] };
      }

      Operator* CreateOperator(Context ctx) const override {
        LOG(FATAL) << "Not Implemented.";
        return NULL;
      }

      Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
        std::vector<int> *in_type) const override;

    private:
      BilinearResizeParam param_;
    };

#endif
  }  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_BILINEAR_RESIZE_INL_H_