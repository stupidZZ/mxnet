/*
 * Written by Haozhi Qi
 */
#ifndef MXNET_OPERATOR_LABEL_CONVERTER_INL_H
#define MXNET_OPERATOR_LABEL_CONVERTER_INL_H

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
    namespace label_converter_enum {
      enum LabelConverterInputs {kData};
      enum LabelConverterOutputs {kOut};
      enum LabelConverterFormat {kSoftmax, kSigmoid};
    }

    struct LabelConverterParam : public dmlc::Parameter<LabelConverterParam> {
      int source_label_form;
      int target_label_form;
      DMLC_DECLARE_PARAMETER(LabelConverterParam) {
        DMLC_DECLARE_FIELD(source_label_form)
        .add_enum("softmax", label_converter_enum::kSoftmax)
        .add_enum("sigmoid", label_converter_enum::kSigmoid)
        .set_default(label_converter_enum::kSoftmax)
        .describe("source label format");

        DMLC_DECLARE_FIELD(target_label_form)
        .add_enum("softmax", label_converter_enum::kSoftmax)
        .add_enum("sigmoid", label_converter_enum::kSigmoid)
        .set_default(label_converter_enum::kSigmoid)
        .describe("target label format");
      };
    };

    template <typename xpu, typename DType>
    class LabelConverter : public Operator {
    public:
      explicit LabelConverter(LabelConverterParam param) : param_(param) {}

      virtual void Forward(const OpContext &ctx,
                           const std::vector<TBlob> &in_data,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &out_data,
                           const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 2, DType> data = in_data[label_converter_enum::kData].get<xpu, 2, DType>(s);
        Tensor<xpu, 2, DType> out = out_data[label_converter_enum::kOut].get<xpu, 2, DType>(s);
        LabelConverterForward(data, out);
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
        Tensor<xpu, 2, DType> grad = in_grad[label_converter_enum::kData].get<xpu, 2, DType>(s);
        Assign(grad, req[label_converter_enum::kData], 0);
      }


    private:
      LabelConverterParam param_;
    };


    template <typename xpu>
    Operator* CreateOp(LabelConverterParam param, int type);

    #if DMLC_USE_CXX11
    class LabelConverterProp : public OperatorProperty {
    public:
      std::vector<std::string> ListArguments() const override {
        return { "data" };
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
        CHECK_EQ(in_shape->size(), 1U) << "label converter only accept one input: [source label]";
        const TShape &data_shape = in_shape->at(0);
        TShape out_label_shape;
        if (param_.source_label_form == label_converter_enum::kSoftmax) {
          if (param_.target_label_form == label_converter_enum::kSigmoid) {
            // for now, we only implement the softmax label to sigmoid label converter
            CHECK_EQ(data_shape.ndim(), 2) << "source label dimension should be 2, while get: " << data_shape.ndim();
            out_label_shape = Shape2(data_shape[0], 80);
          } else {
            return false;
          }
        } else {
          return false;
        }
        out_shape->clear();
        out_shape->push_back(out_label_shape);
        return true;
      }

      bool InferType(std::vector<int> *in_type,
                     std::vector<int> *out_type,
                     std::vector<int> *aux_type) const override {
        int dtype = (*in_type)[0];
        out_type->clear();
        out_type->push_back(dtype);
        return true;
      }

      OperatorProperty* Copy() const override {
        auto ptr = new LabelConverterProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "_contrib_LabelConverter";
      }

      Operator* CreateOperator(Context ctx) const override {
        LOG(FATAL) << "Not Implemented.";
        return NULL;
      }

      Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                 std::vector<int> *in_type) const override;
    protected:
      LabelConverterParam param_;
    };

    #endif  // DMLC_USE_CXX11

  }  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_LABEL_CONVERTER_INL_H