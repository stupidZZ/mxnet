/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file depthwise_convolution-inl.h
 * \brief CUDA depthwise convolution code
 * \author shuqian.qu@hobot.cc; hanhu@microsoft.com
*/
#ifndef MXNET_OPERATOR_XCEPTION_CONVOLUTION_PROP_H_
#define MXNET_OPERATOR_XCEPTION_CONVOLUTION_PROP_H_
#include <algorithm>
#include <vector>
#include <string>
#include "./convolution-inl.h"

namespace mxnet {
namespace op {

namespace xconv {
enum ConvolutionOpInputs {kData, kWeightDepth, kWeightPoint, kBiasDepth, kBiasPoint};
enum ConvolutionOpOutputs {kOut};
enum ConvolutionOpResource {kTempSpace};
enum ConvolutionOpCudnnTune {kOff, kLimited, kFastest};
}

struct XceptionConvolutionParam : public dmlc::Parameter<XceptionConvolutionParam> {
  TShape kernel;
  TShape kernel_point;
  TShape stride;
  TShape stride_point;
  TShape dilate;
  TShape dilate_point;
  TShape pad;
  TShape pad_point;
  uint32_t num_filter;
  uint32_t num_filter_point;
  uint32_t num_group;
  uint32_t num_group_point;
  uint64_t workspace;
  bool no_bias;
  bool no_bias_point;
  dmlc::optional<int> cudnn_tune;
  bool cudnn_off;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(XceptionConvolutionParam) {
    using namespace mshadow;
    using namespace mshadow::expr;
    DMLC_DECLARE_FIELD(kernel).set_default(Shape2(3,3))
    .describe("depthwise convolution kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(kernel_point).set_default(Shape2(1,1))
    .describe("pointwise convolution kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(Shape2(1,1))
    .describe("convolution stride: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride_point).set_default(Shape2(1,1))
    .describe("pointwise convolution stride: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(dilate).set_default(Shape2(1,1))
    .describe("convolution dilate: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(dilate_point).set_default(Shape2(1,1))
    .describe("pointwise convolution dilate: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(pad).set_default(Shape2(1,1))
    .describe("pad for convolution: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(pad_point).set_default(Shape2(0,0))
    .describe("pad for pointwise convolution: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_filter_point).set_range(1, 100000)
    .describe("pointwise convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(num_group_point).set_default(1)
    .describe("Number of pointwise group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum temperal workspace allowed for convolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(true)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(no_bias_point).set_default(true)
    .describe("Whether to disable bias parameter for pointwise.");
    DMLC_DECLARE_FIELD(cudnn_tune)
    .add_enum("off", xconv::kOff)
    .add_enum("limited_workspace", xconv::kLimited)
    .add_enum("fastest", xconv::kFastest)
    .set_default(dmlc::optional<int>())
        .describe("Whether to pick convolution algo by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCW", mshadow::kNCW)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NCDHW", mshadow::kNCDHW)
    .add_enum("NHWC", mshadow::kNHWC)
    .add_enum("NDHWC", mshadow::kNDHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
  }
};

template<typename xpu>
Operator* CreateOp(XceptionConvolutionParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class XceptionConvolutionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias && !param_.no_bias_point) {
      return {"data", "depthwise_kernel_weight", "pointwise_kernel_weight", "depthwise_kernel_bias", "pointwise_kernel_bias"};
    } else if (!param_.no_bias) {
      return {"data", "depthwise_kernel_weight", "pointwise_kernel_weight", "depthwise_kernel_bias"};
    } else if (!param_.no_bias_point) {
      return {"data", "depthwise_kernel_weight", "pointwise_kernel_weight", "pointwise_kernel_bias"};
    } else {
      return {"data", "depthwise_kernel_weight", "pointwise_kernel_weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 1) {
      param_.layout = param_.layout? param_.layout.value() : mshadow::kNCW;
      if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape1(1);
      if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
    } else if (param_.kernel.ndim() == 2) {
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D convolution not supported";
      param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias &&!param_.no_bias_point) {
      CHECK_EQ(in_shape->size(), 5U) << "Input:[data, depthwise_kernel_weight, pointwise_kernel_weight, depthwise_kernel_bias, pointwise_kernel_bias]";
    } else if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 4U) << "Input:[data, depthwise_kernel_weight, pointwise_kernel_weight, depthwise_kernel_bias]";
    } else if (!param_.no_bias_point) {
      CHECK_EQ(in_shape->size(), 4U) << "Input:[data, depthwise_kernel_weight, pointwise_kernel_weight, pointwise_kernel_bias]";
    } else {
      CHECK_EQ(in_shape->size(), 3U) << "Input:[data, depthwise_kernel_weight, pointwise_kernel_weight]";
    }
    // CHECK_EQ(out_shape->size(), 1) << "Output: [output]";
    out_shape->resize(1, TShape());
    const TShape &dshp = (*in_shape)[xconv::kData];
    if (dshp.ndim() ==  0) return false;
    if (param_.kernel.ndim() == 1) {
      // 1d conv
      CHECK_EQ(dshp.ndim(), 3) << "Input data should be 3D in batch-num_filter-x";
      Shape<3> dshape = ConvertLayout(dshp.get<3>(), param_.layout.value(), kNCW);
      Shape<3> wshape = Shape3(param_.num_filter_point / param_.num_group_point, dshape[1] / param_.num_group_point,
                               param_.kernel_point[0]);
      wshape = ConvertLayout(wshape, kNCW, param_.layout.value());
      wshape[0] *= param_.num_group_point;
      SHAPE_ASSIGN_CHECK(*in_shape, xconv::kWeightPoint, wshape);
      if (!param_.no_bias_point) {
        if (!param_.no_bias) {
           SHAPE_ASSIGN_CHECK(*in_shape, xconv::kBiasPoint, Shape1(param_.num_filter_point));
        } else {
           SHAPE_ASSIGN_CHECK(*in_shape, xconv::kBiasDepth, Shape1(param_.num_filter_point));
        }
      }

      const index_t ksize_x = static_cast<index_t>(param_.kernel_point[0]);
      CHECK_EQ(param_.num_filter, param_.num_group) \
          << "input depthwise num_filter must equal group size";
      CHECK_EQ(param_.pad[0], (param_.kernel[0]-1)/2*param_.dilate[0]) \
          << "input depthwise pad must make the output feature map equal to input feature map";
      CHECK_EQ(param_.stride[0], 1) \
          << "input depthwise stride must equal to 1";
      CHECK_EQ(param_.kernel[0] % 2, 1) \
          << "input depthwise kernel size must be even";

      CHECK_EQ(dshape[1] % param_.num_group_point, 0) \
          << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter_point % param_.num_group_point, 0) \
          << "output num_filter must divide group size";
      CHECK_GT(param_.kernel_point.Size(), 0) \
          << "incorrect kernel size: " << param_.kernel_point;
      CHECK_GT(param_.stride_point.Size(), 0) \
          << "incorrect stride size: " << param_.stride_point;
      CHECK_GT(param_.dilate_point.Size(), 0) \
          << "incorrect dilate size: " << param_.dilate_point;
      CHECK(ksize_x <= dshape[2] + 2 * param_.pad_point[0])
          << "kernel size exceed input";
      Shape<3> oshape;
      oshape[0] = dshape[0];
      oshape[1] = param_.num_filter_point;
      oshape[2] = (dshape[2] + 2 * param_.pad_point[0] -
          (param_.dilate_point[0] * (ksize_x - 1) + 1)) / param_.stride_point[0] + 1;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCW, param_.layout.value()));
      return true;
    } else if (param_.kernel_point.ndim() == 2) {
      // 2d conv
      CHECK_EQ(dshp.ndim(), 4U) \
          << "Input data should be 4D in batch-num_filter-y-x";
      Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
      Shape<4> wshape = Shape4(param_.num_filter_point / param_.num_group_point, dshape[1] / param_.num_group_point,
                               param_.kernel_point[0], param_.kernel_point[1]);
      wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
      wshape[0] *= param_.num_group_point;
      SHAPE_ASSIGN_CHECK(*in_shape, xconv::kWeightPoint, wshape);
      
      Shape<4> wshape_depth = Shape4(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
                               param_.kernel[0], param_.kernel[1]);
      wshape_depth = ConvertLayout(wshape_depth, kNCHW, param_.layout.value());
      wshape_depth[0] *= param_.num_group;
      SHAPE_ASSIGN_CHECK(*in_shape, xconv::kWeightDepth, wshape_depth);
      if (!param_.no_bias) {
        if (!param_.no_bias_point) {
           SHAPE_ASSIGN_CHECK(*in_shape, xconv::kBiasDepth, Shape1(param_.num_filter));
           SHAPE_ASSIGN_CHECK(*in_shape, xconv::kBiasPoint, Shape1(param_.num_filter_point));
        } else {
           SHAPE_ASSIGN_CHECK(*in_shape, xconv::kBiasDepth, Shape1(param_.num_filter));
        }
      }
      else {
        if (!param_.no_bias_point) {
           SHAPE_ASSIGN_CHECK(*in_shape, xconv::kBiasDepth, Shape1(param_.num_filter_point));
        }
      }
      
      CHECK_EQ(param_.num_filter, param_.num_group) \
          << "input depthwise num_filter must equal group size";
      CHECK_EQ(param_.pad[0], (param_.kernel[0]-1)/2*param_.dilate[0]) \
          << "input depthwise pad must make the output feature map equal to input feature map";
      CHECK_EQ(param_.pad[1], (param_.kernel[1]-1)/2*param_.dilate[1]) \
          << "input depthwise pad must make the output feature map equal to input feature map";
      CHECK_EQ(param_.stride[0], 1) \
          << "input depthwise stride must equal to 1";
      CHECK_EQ(param_.stride[1], 1) \
          << "input depthwise stride must equal to 1";
      CHECK_EQ(param_.kernel[0] % 2, 1) \
          << "input depthwise kernel size must be even";
      CHECK_EQ(param_.kernel[1] % 2, 1) \
          << "input depthwise kernel size must be even";

      const index_t ksize_y = static_cast<index_t>(param_.kernel_point[0]);
      const index_t ksize_x = static_cast<index_t>(param_.kernel_point[1]);
      CHECK_EQ(dshape[1] % param_.num_group_point, 0U) \
          << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter_point % param_.num_group_point, 0U) \
          << "output num_filter must divide group size";
      CHECK_GT(param_.kernel_point.Size(), 0U) \
          << "incorrect kernel size: " << param_.kernel_point;
      CHECK_GT(param_.stride_point.Size(), 0U) \
          << "incorrect stride size: " << param_.stride_point;
      CHECK_GT(param_.dilate_point.Size(), 0U) \
          << "incorrect dilate size: " << param_.dilate_point;
      Shape<4> oshape;
      oshape[0] = dshape[0];
      oshape[1] = param_.num_filter_point;
      oshape[2] = (dshape[2] + 2 * param_.pad_point[0] -
          (param_.dilate_point[0] * (ksize_y - 1) + 1)) / param_.stride_point[0] + 1;
      oshape[3] = (dshape[3] + 2 * param_.pad_point[1] -
          (param_.dilate_point[1] * (ksize_x - 1) + 1)) / param_.stride_point[1] + 1;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
      // Perform incomplete shape inference. Fill in the missing values in data shape.
      // 1) We can always fill in the batch_size.
      // 2) We can back-calculate the input height/width if the corresponding stride is 1.
      oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
      dshape[0] = oshape[0];
      if (param_.stride_point[0] == 1) {
        dshape[2] = oshape[2] + param_.dilate_point[0] * (ksize_y - 1) - 2 * param_.pad_point[0];
      }
      if (param_.stride_point[1] == 1) {
        dshape[3] = oshape[3] + param_.dilate_point[1] * (ksize_x - 1) - 2 * param_.pad_point[1];
      }
      SHAPE_ASSIGN_CHECK(*in_shape, xconv::kData,
                          ConvertLayout(dshape, kNCHW, param_.layout.value()));
      // Check whether the kernel sizes are valid
      if (dshape[2] != 0) {
        CHECK_LE(ksize_y, dshape[2] + 2 * param_.pad_point[0]) << "kernel size exceed input";
      }
      if (dshape[3] != 0) {
        CHECK_LE(ksize_x, dshape[3] + 2 * param_.pad_point[1]) << "kernel size exceed input";
      }
      return true;
    } else {
      LOG(FATAL) << "Unknown convolution type";
      return false;
    }
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
    auto ptr = new XceptionConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "XceptionConvolution";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
        return {out_grad[xconv::kOut], in_data[xconv::kData], in_data[xconv::kWeightDepth], in_data[xconv::kWeightPoint]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
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

 private:
  XceptionConvolutionParam param_;
};  // class ConvolutionProp
#endif //DMLC_USE_CXX11


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_XCEPTION_CONVOLUTION_INL_H_
