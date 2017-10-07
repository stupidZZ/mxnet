/*!
* Copyright (c) 2017 by Contributors
* \file seperable_convolution-inl.h
* \brief separable convolution op
* \author Han Hu
*/
#ifndef MXNET_OPERATOR_SEPARABLE_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_SEPARABLE_CONVOLUTION_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"


namespace mxnet {
  namespace op {

    namespace sepconv {
      enum SeparableConvolutionOpInputs { kData, kWeight, kBias };
      enum SeparableConvolutionOpOutputs { kOut };
      enum SeparableConvolutionOpResource { kTempSpace };
      enum SeparableConvolutionOpCudnnTune { kOff, kLimited, kFastest };

      struct DepthwiseArgs {
        // Input layer dimensions
        int batch;
        int in_rows;
        int in_cols;
        int in_depth;
        int filter_rows;
        int filter_cols;
        int depth_multiplier;
        int stride_y;
        int stride_x;
        int pad_rows;
        int pad_cols;
        int dilate_y;
        int dilate_x;

        // Output layer dimensions
        int out_rows;
        int out_cols;
        int out_depth;

        DepthwiseArgs()
          : batch(0),
          in_rows(0),
          in_cols(0),
          in_depth(0),
          filter_rows(0),
          filter_cols(0),
          depth_multiplier(1),
          stride_y(1),
          stride_x(1),
          pad_rows(0),
          pad_cols(0),
          dilate_y(1),
          dilate_x(1),
          out_rows(0),
          out_cols(0),
          out_depth(0) {}

        DepthwiseArgs(int batch_t, int in_rows_t, int in_cols_t, int in_depth_t, int filter_rows_t, int filter_cols_t,
          int depth_multiplier_t, int stride_y_t, int stride_x_t, int pad_rows_t, int pad_cols_t, int dilate_y_t,
          int dilate_x_t, int out_rows_t, int out_cols_t, int out_depth_t) {
          batch = batch_t;
          in_rows = in_rows_t;
          in_cols = in_cols_t;
          in_depth = in_depth_t;
          filter_rows = filter_rows_t;
          filter_cols = filter_cols_t;
          depth_multiplier = depth_multiplier_t;
          stride_y = stride_y_t;
          stride_x = stride_x_t;
          pad_rows = pad_rows_t;
          pad_cols = pad_cols_t;
          dilate_y = dilate_y_t;
          dilate_x = dilate_x_t;
          out_rows = out_rows_t;
          out_cols = out_cols_t;
          out_depth = out_depth_t;
        }
      };
    }

    struct SeparableConvolutionParam : public dmlc::Parameter<SeparableConvolutionParam> {
      TShape kernel;
      TShape stride;
      TShape dilate;
      TShape pad;
      uint32_t num_filter;
      uint32_t num_group;
      uint64_t workspace;
      bool no_bias;
      dmlc::optional<int> cudnn_tune;
      bool cudnn_off;
      int32_t backward_feat_method;
      int32_t backward_weight_method;
      int32_t forward_method;

      dmlc::optional<int> layout;
      DMLC_DECLARE_PARAMETER(SeparableConvolutionParam) {
        DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (h, w) or (d, h, w)");
        DMLC_DECLARE_FIELD(stride).set_default(TShape())
          .describe("convolution stride: (h, w) or (d, h, w)");
        DMLC_DECLARE_FIELD(dilate).set_default(TShape())
          .describe("convolution dilate: (h, w) or (d, h, w)");
        DMLC_DECLARE_FIELD(pad).set_default(TShape())
          .describe("pad for convolution: (h, w) or (d, h, w)");
        DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
          .describe("convolution filter(channel) number");
        DMLC_DECLARE_FIELD(num_group).set_default(1)
          .describe("Number of group partitions. Equivalent to slicing input into num_group\n    "
            "partitions, apply convolution on each, then concatenate the results");
        DMLC_DECLARE_FIELD(backward_feat_method).set_default(0)
          .describe("Method flag for backward feat");
        DMLC_DECLARE_FIELD(backward_weight_method).set_default(2)
          .describe("Method flag for backward weight");
        DMLC_DECLARE_FIELD(forward_method).set_default(0)
          .describe("Method flag for forward");
        DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
          .describe("Maximum tmp workspace allowed for convolution (MB).");
        DMLC_DECLARE_FIELD(no_bias).set_default(false)
          .describe("Whether to disable bias parameter.");
        DMLC_DECLARE_FIELD(cudnn_tune)
          .add_enum("off", sepconv::kOff)
          .add_enum("limited_workspace", sepconv::kLimited)
          .add_enum("fastest", sepconv::kFastest)
          .set_default(dmlc::optional<int>())
          .describe("Whether to pick convolution algo by running performance test.\n    "
            "Leads to higher startup time but may give faster speed. Options are:\n    "
            "\'off\': no tuning\n    "
            "\'limited_workspace\': run test and pick the fastest algorithm "
            "that doesn't exceed workspace limit.\n    "
            "\'fastest\': pick the fastest algorithm and ignore workspace limit.\n    "
            "If set to None (default), behavior is determined by environment\n    "
            "variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,\n    "
            "1 for limited workspace (default), 2 for fastest.");
        DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
          .describe("Turn off cudnn for this layer.");
        DMLC_DECLARE_FIELD(layout)
          .add_enum("NCHW", mshadow::kNCHW)
          .add_enum("NHWC", mshadow::kNHWC)
          .add_enum("NCDHW", mshadow::kNCDHW)
          .add_enum("NDHWC", mshadow::kNDHWC)
          .set_default(dmlc::optional<int>())
          .describe("Set layout for input, output and weight. Empty for\n    "
            "default layout: NCHW for 2d and NCDHW for 3d.");
      }
    };

    template<typename xpu, typename DType>
    class SeparableConvolutionOp : public Operator {
    public:
      explicit SeparableConvolutionOp(SeparableConvolutionParam p) {
        this->param_ = p;
        // convert MBytes first to Bytes and then to elements.

        param_.workspace = (param_.workspace << 20) / sizeof(DType);
        CHECK(param_.layout.value() == mshadow::kNCHW ||
          param_.layout.value() == mshadow::kNCDHW)
          << "Only support NCHW and NCDHW layout";
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(req[sepconv::kOut], kWriteTo);
        size_t expected = param_.no_bias ? 2 : 3;
        CHECK_EQ(in_data.size(), expected);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        if (param_.kernel.ndim() > 2) {
          LOG(FATAL) << "Volume convolution is not implmented in mshadow";
        }
        
        //LOG(INFO) << "backward_feat_method flag:" << param_.backward_feat_method;
        //LOG(INFO) << "backward_weight_method flag:" << param_.backward_weight_method;

        Tensor<xpu, 4, DType> data = in_data[sepconv::kData].get<xpu, 4, DType>(s);

        int num_group = data.size(1);

        Shape<3> wmat_shape =
          Shape3(num_group,
            param_.num_filter / num_group,
            data.shape_[1] / num_group * param_.kernel[0] * param_.kernel[1]);
        Tensor<xpu, 3, DType> weight =
          in_data[sepconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);

        Tensor<xpu, 4, DType> out = out_data[sepconv::kOut].get<xpu, 4, DType>(s);


        int out_rows = (data.size(2) + 2 * param_.pad[0] -
          (param_.dilate[0] * (param_.kernel[0] - 1) + 1)) / param_.stride[0] + 1;
        int out_cols = (data.size(3) + 2 * param_.pad[1] -
          (param_.dilate[1] * (param_.kernel[1] - 1) + 1)) / param_.stride[1] + 1;
        
        CHECK_EQ(param_.num_filter % data.size(1), 0);
        sepconv::DepthwiseArgs args(data.size(0), data.size(2), data.size(3), data.size(1), param_.kernel[0], param_.kernel[1],
          param_.num_filter / data.size(1), param_.stride[0], param_.stride[1],
          param_.pad[0], param_.pad[1], param_.dilate[0], param_.dilate[1], out_rows,
          out_cols, param_.num_filter);

        if (param_.forward_method >= 0){
          SeparableConv2dForward(out, data, weight, args);

          if (!param_.no_bias) {
            // add bias, broadcast bias to dim 1: channel
            Tensor<xpu, 1, DType> bias = in_data[sepconv::kBias].get<xpu, 1, DType>(s);
            out += broadcast<1>(bias, out.shape_);
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
        // TODO(bing): check the BLAS Handle, be careful
        if (param_.kernel.ndim() > 2) {
          LOG(FATAL) << "Volume convolution is not implmented in mshadow";
        }
        CHECK_EQ(out_grad.size(), 1);
        size_t expected = param_.no_bias == 0 ? 3 : 2;
        CHECK(in_data.size() == expected && in_grad.size() == expected);
        CHECK_EQ(req.size(), expected);
        CHECK_EQ(in_data[sepconv::kWeight].CheckContiguous(), true);
        // get data
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4, DType> data = in_data[sepconv::kData].get<xpu, 4, DType>(s);

        int num_group = data.size(1);

        Shape<3> wmat_shape =
          Shape3(num_group,
            param_.num_filter / num_group,
            data.shape_[1] / num_group * param_.kernel[0] * param_.kernel[1]);
        Tensor<xpu, 3, DType> wmat =
          in_data[sepconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
        Tensor<xpu, 4, DType> grad = out_grad[sepconv::kOut].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> gdata = in_grad[sepconv::kData].get<xpu, 4, DType>(s);

        int out_rows = grad.size(2);
        int out_cols = grad.size(3);
        //int out_rows = (data.size(2) + 2 * param_.pad[0] -
        //  (param_.dilate[0] * (param_.kernel[0] - 1) + 1)) / param_.stride[0] + 1;
        //int out_col = (data.size(3) + 2 * param_.pad[1] -
        //  (param_.dilate[1] * (param_.kernel[1] - 1) + 1)) / param_.stride[1] + 1;

        CHECK_EQ(param_.num_filter % data.size(1), 0);
        sepconv::DepthwiseArgs args(data.size(0), data.size(2), data.size(3), data.size(1), param_.kernel[0], param_.kernel[1],
          param_.num_filter / data.size(1), param_.stride[0], param_.stride[1],
          param_.pad[0], param_.pad[1], param_.dilate[0], param_.dilate[1], out_rows,
          out_cols, param_.num_filter);

       	 
        if (param_.backward_feat_method >= 0){
          SeparableConv2dBackwardInput(gdata, grad, wmat, args);
        }
        const index_t nbatch = data.size(0);
        Tensor<xpu, 1, DType> workspace =
          ctx.requested[sepconv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(this->InitTemp(data.shape_, grad.shape_)), s);

        Shape<1> wmat_shape1 = Shape1(wmat_shape.Size());
        Tensor<xpu, 1, DType> gwmat1;
        Tensor<xpu, 3, DType> gwmat;

        //bool use_mshadow_epx = true;
        if (param_.num_filter == num_group && param_.backward_weight_method == 0) {
          gwmat1 =
            in_grad[sepconv::kWeight].get_with_shape<xpu, 1, DType>(wmat_shape1, s);
        }
        else {
          gwmat =
            in_grad[sepconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
        }
        
        
        if (param_.backward_weight_method <= -2){ }
        else if (param_.num_filter == num_group && param_.backward_weight_method == 1){
            SeparableConv2dBackwardFilter(grad, data, gwmat, args);
        }
        else if (param_.num_filter == num_group && param_.backward_weight_method == 2){
            SeparableConv2dBackwardFilter_share(grad, data, gwmat, args);
        }
       	else { 
        for (index_t i = 0; i < nbatch; i += nstep_) {
          const index_t step = std::min(nstep_, nbatch - i);
          Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(workspace.dptr_,
            Shape2(shape_colunit_[0],
              shape_colunit_[1] * step), s);
          Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
            workspace.dptr_ + temp_col.shape_.Size(),
            Shape3(shape_dstunit_[0],
              shape_dstunit_[1],
              shape_dstunit_[2] * step), s);
          temp_dst = reshape(swapaxis<1, 0>(grad.Slice(i, i + step)), temp_dst.shape_);
          if (param_.backward_weight_method >= -1){
          if (param_.pad[0] == 0 && param_.pad[1] == 0) {
            temp_col = unpack_patch2col(data.Slice(i, i + step),
              param_.kernel[0],
              param_.kernel[1],
              param_.stride[0],
              param_.stride[1],
              param_.dilate[0],
              param_.dilate[1]);
          }
          else {
            temp_col = unpack_patch2col(pad(data.Slice(i, i + step), param_.pad[0], param_.pad[1]),
              param_.kernel[0],
              param_.kernel[1],
              param_.stride[0],
              param_.stride[1],
              param_.dilate[0],
              param_.dilate[1]);
          }
          }
          
          if (param_.num_filter == num_group && param_.backward_weight_method == 0) {

            if (i == 0) {
              Assign(gwmat1, req[sepconv::kWeight], sumall_except_dim<0>(reshape(temp_col, Shape2(num_group * param_.kernel[0] * param_.kernel[1],
                shape_colunit_[1] * step))
                * reshape(swapaxis<1, 0>(reshape(repmat(reshape(temp_dst, Shape1(temp_dst.shape_.Size())),
                  param_.kernel[0] * param_.kernel[1]),
                  Shape3(param_.kernel[0] * param_.kernel[1], num_group, shape_dstunit_[2] * step))),
                  Shape2(param_.kernel[0] * param_.kernel[1] * num_group, shape_dstunit_[2] * step))));

            }
            else {
              gwmat1 += sumall_except_dim<0>(reshape(temp_col, Shape2(num_group * param_.kernel[0] * param_.kernel[1],
                shape_colunit_[1] * step))
                * reshape(swapaxis<1, 0>(reshape(repmat(reshape(temp_dst, Shape1(temp_dst.shape_.Size())),
                  param_.kernel[0] * param_.kernel[1]),
                  Shape3(param_.kernel[0] * param_.kernel[1], num_group, shape_dstunit_[2] * step))),
                  Shape2(param_.kernel[0] * param_.kernel[1] * num_group, shape_dstunit_[2] * step)));
            }
          }
          else if (param_.backward_weight_method == 3){
            //very slow
            const index_t gstride = temp_col.size(0) / num_group;
            for (uint32_t gid = 0; gid < num_group; ++gid) {
              Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
              if (i == 0) {
                Tensor<xpu, 2, DType> tmp_gwmat = gwmat[gid];
                Assign(tmp_gwmat, req[sepconv::kWeight], dot(temp_dst[gid], tmpc.T()));
              }
              else {
                gwmat[gid] += dot(temp_dst[gid], tmpc.T());
              }
            }

          }

        }
        }
        //a little slower by cuda kernel
        //SeparableConv2dBackwardFilter(grad, data, gwmat, args);
        
        if (!param_.no_bias) {
          Tensor<xpu, 1, DType> gbias = in_grad[sepconv::kBias].get<xpu, 1, DType>(s);
          Assign(gbias, req[sepconv::kBias], sumall_except_dim<1>(grad));
        }
      }

    private:
      inline index_t InitTemp(const mshadow::Shape<4> &ishape,
        const mshadow::Shape<4> &oshape) {
        int num_group = ishape[1];
        const int ksize_y = param_.kernel[0];
        const int ksize_x = param_.kernel[1];
        shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
          oshape[2] * oshape[3]);
        shape_dstunit_ = mshadow::Shape3(num_group,
          param_.num_filter / num_group,
          oshape[2] * oshape[3]);
        // param_.workspace is in elements of sizeof(DType)
        // if param_.workspace is set to zero the nstep_ equals ishape[0] (batch)
        nstep_ = std::max(
          std::min(
            static_cast<index_t>(
              param_.workspace / (shape_colunit_.Size() + shape_dstunit_.Size())),
            ishape[0]),
          1U);

        mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
          shape_colunit_[1] * nstep_);
        mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0],
          shape_dstunit_[1],
          shape_dstunit_[2] * nstep_);
        index_t required_size = scol.Size() + sdst.Size();
        CHECK_GE(param_.workspace, required_size)
          << "\nMinimum workspace size: " << required_size * sizeof(DType) << " Bytes\n"
          << "Given: " << param_.workspace * sizeof(DType) << " Bytes";
        return required_size;
      }
      mshadow::Shape<2> shape_colunit_;
      mshadow::Shape<3> shape_dstunit_;
      index_t nstep_;
      SeparableConvolutionParam param_;
    };  // class ConvolutionOp

    template<typename xpu>
    Operator* CreateOp(SeparableConvolutionParam param, int dtype,
      std::vector<TShape> *in_shape,
      std::vector<TShape> *out_shape,
      Context ctx);

#if DMLC_USE_CXX11
    class SeparableConvolutionProp : public OperatorProperty {
    public:
      std::vector<std::string> ListArguments() const override {
        if (!param_.no_bias) {
          return{ "data", "weight", "bias" };
        }
        else {
          return{ "data", "weight" };
        }
      }

      void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
        using namespace mshadow;
        param_.Init(kwargs);
        if (param_.kernel.ndim() == 2) {
          param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
          if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
          if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
          if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
        }
        else {
          CHECK_EQ(param_.kernel.ndim(), 3) << param_.kernel.ndim() << "D convolution not supported";
          param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCDHW;
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
        if (!param_.no_bias) {
          CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
        }
        else {
          CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
        }
        // CHECK_EQ(out_shape->size(), 1) << "Output: [output]";
        out_shape->resize(1, TShape());
        const TShape &dshp = (*in_shape)[sepconv::kData];
        if (dshp.ndim() == 0) return false;
        if (param_.kernel.ndim() == 2) {
          // 2d conv
          CHECK_EQ(dshp.ndim(), 4) \
            << "Input data should be 4D in batch-num_filter-y-x";
          Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);

          int num_group = dshape[1];

          //num_group = dshape[1];
          Shape<4> wshape = Shape4(param_.num_filter / num_group, dshape[1] / num_group,
            param_.kernel[0], param_.kernel[1]);
          wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
          wshape[0] *= num_group;
          SHAPE_ASSIGN_CHECK(*in_shape, sepconv::kWeight, wshape);
          if (!param_.no_bias) {
            SHAPE_ASSIGN_CHECK(*in_shape, sepconv::kBias, Shape1(param_.num_filter));
          }

          const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
          const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
          CHECK_EQ(dshape[1] % num_group, 0) \
            << "input num_filter must divide group size";
          CHECK_EQ(param_.num_filter % num_group, 0) \
            << "output num_filter must divide group size";
          CHECK_GT(param_.kernel.Size(), 0) \
            << "incorrect kernel size: " << param_.kernel;
          CHECK_GT(param_.stride.Size(), 0) \
            << "incorrect stride size: " << param_.stride;
          CHECK_GT(param_.dilate.Size(), 0) \
            << "incorrect dilate size: " << param_.dilate;
          Shape<4> oshape;
          oshape[0] = dshape[0];
          oshape[1] = param_.num_filter;
          oshape[2] = (dshape[2] + 2 * param_.pad[0] -
            (param_.dilate[0] * (ksize_y - 1) + 1)) / param_.stride[0] + 1;
          oshape[3] = (dshape[3] + 2 * param_.pad[1] -
            (param_.dilate[1] * (ksize_x - 1) + 1)) / param_.stride[1] + 1;
          SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
          // Perform incomplete shape inference. Fill in the missing values in data shape.
          // 1) We can always fill in the batch_size.
          // 2) We can back-calculate the input height/width if the corresponding stride is 1.
          oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
          dshape[0] = oshape[0];
          if (param_.stride[0] == 1) {
            dshape[2] = oshape[2] + param_.dilate[0] * (ksize_y - 1) - 2 * param_.pad[0];
          }
          if (param_.stride[1] == 1) {
            dshape[3] = oshape[3] + param_.dilate[1] * (ksize_x - 1) - 2 * param_.pad[1];
          }
          SHAPE_ASSIGN_CHECK(*in_shape, sepconv::kData,
            ConvertLayout(dshape, kNCHW, param_.layout.value()));
          // Check whether the kernel sizes are valid
          if (dshape[2] != 0) {
            CHECK_LE(ksize_y, dshape[2] + 2 * param_.pad[0]) << "kernel size exceed input";
          }
          if (dshape[3] != 0) {
            CHECK_LE(ksize_x, dshape[3] + 2 * param_.pad[1]) << "kernel size exceed input";
          }
          return true;
        }
        else if (param_.kernel.ndim() == 3) {
          // 3d conv
          CHECK_EQ(dshp.ndim(), 5) \
            << "Input data should be 5D in batch-num_filter-depth-y-x";
          Shape<5> dshape = ConvertLayout(dshp.get<5>(), param_.layout.value(), kNCDHW);

          int num_group = dshape[1];
          Shape<5> wshape = Shape5(param_.num_filter / num_group, dshape[1] / num_group,
            param_.kernel[0], param_.kernel[1], param_.kernel[2]);
          wshape = ConvertLayout(wshape, kNCDHW, param_.layout.value());
          wshape[0] *= num_group;
          SHAPE_ASSIGN_CHECK(*in_shape, sepconv::kWeight, wshape);
          if (!param_.no_bias) {
            SHAPE_ASSIGN_CHECK(*in_shape, sepconv::kBias, Shape1(param_.num_filter));
          }

          const index_t ksize_d = static_cast<index_t>(param_.kernel[0]);
          const index_t ksize_y = static_cast<index_t>(param_.kernel[1]);
          const index_t ksize_x = static_cast<index_t>(param_.kernel[2]);
          CHECK_EQ(dshape[1] % num_group, 0)
            << "input num_filter must divide group size";
          CHECK_EQ(param_.num_filter % num_group, 0)
            << "output num_filter must divide group size";
          CHECK_GT(param_.kernel.Size(), 0) \
            << "incorrect kernel size: " << param_.kernel;
          CHECK_GT(param_.stride.Size(), 0) \
            << "incorrect stride size: " << param_.stride;
          CHECK_GT(param_.dilate.Size(), 0) \
            << "incorrect dilate size: " << param_.dilate;
          CHECK_EQ(param_.dilate.Size(), 1)
            << "Dilate is not supported in 3d convolution";
          Shape<5> oshape;
          oshape[0] = dshape[0];
          oshape[1] = param_.num_filter;
          oshape[2] = (dshape[2] + 2 * param_.pad[0] -
            (1 * (ksize_d - 1) + 1)) / param_.stride[0] + 1;
          oshape[3] = (dshape[3] + 2 * param_.pad[1] -
            (1 * (ksize_y - 1) + 1)) / param_.stride[1] + 1;
          oshape[4] = (dshape[4] + 2 * param_.pad[2] -
            (1 * (ksize_x - 1) + 1)) / param_.stride[2] + 1;
          SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCDHW, param_.layout.value()));
          // Perform incomplete shape inference. Fill in the missing values in data shape.
          // 1) We can always fill in the batch_size.
          // 2) We can back-calculate the input depth/height/width if the corresponding stride is 1.
          oshape = ConvertLayout((*out_shape)[0].get<5>(), param_.layout.value(), kNCDHW);
          dshape[0] = oshape[0];
          if (param_.stride[0] == 1) {
            dshape[2] = oshape[2] + 1 * (ksize_d - 1) - 2 * param_.pad[0];
          }
          if (param_.stride[1] == 1) {
            dshape[3] = oshape[3] + 1 * (ksize_y - 1) - 2 * param_.pad[1];
          }
          if (param_.stride[2] == 1) {
            dshape[4] = oshape[4] + 1 * (ksize_x - 1) - 2 * param_.pad[2];
          }
          SHAPE_ASSIGN_CHECK(*in_shape, sepconv::kData,
            ConvertLayout(dshape, kNCDHW, param_.layout.value()));
          // Check whether the kernel sizes are valid
          if (dshape[2] != 0) {
            CHECK_LT(ksize_d, dshape[2] + 2 * param_.pad[0]) << "kernel size exceed input";
          }
          if (dshape[3] != 0) {
            CHECK_LE(ksize_y, dshape[3] + 2 * param_.pad[1]) << "kernel size exceed input";
          }
          if (dshape[4] != 0) {
            CHECK_LE(ksize_x, dshape[4] + 2 * param_.pad[2]) << "kernel size exceed input";
          }
          return true;
        }
        else {
          LOG(FATAL) << "Unknown convolution type";
          return false;
        }
      }

      bool InferType(std::vector<int> *in_type,
        std::vector<int> *out_type,
        std::vector<int> *aux_type) const override {
        CHECK_GE(in_type->size(), 1);
        int dtype = (*in_type)[0];
        CHECK_NE(dtype, -1) << "First input must have specified type";
        for (index_t i = 0; i < in_type->size(); ++i) {
          if ((*in_type)[i] == -1) {
            (*in_type)[i] = dtype;
          }
          else {
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
        auto ptr = new SeparableConvolutionProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "_contrib_SeparableConvolution";
      }

      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        return{ out_grad[sepconv::kOut], in_data[sepconv::kData], in_data[sepconv::kWeight] };
      }

      std::vector<ResourceRequest> ForwardResource(
        const std::vector<TShape> &in_shape) const override {
        return{ ResourceRequest::kTempSpace };
      }

      std::vector<ResourceRequest> BackwardResource(
        const std::vector<TShape> &in_shape) const override {
        return{ ResourceRequest::kTempSpace };
      }

      Operator* CreateOperator(Context ctx) const override {
        LOG(FATAL) << "Not Implemented.";
        return NULL;
      }

      Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
        std::vector<int> *in_type) const override;

    private:
      SeparableConvolutionParam param_;
    };  // class ConvolutionProp
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONVOLUTION_INL_H_


