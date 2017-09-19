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
#ifndef MXNET_OPERATOR_XCEPTION_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_XCEPTION_CONVOLUTION_INL_H_
#include <algorithm>
#include <vector>
#include <string>
#include "./convolution-inl.h"
#include "./xception_convolution_prop.h"
#include "../common/cuda_utils.h"

#if MXNET_USE_CUDA
#include <cub/cub.cuh>
#include "./depthwise_convolution_tf.cuh"

#include "./mshadow_op.h"
#include "./mxnet_op.h"
#include <mutex>
#include "./cudnn_algoreg-inl.h"

namespace mxnet {
namespace op {

using namespace tf::depthwise_conv;

template<typename DType>
class XceptionConvolutionOp : public Operator {
 public:
  explicit XceptionConvolutionOp(const XceptionConvolutionParam& param,
                                  const std::vector<TShape>& in_shape,
                                  const std::vector<TShape>& out_shape,
                                  const Context& ctx) {
    args_.batch = in_shape[xconv::kData][0];
    args_.in_channel = in_shape[xconv::kData][1];
    args_.in_height = in_shape[xconv::kData][2];
    args_.in_width = in_shape[xconv::kData][3];
    args_.filter_height = in_shape[xconv::kWeightDepth][2];
    args_.filter_width = in_shape[xconv::kWeightDepth][3];
    args_.stride_height = param.stride[0];
    args_.stride_width = param.stride[1];
    args_.pad_height = param.pad[0];
    args_.pad_width = param.pad[1];
    args_.out_channel = args_.in_channel;
    args_.out_height = args_.in_height;
    args_.out_width = args_.in_width;

    //args_.out_channel = out_shape[xconv::kOut][1];
    //args_.out_height = out_shape[xconv::kOut][2];
    //args_.out_width = out_shape[xconv::kOut][3];
    bias_term_ = !param.no_bias;
    
    CHECK_EQ(args_.pad_height, (args_.filter_height-1)/2 * param.dilate[0]);
    CHECK_EQ(args_.filter_height % 2, 1);
    //CHECK_EQ(args_.in_channel, args_.out_channel);
    //CHECK_EQ(args_.in_height, args_.out_height);
    //CHECK_EQ(args_.in_channel, args_.out_channel);
    //CHECK_EQ(args_.in_channel, args_.out_channel);
    
    //CUDNN part
    using namespace mshadow;
    this->param_ = param;
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    init_cudnn_ = false;
    init_temp_size_ = false;
    dtype_ = DataType<DType>::kCudnnFlag;

#if CUDNN_MAJOR >= 5
    MSHADOW_LAYOUT_SWITCH(param_.layout.value(), Layout, {
      format_ = LayoutType<Layout>::kCudnnFlag;
    });
#else
    CHECK(param_.layout.value() == kNCHW || param_.layout.value() == kNCDHW)
      << "Need CuDNN > 5.0 for layout support";
#endif

    InitDescriptors(ctx, in_shape, out_shape);

    if (!param_.cudnn_tune) {
      param_.cudnn_tune = dmlc::GetEnv("MXNET_CUDNN_AUTOTUNE_DEFAULT", 1);
    }
    SelectAlgo(ctx, in_shape, out_shape);
  }

  ~XceptionConvolutionOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyTensorDescriptor(in_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(out_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyTensorDescriptor(bias_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyFilterDescriptor(filter_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyConvolutionDescriptor(conv_desc_), CUDNN_STATUS_SUCCESS);
    }
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args);

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args);

 private:
  void InitDescriptors(const Context& ctx,
                       const std::vector<TShape>& in_shape,
                       const std::vector<TShape>& out_shape) {
    using namespace mshadow;
    size_t expected = 3;
    if (!param_.no_bias) ++expected;
    if (!param_.no_bias_point) ++expected;
                           
    CHECK_EQ(in_shape.size(), expected);
    CHECK_EQ(out_shape.size(), 1U);
    CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnCreateTensorDescriptor(&bias_desc_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnCreateFilterDescriptor(&filter_desc_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnCreateConvolutionDescriptor(&conv_desc_), CUDNN_STATUS_SUCCESS);

    TShape dshape = in_shape[xconv::kData];
    TShape wshape = in_shape[xconv::kWeightPoint];
    TShape oshape = out_shape[xconv::kOut];
    TShape dstride, ostride;
    wshape[0] /= param_.num_group_point;
    if (param_.kernel.ndim() == 2) {
      // 2d conv
      #if CUDNN_MAJOR >= 6
      CHECK_EQ(cudnnSetConvolution2dDescriptor(conv_desc_,
                                               param_.pad_point[0],
                                               param_.pad_point[1],
                                               param_.stride_point[0],
                                               param_.stride_point[1],
                                               param_.dilate_point[0],
                                               param_.dilate_point[1],
                                               CUDNN_CROSS_CORRELATION, 
                                               dtype_), CUDNN_STATUS_SUCCESS);
      #else 
      CHECK_EQ(cudnnSetConvolution2dDescriptor(conv_desc_,
                                               param_.pad_point[0],
                                               param_.pad_point[1],
                                               param_.stride_point[0],
                                               param_.stride_point[1],
                                               1,
                                               1,
                                               CUDNN_CROSS_CORRELATION), CUDNN_STATUS_SUCCESS);
      #endif

      #if CUDNN_MAJOR >= 5
      wshape = ConvertLayout(wshape.get<4>(), param_.layout.value(), kNCHW);
      CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc_,
                                          dtype_,
                                          format_,
                                          wshape[0],
                                          wshape[1],
                                          wshape[2],
                                          wshape[3]), CUDNN_STATUS_SUCCESS);
      #else
      CHECK_EQ(param_.layout.value(), kNCHW) << "CuDNN V4 only support NCHW layout";
      CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc_,
                                          dtype_,
                                          wshape[0],
                                          wshape[1],
                                          wshape[2],
                                          wshape[3]), CUDNN_STATUS_SUCCESS);
      #endif

      dstride = ConvertLayout(Shape4(dshape[1] * dshape[2] * dshape[3],
                                     dshape[2] * dshape[3],
                                     dshape[3],
                                     1),
                              param_.layout.value(), kNCHW);
      dshape = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);

      ostride = ConvertLayout(Shape4(oshape[1] * oshape[2] * oshape[3],
                                     oshape[2] * oshape[3],
                                     oshape[3],
                                     1),
                              param_.layout.value(), kNCHW);
      oshape = ConvertLayout(oshape.get<4>(), param_.layout.value(), kNCHW);
    } 
    dshape[1] /= param_.num_group_point;
    oshape[1] /= param_.num_group_point;
    weight_offset_ = wshape.Size();
    data_offset_ = dstride[1] * dshape[1];
    out_offset_ = ostride[1] * oshape[1];

    CHECK_EQ(cudnnSetTensorNdDescriptor(in_desc_,
                                        dtype_,
                                        static_cast<int>(dshape.ndim()),
                                        reinterpret_cast<int*>(&dshape[0]),
                                        reinterpret_cast<int*>(&dstride[0])),
             CUDNN_STATUS_SUCCESS);

    CHECK_EQ(cudnnSetTensorNdDescriptor(out_desc_,
                                        dtype_,
                                        static_cast<int>(oshape.ndim()),
                                        reinterpret_cast<int*>(&oshape[0]),
                                        reinterpret_cast<int*>(&ostride[0])),
             CUDNN_STATUS_SUCCESS);

    if (!param_.no_bias_point) {
      TShape bias = in_shape[xconv::kBiasPoint];
      bias_offset_ = bias[0] / param_.num_group_point;
      std::vector<int> bias_shape = {1,
                                     static_cast<int>(bias[0] / param_.num_group_point),
                                     1, 1};
      std::vector<int> bias_stride = {static_cast<int>(bias_offset_), 1, 1, 1};
      if (param_.kernel.ndim() == 3) {
        bias_shape.push_back(1);
        bias_stride.push_back(1);
      }
      CHECK_EQ(cudnnSetTensorNdDescriptor(bias_desc_,
                                          dtype_,
                                          static_cast<int>(bias_shape.size()),
                                          &bias_shape[0],
                                          &bias_stride[0]), CUDNN_STATUS_SUCCESS);
    }
    init_cudnn_ = true;
  }

  void SelectAlgo(const Context& ctx,
                  const std::vector<TShape>& in_shape,
                  const std::vector<TShape>& out_shape) {
    std::string key = CuDNNAlgoReg::Get()->GetKey(param_, in_shape, out_shape);
    if (CuDNNAlgoReg::Get()->Find(key, &algo_, &back_algo_, &back_algo_w_)) return;

    Engine::VarHandle var = Engine::Get()->NewVariable();
    Engine::Get()->PushSync([=](RunContext rctx) {
      mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
      size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));
      if (!param_.cudnn_tune.value()) {
        CHECK_EQ(cudnnGetConvolutionForwardAlgorithm(s->dnn_handle_,
                 in_desc_,
                 filter_desc_,
                 conv_desc_,
                 out_desc_,
                 CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &(this->algo_)), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnGetConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
                 in_desc_,
                 out_desc_,
                 conv_desc_,
                 filter_desc_,
                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &(this->back_algo_w_)), CUDNN_STATUS_SUCCESS);
        CHECK_EQ(cudnnGetConvolutionBackwardDataAlgorithm(s->dnn_handle_,
                 filter_desc_,
                 out_desc_,
                 conv_desc_,
                 in_desc_,
                 CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &(this->back_algo_)), CUDNN_STATUS_SUCCESS);
      } else {
        const int kMaxAlgos = 10;
        int nalgo = kMaxAlgos;
        int i;

        cudnnConvolutionFwdAlgoPerf_t fwd_algo[kMaxAlgos];
        CHECK_EQ(cudnnFindConvolutionForwardAlgorithm(s->dnn_handle_,
                 in_desc_,
                 filter_desc_,
                 conv_desc_,
                 out_desc_,
                 kMaxAlgos,
                 &nalgo,
                 fwd_algo), CUDNN_STATUS_SUCCESS);
        i = 0;
        while (i < nalgo
               && (fwd_algo[i].status != CUDNN_STATUS_SUCCESS
               || (param_.cudnn_tune.value() == xconv::kLimited
               && fwd_algo[i].memory > workspace_byte))) ++i;
        if (i == nalgo) {
          LOG(FATAL) << "Failed to find an convolution algorithm.";
        } else {
          this->algo_ = fwd_algo[i].algo;
        }

        cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo[kMaxAlgos];
        CHECK_EQ(cudnnFindConvolutionBackwardFilterAlgorithm(s->dnn_handle_,
                 in_desc_,
                 out_desc_,
                 conv_desc_,
                 filter_desc_,
                 kMaxAlgos,
                 &nalgo,
                 bwd_filter_algo), CUDNN_STATUS_SUCCESS);
        i = 0;
        while (i < nalgo
               && (bwd_filter_algo[i].status != CUDNN_STATUS_SUCCESS
               || (param_.cudnn_tune.value() == xconv::kLimited
               && bwd_filter_algo[i].memory > workspace_byte))) ++i;
        if (i == nalgo) {
          LOG(FATAL) << "Failed to find an convolution algorithm.";
        } else {
          this->back_algo_w_ = bwd_filter_algo[i].algo;
        }

        cudnnConvolutionBwdDataAlgoPerf_t bwd_data_algo[kMaxAlgos];
        CHECK_EQ(cudnnFindConvolutionBackwardDataAlgorithm(s->dnn_handle_,
                 filter_desc_,
                 out_desc_,
                 conv_desc_,
                 in_desc_,
                 kMaxAlgos,
                 &nalgo,
                 bwd_data_algo), CUDNN_STATUS_SUCCESS);
        i = 0;
        while (i < nalgo
               && (bwd_data_algo[i].status != CUDNN_STATUS_SUCCESS
               || (param_.cudnn_tune.value() == xconv::kLimited
               && bwd_data_algo[i].memory > workspace_byte))) ++i;
        if (i == nalgo) {
          LOG(FATAL) << "Failed to find an convolution algorithm.";
        } else {
          this->back_algo_ = bwd_data_algo[i].algo;
        }
        CuDNNAlgoReg::Get()->Register(key, this->algo_, this->back_algo_, this->back_algo_w_);
      }
    }, ctx, {}, {var});
    Engine::Get()->WaitForVar(var);
    Engine::Get()->DeleteVariable([](RunContext s) {}, ctx, var);
  }

  void GetTempSize(const OpContext& ctx) {
    if (init_temp_size_) return;
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t back_size = 0, back_size_w = 0;
    CHECK_EQ(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               conv_desc_,
               in_desc_,
               back_algo_,
               &back_size), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnGetConvolutionBackwardFilterWorkspaceSize(s->dnn_handle_,
               in_desc_,
               out_desc_,
               conv_desc_,
               filter_desc_,
               back_algo_w_,
               &back_size_w), CUDNN_STATUS_SUCCESS);
    backward_workspace_byte_ = std::max(back_size, back_size_w);
    CHECK_EQ(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               conv_desc_,
               out_desc_,
               algo_,
               &forward_workspace_byte_), CUDNN_STATUS_SUCCESS);

    forward_workspace_ = forward_workspace_byte_ / sizeof(DType) + 1;
    backward_workspace_ = backward_workspace_byte_ / sizeof(DType) + 1;
    // ugly fix CUDNN algorithm selection
    // safe to remove after CuDNN fix 3D conv selection
    // if (param_.kernel.ndim() == 3) {
    //   back_algo_w_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    // }
    init_temp_size_ = true;
  }

  DepthwiseArgs args_;
  bool bias_term_;

  bool init_cudnn_;
  bool init_temp_size_;
  size_t forward_workspace_;
  size_t backward_workspace_;
  size_t forward_workspace_byte_;
  size_t backward_workspace_byte_;
  size_t data_offset_;
  size_t out_offset_;
  size_t weight_offset_;
  size_t bias_offset_;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t algo_;
  cudnnConvolutionBwdDataAlgo_t back_algo_;
  cudnnConvolutionBwdFilterAlgo_t back_algo_w_;
  cudnnTensorFormat_t format_;
  XceptionConvolutionParam param_;
};  // class XceptionConvolutionOp

namespace xception_conv {
namespace cuda {
template<typename DType, int kFilterWidth, int kFilterHeight>
__global__ void __launch_bounds__(1024, 2)
DepthwiseConv2dBackwardFilterKernel(const DepthwiseArgs args,
                                     const DType* out_grad,
                                     const DType* input,
                                     DType* filter_grad) {
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int channel = args.in_channel;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int filter_pixels = filter_width * filter_height;
  const int out_pixels = out_height * out_width;
  const int in_pixels = in_height * in_width;
  const int batch_channel_num = channel * args.batch;
  const int candidate_reduce_thread_num = out_pixels % blockDim.x;

  for (int b = blockIdx.x; b < batch_channel_num; b += gridDim.x) {
    const int local_batch = b / channel;
    const int local_channel = b % channel;
    const int filter_offset_temp = local_channel * filter_pixels;
    const int out_grad_offset_temp = (local_batch * channel * out_pixels) +
        (local_channel * out_pixels);

    for (int out_id = threadIdx.x; out_id < out_pixels; out_id += blockDim.x) {
      const int reduce_thread_num = ((out_pixels - out_id) > candidate_reduce_thread_num) ?
          blockDim.x : candidate_reduce_thread_num;

      const int out_w = out_id % out_width;
      const int out_h = (out_id / out_width) % out_height;
      const int out_grad_offset = out_grad_offset_temp + (out_h * out_width) + (out_w);
      const DType out_g = ldg(out_grad + out_grad_offset);

      const int in_h_start = out_h * stride_height - pad_height;
      const int in_w_start = out_w * stride_width - pad_width;
      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_h = in_h_start + f_h;
        const int input_offset_temp = (local_batch * channel * in_pixels) +
            (local_channel * in_pixels) + (in_h * in_width);
        const int filter_offset_h = filter_width * f_h;

        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_w = in_w_start + f_w;
          DType partial_grad = DType(0.0f);
          if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
            const int input_offset = input_offset_temp + in_w;
            partial_grad = ldg(input + input_offset) * out_g;
          }
          // reduce all valid partial grad in a block
          typedef cub::BlockReduce<DType, mshadow::cuda::kBaseThreadNum> BlockReduceT;
          __shared__ typename BlockReduceT::TempStorage temp_storage_reduce;
          DType aggregate = BlockReduceT(temp_storage_reduce).Sum(partial_grad, reduce_thread_num);
          if (threadIdx.x == 0) {
            DType* addr = filter_grad + f_w + filter_offset_h + filter_offset_temp;
            atomicAdd(addr, aggregate);
          }
          __syncthreads();
        }  // for filter_width
      }  // for filter_height
    }  // for out_pixels
    __syncthreads();
  }  // for batch_channel_num
}
}  // namespace cuda

template<typename DType>
void DepthwiseConv2dForwardGpu(mshadow::Stream<gpu> *stream,
                               const DepthwiseArgs& args,
                               const std::vector<TBlob> &in_data,
                               const TBlob &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace tf::depthwise_conv;
  using namespace tf::depthwise_conv::cuda;
  Tensor<gpu, 4, DType> data = in_data[xconv::kData].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> weight_depth = in_data[xconv::kWeightDepth].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> out = out_data.get<gpu, 4, DType>(stream);

  // select kernel
  if (CanLaunchDepthwiseConv2dGPUSmall(args)) {
    LaunchDepthwiseConv2dGPUSmall<DType, DIRECTION_FORWARD>(
        stream,
        args,
        data.dptr_,
        weight_depth.dptr_,
        out.dptr_);
  } else {
    int num_output = out_data.shape_.Size();
    int block_num = std::min(num_output/mshadow::cuda::kBaseThreadNum + 1,
                             mshadow::cuda::kMaxGridNum);
    auto s = mshadow::Stream<gpu>::GetStream(stream);
    if (args.filter_height == 3 && args.filter_width == 3) {
      DepthwiseConv2dForwardKernel<DType, 3, 3>
          <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(data.dptr_,
                                                               weight_depth.dptr_,
                                                               args,
                                                               num_output,
                                                               out.dptr_);
    } else {
      DepthwiseConv2dForwardKernel<DType, -1, -1>
          <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(data.dptr_,
                                                               weight_depth.dptr_,
                                                               args,
                                                               num_output,
                                                               out.dptr_);
    }
    MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dForwardKernel);
  }
}

template<typename DType>
void DepthwiseConv2dBackwardDataGpu(mshadow::Stream<gpu> *stream,
                                    const DepthwiseArgs& args,
                                    const TBlob &out_grad,
                                    const std::vector<TBlob> &in_data,
                                    const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace tf::depthwise_conv;
  using namespace tf::depthwise_conv::cuda;
  Tensor<gpu, 4, DType> out_g = out_grad.get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> weight_depth = in_data[xconv::kWeightDepth].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> in_data_g = in_grad[xconv::kData].get<gpu, 4, DType>(stream);
  // select kernel
  if (CanLaunchDepthwiseConv2dGPUSmall(args)) {
    LaunchDepthwiseConv2dGPUSmall<DType, DIRECTION_BACKWARD>(
        stream,
        args,
        out_g.dptr_,
        weight_depth.dptr_,
        in_data_g.dptr_);
  } else {
    int num_in_grad = in_grad[xconv::kData].shape_.Size();
    auto s = mshadow::Stream<gpu>::GetStream(stream);
    int block_num = std::min(num_in_grad/mshadow::cuda::kBaseThreadNum + 1,
                             mshadow::cuda::kMaxGridNum);
    DepthwiseConv2dBackwardDataKernel<DType>
        <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(args,
                                                             out_g.dptr_,
                                                             weight_depth.dptr_,
                                                             in_data_g.dptr_,
                                                             num_in_grad);
    MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dBackwardDataKernel);
  }
}

template<typename DType>
void DepthwiseConv2dBackwardFilterGpu(mshadow::Stream<gpu> *stream,
                                      const DepthwiseArgs& args,
                                      const TBlob &out_grad,
                                      const std::vector<TBlob> &in_data,
                                      const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace tf::depthwise_conv;
  Tensor<gpu, 4, DType> out_g = out_grad.get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> in_d = in_data[xconv::kData].get<gpu, 4, DType>(stream);
  Tensor<gpu, 4, DType> weight_depth_grad = in_grad[xconv::kWeightDepth].get<gpu, 4, DType>(stream);
  // select kernel
  if (TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<DType>(stream, args,
                                                            out_g.dptr_,
                                                            in_d.dptr_,
                                                            weight_depth_grad.dptr_)) {
    return;
  } else {
    int num_out_grad = out_grad.shape_.Size();
    auto s = mshadow::Stream<gpu>::GetStream(stream);
    int block_num = std::min(args.out_channel * args.batch, mshadow::cuda::kMaxGridNum);
    if (args.filter_width == 3 && args.filter_height == 3) {
      cuda::DepthwiseConv2dBackwardFilterKernel<DType, 3, 3>
          <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(args,
                                                               out_g.dptr_,
                                                               in_d.dptr_,
                                                               weight_depth_grad.dptr_);
    } else {
      cuda::DepthwiseConv2dBackwardFilterKernel<DType, -1, -1>
          <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(args,
                                                               out_g.dptr_,
                                                               in_d.dptr_,
                                                               weight_depth_grad.dptr_);
    }
    MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dBackwardFilterKernel);
  }
}
}  // namespace xception_conv

template<typename DType>
void XceptionConvolutionOp<DType>::Forward(const OpContext &ctx,
                                            const std::vector<TBlob> &in_data,
                                            const std::vector<OpReqType> &req,
                                            const std::vector<TBlob> &out_data,
                                            const std::vector<TBlob> &aux_states) {
  using namespace mshadow;
  using namespace mshadow::expr;
  auto stream = ctx.get_stream<gpu>();
  CHECK_EQ(req[xconv::kOut], kWriteTo);
  
  Tensor<gpu, 4, DType> depth_in_data = in_data[conv::kData].get<gpu, 4, DType>(stream);
 
  GetTempSize(ctx);
  Tensor<gpu, 1, DType> temp_buffer =
        ctx.requested[xconv::kTempSpace].get_space_typed<gpu, 1, DType>(
            mshadow::Shape1(forward_workspace_ + depth_in_data.shape_.Size()), stream);
  CHECK_EQ(temp_buffer.CheckContiguous(), true);

  Tensor<gpu, 4, DType> depth_out_data(temp_buffer.dptr_, depth_in_data.shape_, stream);
  TBlob depth_out_data_blob(depth_out_data);

  // output forward
  xception_conv::DepthwiseConv2dForwardGpu<DType>(stream, args_, in_data, depth_out_data_blob);

  // bias forward
  if (bias_term_) {
    Tensor<gpu, 1, DType> bias = in_data[xconv::kBiasDepth].get<gpu, 1, DType>(stream);
    Tensor<gpu, 3, DType> output_3d = depth_out_data_blob.get_with_shape<gpu, 3, DType>(
        Shape3(args_.batch, args_.out_channel, args_.out_height * args_.out_width), stream);
    // has bias term, broadcast it to the same shape of output_3d in channel dim
    output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
  }

    size_t expected = 3;
    if (!param_.no_bias) ++expected;
    if (!param_.no_bias_point) ++expected;
    //size_t expected = param_.no_bias ? 2 : 3;
    DType *data_ptr = NULL;
    DType *wmat_ptr = NULL;
    DType *out_ptr = NULL;
    //CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 1, DType> workspace(temp_buffer.dptr_+depth_out_data.MSize(), mshadow::Shape1(forward_workspace_), s);
    //GetTempSize(ctx);
    //Tensor<gpu, 1, DType> workspace =
    //    ctx.requested[xconv::kTempSpace].get_space_typed<gpu, 1, DType>(
    //        mshadow::Shape1(forward_workspace_), s);

    if (param_.kernel.ndim() == 2) {
      Tensor<gpu, 4, DType> data = depth_out_data_blob.get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> wmat = in_data[xconv::kWeightPoint].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data[xconv::kOut].get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(wmat.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      data_ptr = data.dptr_;
      wmat_ptr = wmat.dptr_;
      out_ptr = out.dptr_;
    } 
    for (uint32_t g = 0; g < param_.num_group_point; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      typename DataType<DType>::ScaleType beta_add = 1.0f;
      CHECK_EQ(cudnnConvolutionForward(s->dnn_handle_,
                                       &alpha,
                                       in_desc_,
                                       data_ptr + data_offset_ * g,
                                       filter_desc_,
                                       wmat_ptr + weight_offset_ * g,
                                       conv_desc_,
                                       algo_,
                                       workspace.dptr_,
                                       forward_workspace_byte_,
                                       req[xconv::kOut] == kAddTo? &beta_add : &beta,
                                       out_desc_,
                                       out_ptr + out_offset_ * g), CUDNN_STATUS_SUCCESS);
      if (!param_.no_bias_point) {
        Tensor<gpu, 1, DType> bias;
        if (!param_.no_bias) {
           bias = in_data[xconv::kBiasPoint].get<gpu, 1, DType>(s);
        }
        else {
           bias = in_data[xconv::kBiasPoint].get<gpu, 1, DType>(s);
        }
        #if CUDNN_MAJOR >= 4
        CHECK_EQ(cudnnAddTensor(s->dnn_handle_,
                                &alpha,
                                bias_desc_,
                                bias.dptr_ + bias_offset_ * g,
                                &beta_add,
                                out_desc_,
                                out_ptr + out_offset_ * g), CUDNN_STATUS_SUCCESS);
        #endif
        #if CUDNN_MAJOR == 3
        CHECK_EQ(cudnnAddTensor(s->dnn_handle_,
                                CUDNN_ADD_SAME_C,
                                &alpha,
                                bias_desc_,
                                bias.dptr_ + bias_offset_ * g,
                                &beta_add,
                                out_desc_,
                                out_ptr + out_offset_ * g), CUDNN_STATUS_SUCCESS);
        #endif
      }
    }
}

template<typename DType>
void XceptionConvolutionOp<DType>::Backward(const OpContext &ctx,
                                             const std::vector<TBlob> &out_grad,
                                             const std::vector<TBlob> &in_data,
                                             const std::vector<TBlob> &out_data,
                                             const std::vector<OpReqType> &req,
                                             const std::vector<TBlob> &in_grad,
                                             const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  using namespace mshadow::expr;
    size_t expected = 3;
    if (!param_.no_bias) ++expected;
    if (!param_.no_bias_point) ++expected;
    DType *grad_ptr = NULL;
    DType *wmat_ptr = NULL;
    DType *gwmat_ptr = NULL;
    DType *data_ptr = NULL;
    DType *gdata_ptr = NULL;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(param_.kernel.ndim(), 2);
      Tensor<gpu, 4, DType> grad = out_grad[xconv::kOut].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> wmat = in_data[xconv::kWeightPoint].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> gwmat = in_grad[xconv::kWeightPoint].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> depth_in_data = in_data[conv::kData].get<gpu, 4, DType>(s);
      //Tensor<gpu, 4, DType> gdata = in_grad[conv::kData].get<gpu, 4, DType>(s);
   
    
    Tensor<gpu, 1, DType> temp_buffer =
        ctx.requested[xconv::kTempSpace].get_space_typed<gpu, 1, DType>(
            mshadow::Shape1(backward_workspace_ + depth_in_data.shape_.Size()), s);
    CHECK_EQ(temp_buffer.CheckContiguous(), true);

    Tensor<gpu, 4, DType> data(temp_buffer.dptr_, depth_in_data.shape_, s);
    TBlob depth_out_data_blob(data);
    //Tensor<gpu, 4, DType> gdata(data.dptr_+data.MSize(), depth_in_data.shape_, s);
    //TBlob depth_out_gdata_blob(gdata);

      // forward depthwise convolution
      //Tensor<gpu, 4, DType> data =
      //  ctx.requested[xconv::kTempSpaceDepthOut].get_space_typed<gpu, 1, DType>(
      //  in_data[0].shape_, s);
      //TBlob depth_out_data_blob(depth_out_data);

      // output forward
      xception_conv::DepthwiseConv2dForwardGpu<DType>(s, args_, in_data, depth_out_data_blob);

      // bias forward
      if (bias_term_) {
        Tensor<gpu, 1, DType> bias = in_data[xconv::kBiasDepth].get<gpu, 1, DType>(s);
        Tensor<gpu, 3, DType> output_3d = depth_out_data_blob.get_with_shape<gpu, 3, DType>(
          Shape3(args_.batch, args_.out_channel, args_.out_height * args_.out_width), s);
    // has bias term, broadcast it to the same shape of output_3d in channel dim
        output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
      }

      grad_ptr = grad.dptr_;
      wmat_ptr = wmat.dptr_;
      gwmat_ptr = gwmat.dptr_;
      data_ptr = data.dptr_;
      gdata_ptr = data.dptr_;

    Tensor<gpu, 1, DType> workspace(data.dptr_+data.MSize(), mshadow::Shape1(backward_workspace_), s);
    //Tensor<gpu, 1, DType> workspace =
    //  ctx.requested[xconv::kTempSpace].get_space_typed<gpu, 1, DType>(
    //  mshadow::Shape1(backward_workspace_), s);
    for (uint32_t g = 0; g < param_.num_group_point; ++g) {
        typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      typename DataType<DType>::ScaleType beta_add = 1.0f;
      if (!param_.no_bias_point) {
        Tensor<gpu, 1, DType> gbias;
        int bias_ind = 3;
          if (!param_.no_bias) {
            gbias = in_grad[xconv::kBiasPoint].get<gpu, 1, DType>(s);
            bias_ind = xconv::kBiasPoint;
          }
          else {
            gbias = in_grad[xconv::kBiasDepth].get<gpu, 1, DType>(s);
            bias_ind = xconv::kBiasDepth;
          }
        CHECK_EQ(cudnnConvolutionBackwardBias(s->dnn_handle_,
                                              &alpha,
                                              out_desc_,
                                              grad_ptr + out_offset_ * g,
                                              req[bias_ind] == kAddTo ? &beta_add : &beta,
                                              bias_desc_,
                                              gbias.dptr_ + bias_offset_ * g),
                 CUDNN_STATUS_SUCCESS);
      }
      #if CUDNN_MAJOR <= 4
      CHECK_EQ(cudnnConvolutionBackwardFilter_v3(s->dnn_handle_,
               &alpha,
               in_desc_,
               data_ptr + data_offset_ * g,
               out_desc_,
               grad_ptr + out_offset_ * g,
               conv_desc_,
               back_algo_w_,
               workspace.dptr_,
               backward_workspace_byte_,
               req[xconv::kWeightPoint] == kAddTo? &beta_add : &beta,
               filter_desc_,
               gwmat_ptr + weight_offset_ * g), CUDNN_STATUS_SUCCESS);
      #elif CUDNN_MAJOR >= 5
      CUDNN_CALL(cudnnConvolutionBackwardFilter(s->dnn_handle_,
               &alpha,
               in_desc_,
               data_ptr + data_offset_ * g,
               out_desc_,
               grad_ptr + out_offset_ * g,
               conv_desc_,
               back_algo_w_,
               workspace.dptr_,
               backward_workspace_byte_,
               req[xconv::kWeightPoint] == kAddTo? &beta_add : &beta,
               filter_desc_,
               gwmat_ptr + weight_offset_ * g));
      #endif
      #if CUDNN_MAJOR <= 4
      CHECK_EQ(cudnnConvolutionBackwardData_v3(s->dnn_handle_,
               &alpha,
               filter_desc_,
               wmat_ptr + weight_offset_ * g,
               out_desc_,
               grad_ptr + out_offset_ * g,
               conv_desc_,
               back_algo_,
               workspace.dptr_,
               backward_workspace_byte_,
               &beta,
               in_desc_,
               gdata_ptr + data_offset_ * g), CUDNN_STATUS_SUCCESS);
      #elif CUDNN_MAJOR >= 5
      CHECK_EQ(cudnnConvolutionBackwardData(s->dnn_handle_,
               &alpha,
               filter_desc_,
               wmat_ptr + weight_offset_ * g,
               out_desc_,
               grad_ptr + out_offset_ * g,
               conv_desc_,
               back_algo_,
               workspace.dptr_,
               backward_workspace_byte_,
               &beta,
               in_desc_,
               gdata_ptr + data_offset_ * g), CUDNN_STATUS_SUCCESS);
      #endif
    }

  // backward data
  if (req[xconv::kData] != kNullOp) {
    if (req[xconv::kData] != kAddTo) {
      mshadow::Tensor<gpu, 4, DType> igrad = in_grad[xconv::kData].get<gpu, 4, DType>(s);
      igrad = 0.0f;
    }
    xception_conv::DepthwiseConv2dBackwardDataGpu<DType>(s,
                                                          args_,
                                                          depth_out_data_blob,
                                                          in_data,
                                                          in_grad);
  }

  // backward filter
  if (req[xconv::kWeightDepth] != kNullOp) {
    if (req[xconv::kWeightDepth] != kAddTo) {
      mshadow::Tensor<gpu, 4, DType> wgrad = in_grad[xconv::kWeightDepth].get<gpu, 4, DType>(s);
      wgrad = 0.0f;
    }
    xception_conv::DepthwiseConv2dBackwardFilterGpu<DType>(s,
                                                            args_,
                                                            depth_out_data_blob,
                                                            in_data,
                                                            in_grad);
  }

  // backward bias
  if (bias_term_) {
    Tensor<gpu, 1, DType> dbias = in_grad[xconv::kBiasDepth].get<gpu, 1, DType>(s);
    Tensor<gpu, 3, DType> dout = depth_out_data_blob.get_with_shape<gpu, 3, DType>(
        Shape3(args_.batch, args_.out_channel, args_.out_height * args_.out_width), s);
    ASSIGN_DISPATCH(dbias, req[xconv::kBiasDepth], sumall_except_dim<1>(dout));
  }

  //Tensor<gpu, 4, DType> out_t = in_grad[0].get<gpu, 4, DType>(s);
  //out_t = F<mshadow_op::identity>(data);
}

}  // namespace op
}  // namespace mxnet
#endif

#endif  // MXNET_OPERATOR_XCEPTION_CONVOLUTION_INL_H_
