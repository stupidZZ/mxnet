/*!
 * Copyright (c) 2017 by Contributors
 * \file convolution.cu
 * \brief
 * \author Bing Xu, Jun Wu
*/

#include <vector>

#include "./xception_convolution-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(XceptionConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  op = new XceptionConvolutionOp<float>(param, *in_shape, *out_shape, ctx);
  return op;
}

}  // namespace op
}  // namespace mxnet

namespace tf {
    namespace depthwise_conv {
// Returns whether depthwise convolution forward or backward input pass can be
// performed using the faster ('Small') variant of the kernel.
bool CanLaunchDepthwiseConv2dGPUSmall(const DepthwiseArgs& args) {
  return args.stride_height == 1 && args.stride_width == 1 && args.in_height <= 32 &&
      args.in_width <= 32 && args.in_height == args.out_height &&
      args.in_width == args.out_width && args.pad_height >= 0 &&
      args.pad_height < args.filter_height && args.pad_width >= 0 &&
      args.pad_width < args.filter_width &&
      args.filter_height * args.filter_width <= (args.in_height + 1) / 2 * args.in_width;
}

// Returns whether depthwise convolution backward filter pass can be performed
// using the faster ('Small') variant of the kernel.
bool CanLaunchDepthwiseConv2dBackwardFilterGPUSmall(const DepthwiseArgs args,
                                                    const int block_height) {
  return args.stride_height == 1 && args.stride_width == 1 && args.in_height <= 32 &&
      args.in_width <= 32 && args.in_height == args.out_height &&
      args.in_width == args.out_width && args.pad_height >= 0 &&
      args.pad_height < args.filter_height && args.pad_width >= 0 &&
      args.pad_width < args.filter_width && block_height <= args.in_height &&
      args.filter_height * args.filter_width <= block_height * args.in_width;
}

    }
}

