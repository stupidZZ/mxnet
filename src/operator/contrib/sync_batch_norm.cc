/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cc
 * \brief
 * \author Bing Xu, Chris Olivier
*/

#include "sync_batch_norm-inl.h"
#include <nnvm/op_attr_types.h>
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../mkl/mkl_memory-inl.h"
#include "../mkl/mkl_batch_norm-inl.h"
#endif  // MXNET_USE_MKL2017

/*! \brief inverse standard deviation <-> variance */
#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/sqrt((__var$) + DType(__eps$)))
#define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))

namespace mxnet {
namespace op {
namespace sync_batchnorm {

template<typename DType>
class DeviceTensor3 {
  DeviceTensor3(const DeviceTensor3&) = delete;

 public:
  inline DeviceTensor3(const TBlob& blob, const size_t indexOfChannel)
    : dptr_(blob.dptr<DType>())
      , indexOfChannel_(indexOfChannel)
      , shape_(3) {
    if (indexOfChannel) {
      shape_[0] = 1;
      for (size_t i = 0; i < indexOfChannel_; ++i) {
        shape_[0] *= blob.shape_[i];
      }
    } else {
      shape_[0] = 0;
    }
    shape_[1] = blob.shape_[indexOfChannel_];
    shape_[2] = 1;
    for (size_t i = indexOfChannel_ + 1, n = blob.shape_.ndim(); i < n; ++i) {
      shape_[2] *= blob.shape_[i];
    }
  }

  inline size_t Size() const {
    size_t n = 1;
    for (int i = 0; i < 3; ++i) {
      n *= shape_[i];
    }
    return n;
  }

  inline size_t ChannelCount() const {
    return shape_[1];
  }

  inline size_t BatchSize() const {
    return shape_[0];
  }

  inline size_t SpatialSize() const {
    return shape_[2];
  }

  DType *dptr_;
  size_t indexOfChannel_;
  TShape shape_;
};

/*! \brief offset, given indices such as bn, channel, depth, row, column */
static inline index_t offset(const TShape& shape,
                             const size_t *indices,
                             const size_t indicesSize) {
  const size_t dim = shape.ndim();
  size_t offset = 0;
  for (size_t i = 0; i < dim; ++i) {
    offset *= shape[i];
    if (indicesSize > i) {
      offset += indices[i];
    }
  }
  return offset;
}

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType, typename OnData>
static inline void ForEachFast(const DeviceTensor3<DType> &tensor,
                               const size_t channel,
                               OnData onData) {
  const size_t num        = tensor.BatchSize();
  const size_t matrixSize = tensor.SpatialSize();

  size_t indices[2] = {0, channel};

  for (size_t batchItem = 0; batchItem < num; ++batchItem) {
    indices[0] = batchItem;
    DType *data = tensor.dptr_ + offset(tensor.shape_, &indices[0],
                                        sizeof(indices)/sizeof(indices[0]));
    for (size_t i = 0; i < matrixSize; ++i) {
      onData(data++);
    }
  }
}

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType1, typename DType2, typename OnData>
static inline void ForEachFast(const DeviceTensor3<DType1> &in_data,
                               const DeviceTensor3<DType2> &out_data,
                               const size_t channel,
                               OnData onData) {
  const size_t num        = in_data.BatchSize();
  const size_t matrixSize = in_data.SpatialSize();

  size_t indices[2] = {0, channel};

  for (size_t batchItem = 0; batchItem < num; ++batchItem) {
    indices[0] = batchItem;
    const size_t off = offset(in_data.shape_, &indices[0], sizeof(indices)/sizeof(indices[0]));
    const DType1 *data = in_data.dptr_ + off;
    DType2 *odata = out_data.dptr_ + off;
    for (size_t i = 0; i < matrixSize; ++i) {
      onData(data++, odata++);
    }
  }
}

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType, typename OnData>
static inline void ForEachFast(const DeviceTensor3<DType>& tensor,
                               OnData onData) {
  const size_t num        = tensor.BatchSize();
  const size_t channels   = tensor.ChannelCount();
  const size_t matrixSize = tensor.SpatialSize();

  for (size_t batchItem = 0; batchItem < num; ++batchItem) {
#pragma openmp for
    for (size_t channel = 0; channel < channels; ++channel) {
      size_t indices[2] = { batchItem, channel };
      const size_t off = offset(tensor.shape_, &indices[0], sizeof(indices)/sizeof(indices[0]));
      const DType *inData = tensor.dptr_ + off;
      for (size_t i = 0; i < matrixSize; ++i) {
        onData(channel, inData++);
      }
    }
  }
}

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType, typename OnData>
static inline void ForEachFast(const DeviceTensor3<DType>& in_data,
                               const DeviceTensor3<DType>& out_data,
                               OnData onData) {
  const size_t num        = in_data.BatchSize();
  const size_t channels   = in_data.ChannelCount();
  const size_t matrixSize = in_data.SpatialSize();

  for (size_t batchItem = 0; batchItem < num; ++batchItem) {
#pragma omp parallel for
    for (int channel = 0; channel < channels; ++channel) {
      size_t indices[2] = { batchItem, static_cast<size_t>(channel) };
      const size_t off = offset(in_data.shape_, &indices[0], sizeof(indices)/sizeof(indices[0]));
      const DType *inData = in_data.dptr_ + off;
      DType *outData = out_data.dptr_ + off;
      for (size_t i = 0; i < matrixSize; ++i) {
        onData(channel, inData++, outData++);
      }
    }
  }
}

/*! \brief Compute the mean of each input channel */
template<typename DType, typename AccReal>
static inline void ComputeMean(const DeviceTensor3<DType> &tensor,
                               AccReal *save_mean) {
  const size_t channelCount = tensor.ChannelCount();

  for (size_t i = 0; i < channelCount; ++i) {
    save_mean[i] = 0;
  }

  ForEachFast(tensor,
              [&save_mean](const size_t channel, const DType *in_data){
                save_mean[channel] += *in_data;
              });

  const size_t itemCount = tensor.Size() / channelCount;
  for (size_t i = 0, n = channelCount; i < n; ++i) {
    save_mean[i] /= itemCount;
  }
}

/*! \brief Compute the variance of each input channel, as well as update moving mean/variants */
template<typename DType, typename AccReal>
static inline void ComputeVariance(const DeviceTensor3<DType> &tensor,
                                   const AccReal *mean_data,
                                   const DType eps,
                                   const TShape &oshape,
                                   AccReal *save_std) {
  const size_t channels   = tensor.ChannelCount();
  for (size_t i = 0; i < channels; ++i) {
    save_std[i] = 0;
  }
  ForEachFast(tensor,
              [&save_std, &mean_data](const index_t channel, const DType *current_in_data) {
                const AccReal mean = mean_data[channel];
                const AccReal current = *current_in_data;
                save_std[channel] += (current - mean) * (current - mean);
              });

  const size_t itemCount = tensor.Size() / channels;
#pragma omp parallel for
  for (int channel = 0; channel < channels; ++channel) {
    const AccReal sum = save_std[channel];

    AccReal invstd;
    if (sum == 0 && eps == 0.0) {
      // Nobody likes to divide by zero
      invstd = 0;
    } else {
      const AccReal variance = sum/itemCount;
      invstd = VARIANCE_TO_INVSTD(variance, eps);
    }
    save_std[channel] = invstd;
  }
}

}  // namespace sync_batchnorm

/*! \brief Forward CPU */
template <typename xpu, typename DType, typename AccReal>
void SyncBatchNormOp<xpu, DType, AccReal>::DoForward(mshadow::Stream<cpu> *,
                                                 const OpContext &ctx,
                                                 const std::vector<TBlob> &in_data,
                                                 const std::vector<OpReqType> &req,
                                                 const std::vector<TBlob> &out_data,
                                                 const std::vector<TBlob> &aux_states) {
  // NOT_IMPLEMENTED;
    return;
}

template <typename xpu, typename DType, typename AccReal>
void SyncBatchNormOp<xpu, DType, AccReal>::DoBackward(mshadow::Stream<cpu> *,
                                                  const OpContext &ctx,
                                                  const std::vector<TBlob> &out_grad,
                                                  const std::vector<TBlob> &in_data,
                                                  const std::vector<TBlob> &out_data,
                                                  const std::vector<OpReqType> &req,
                                                  const std::vector<TBlob> &in_grad,
                                                  const std::vector<TBlob> &aux_states) {
  // NOT_IMPLEMENTED;
    return;
}


template<>
Operator *CreateOp<cpu>(const SyncBatchNormParam& param, const int dtype, const TShape& shape) {
  Operator *op = nullptr;
#define SYNCBATCHNORM_LOG_MKL_INFO() ((void)0)
  if (!op) {
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype,
                                DType,
                                AccReal, {
                                  SYNCBATCHNORM_LOG_MKL_INFO();
                                  op = new SyncBatchNormOp<cpu, DType, AccReal>(param); });                            
  }
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SyncBatchNormProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                          std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK_GE(in_shape->size(), 1U);
  
  SyncBatchNormParam tparam;
  tparam.Copy(param_);
  tparam.dev_id = ctx.dev_id;

  DO_BIND_DISPATCH(CreateOp, tparam, (*in_type)[0], (*in_shape)[0]);
}

DMLC_REGISTER_PARAMETER(SyncBatchNormParam);

MXNET_REGISTER_OP_PROPERTY(SyncBatchNorm, SyncBatchNormProp)
.describe(R"code(Batch normalization.

Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
well as offset ``beta``.

Assume the input has more than one dimension and we normalize along axis 1.
We first compute the mean and variance along this axis:

.. math::

  data\_mean[i] = mean(data[:,i,:,...]) \\
  data\_var[i] = var(data[:,i,:,...])

Then compute the normalized output, which has the same shape as input, as following:

.. math::

  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

Both *mean* and *var* returns a scalar by treating the input as a vector.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
``data_var`` as well, which are needed for the backward pass.

Besides the inputs and the outputs, this operator accepts two auxiliary
states, ``moving_mean`` and ``moving_var``, which are *k*-length
vectors. They are global statistics for the whole dataset, which are updated
by::

  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  moving_var = moving_var * momentum + data_var * (1 - momentum)

If ``use_global_stats`` is set to be true, then ``moving_mean`` and
``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
the output. It is often used during inference.

Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
then set ``gamma`` to 1 and its gradient to 0.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("moving_mean", "NDArray-or-Symbol", "running mean of input")
.add_argument("moving_var", "NDArray-or-Symbol", "running variance of input")
.add_arguments(SyncBatchNormParam::__FIELDS__());

NNVM_REGISTER_OP(SyncBatchNorm)
.set_attr<nnvm::FSetInputVarAttrOnCompose>(
  "FSetInputVarAttrOnCompose",
  [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
    if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
    if (index == 3) {
      var->attrs.dict["__init__"] = "[\"zero\", {}]";
    } else if (index == 4) {
      var->attrs.dict["__init__"] = "[\"one\", {}]";
    }
  });

}  // namespace op
}  // namespace mxnet

