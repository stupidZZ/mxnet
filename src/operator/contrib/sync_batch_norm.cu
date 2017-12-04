/*!
 * Copyright (c) 2017 by Contributors
 * \file batch_norm.cu
 * \brief CUDA Batch Normalization code
 * \author Chris Olivier, Bing Xu
 * Adapted from Torch
*/
#include <cuda_runtime_api.h>
#include <atomic>
#include <algorithm>
#include <atomic>
#include <thread>
#include "sync_batch_norm-inl.h"

#define WRITE_DATA_FLAG       1
#define WRITE_GAMMA_FLAG      2
#define WRITE_BETA_FLAG       4
#define FIX_GAMMA_FLAG        8
#define IS_TRAINING_FLAG      16
#define USE_GLOBAL_STATS_FLAG 32

#include <mshadow/cuda/tensor_gpu-inl.cuh>
#include "../../common/cuda_utils.h"

/*! \brief inverse standard deviation <-> variance */
#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/sqrt((__var$) + DType(__eps$)))
#define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))

#define SYNC_BATCH_NORM_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)
    

namespace mxnet {
namespace op {
namespace sync_batchnorm {
namespace cuda {

static const unsigned WARP_SIZE = 32;

// The maximum number of threads in a block
static const unsigned MAX_BLOCK_SIZE = 512U;

template<typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ __forceinline__ Out to(const In v) { return (Out) v; }
};

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static unsigned getNumThreads(int nElem) {
  unsigned threadSizes[5] = {32, 64, 128, 256, MAX_BLOCK_SIZE};
  for (int i = 0; i != 5; ++i) {
    if (static_cast<unsigned>(nElem) <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template<typename DType, typename AccReal>
struct Float2 {
  AccReal v1, v2;
  __device__ Float2() {}
  __device__ Float2(DType v1, DType v2)
    : v1(ScalarConvert<DType, AccReal>::to(v1))
      , v2(ScalarConvert<DType, AccReal>::to(v2)) {}
  __device__ Float2(DType v)
    : v1(ScalarConvert<DType, AccReal>::to(v))
      , v2(ScalarConvert<DType, AccReal>::to(v)) {}
  __device__ Float2(int v)
    : v1(ScalarConvert<int, AccReal>::to(v))
      , v2(ScalarConvert<int, AccReal>::to(v)) {}
  __device__ Float2 &operator+=(const Float2 &a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template<typename DType, typename AccReal, typename DeviceTensor3>
struct SumOp {
  __device__ SumOp(const DeviceTensor3 t) : tensor(t) {}
  __device__ __forceinline__ AccReal operator()(int batch, int plane, int n) {
    return ScalarConvert<DType, AccReal>::to(tensor(batch, plane, n));
  }
  const DeviceTensor3 tensor;
};

template<typename DType, typename AccReal, typename DeviceTensor3>
struct VarOp {
  __device__ VarOp(AccReal m, const DeviceTensor3 t)
    : mean(m)
      , tensor(t) {
  }
  __device__ __forceinline__ AccReal operator()(int batch, int plane, int n) {
    DType val = tensor(batch, plane, n);
    return (val - mean) * (val - mean);
  }
  const AccReal mean;
  const DeviceTensor3 tensor;
};

template<typename DType, typename AccReal, typename DeviceTensor3>
struct GradOp {
  __device__ GradOp(AccReal m, const DeviceTensor3 i, const DeviceTensor3 g)
    : mean(m), input(i), gradOutput(g) {}
  __device__ __forceinline__ Float2<DType, AccReal> operator()(int batch, int plane, int n) {
    const DType g = gradOutput(batch, plane, n);
    const DType c = ScalarConvert<AccReal, DType>::to(input(batch, plane, n) - mean);
    return Float2<DType, AccReal>(g, g * c);
  }
  const AccReal mean;
  const DeviceTensor3 input;
  const DeviceTensor3 gradOutput;
};

// Sum across all threads within a warp
template<typename T>
static __device__ __forceinline__ T warpSum(T val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += __shfl_xor(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ T values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

template<typename DType, typename AccReal>
static __device__ __forceinline__ Float2<DType, AccReal> warpSum(Float2<DType, AccReal> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
template<typename T, typename Op, typename DeviceTensor3>
static __device__ T reduce(Op op, DeviceTensor3 tensor, int plane) {
  T sum = (T) 0;
  for (int batch = 0; batch < tensor.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < tensor.getSize(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T) 0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

template <typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor3>
__global__ void SyncBatchNormalizationUpdateOutputInferenceKernel(
  DeviceTensor3 input,
  DeviceTensor3 output,
  DeviceTensor1 runningMean,
  DeviceTensor1 runningVar,
  DeviceTensor1 saveMean,
  DeviceTensor1 saveInvStd,
  DeviceTensor1 weight,
  DeviceTensor1 bias,
  const DType epsilon,
  const uint32_t flags) {
  int plane = blockIdx.x;

  AccReal invstd = VARIANCE_TO_INVSTD(runningVar[plane], epsilon);
  AccReal mean = ScalarConvert<DType, AccReal>::to(runningMean[plane]);
  AccReal gamma = ((flags & FIX_GAMMA_FLAG) == 0 && weight.numElements() > 0)
                  ? ScalarConvert<DType, AccReal>::to(weight[plane])
                  : ScalarConvert<int, AccReal>::to(1);
  AccReal beta = bias.numElements() > 0 ? ScalarConvert<DType, AccReal>::to(bias[plane])
                                        : ScalarConvert<int, AccReal>::to(0);
  if (threadIdx.x == 0) {
    saveMean[plane] = runningMean[plane];
    saveInvStd[plane] = VARIANCE_TO_INVSTD(runningVar[plane], epsilon);
    if ((flags & WRITE_GAMMA_FLAG) != 0 && (flags & FIX_GAMMA_FLAG) != 0
        && weight.numElements() > 0) {
      weight[plane] = AccReal(1);
    }
  }
  // Write normalized and update the output
  for (int batch = 0, nbatch = input.getSize(0); batch < nbatch; ++batch) {
    for (int x = threadIdx.x, nx = input.getSize(2); x < nx; x += blockDim.x) {
      const DType inp = input(batch, plane, x);
      output(batch, plane, x) =
        ScalarConvert<AccReal, DType>::to(gamma * (inp - mean) * invstd + beta);
    }
  }
}

template<typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor3>
__global__ void SyncBatchNormalizationUpdateSumKernel(
  DeviceTensor3 input,
  DeviceTensor1 saveSum) {
  const int plane = blockIdx.x;

  const AccReal sum = reduce<AccReal>(
    SumOp<DType, AccReal, DeviceTensor3>(input), input, plane);
  __syncthreads();

  if (threadIdx.x == 0) {
    saveSum[plane] = ScalarConvert<AccReal, DType>::to(sum);
  }
}

template<typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor3>
__global__ void SyncBatchNormalizationUpdateVarNKernel(
  DeviceTensor3 input,
  const int inst_sum, 
  DeviceTensor1 saveSum,
  DeviceTensor1 saveVarN) {
  const int plane = blockIdx.x;
  AccReal norm = AccReal(1) / inst_sum;
  AccReal mean = saveSum[plane] * norm;
  
  const AccReal varN = reduce<AccReal>(VarOp<DType, AccReal, DeviceTensor3>(mean, input),
                                       input, plane);
  
  if (threadIdx.x == 0) {
    saveVarN[plane] = varN;
  }
}

template<typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor3>
__global__ void SyncBatchNormalizationUpdateOutputKernel(
  DeviceTensor3 input,
  DeviceTensor3 output,
  DeviceTensor1 weight,
  DeviceTensor1 bias,
  const int inst_sum, 
  const AccReal epsilon,
  DeviceTensor1 saveMean,
  DeviceTensor1 saveInvStd,
  DeviceTensor1 saveSum,
  DeviceTensor1 saveVarN,
  const uint32_t flags,
  const int comm_key) {
  //input, output, weight, bias, varN_inst_sum, eps, saveMean, saveInvStd, saveSum, saveVarN, flags    
  
  const int plane = blockIdx.x;
  
  const AccReal norm = AccReal(1) / inst_sum;
  
  AccReal mean = saveSum[plane] * norm;
  AccReal varN = saveVarN[plane];
  AccReal invStd = 0;
  if (varN != AccReal(0) || epsilon != AccReal(0)) {
    invStd = AccReal(1.0) / sqrt(varN * norm + epsilon);
  }
  
  // Save the mean, variance, and moving averages
  if (threadIdx.x == 0) {

    // For one item (0th) per plane (channel), write the per-channel data (ie mean, variance, etc)
    // Momentum based writeback
    saveMean[plane] = ScalarConvert<AccReal, DType>::to(mean);
    saveInvStd[plane] = invStd;
    if ((flags & WRITE_GAMMA_FLAG) != 0 && (flags & FIX_GAMMA_FLAG) != 0
        && weight.numElements() > 0) {
      weight[plane] = AccReal(1);
    }
  }
  
  // Write normalized and update the output
  const AccReal gamma = weight.numElements() > 0
                        ? ScalarConvert<DType, AccReal>::to(weight[plane])
                        : ScalarConvert<int, AccReal>::to(1);
  //if (threadIdx.x == 0 && comm_key==1 && plane==0) {
  //  printf("sync gamma:%f\n", ScalarConvert<DType, AccReal>::to(weight[plane]));
  //}
  const AccReal beta = bias.numElements() > 0 ? ScalarConvert<DType, AccReal>::to(bias[plane])
                                              : ScalarConvert<int, AccReal>::to(0);
  for (int batch = 0, nbatch = input.getSize(0); batch < nbatch; ++batch) {
    for (int x = threadIdx.x, nx = input.getSize(2); x < nx; x += blockDim.x) {
      const DType inp = input(batch, plane, x);
      output(batch, plane, x) =
        ScalarConvert<AccReal, DType>::to(gamma * (inp - mean) * invStd + beta);
      //if(batch == 0 && plane == 0 && x == 0 && comm_key == 1) {
      //  printf("x:%d gamma:%.5f, inp:%.5f, mean:%.5f, invStd:%.5f, beta:%.5f\n", x, gamma, inp, mean, invStd, beta);
      //}
    }
  }
}



template<typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor3>
__global__ void BatchNormalizationUpdateOutputKernel(
  DeviceTensor3 input,
  DeviceTensor3 output,
  DeviceTensor1 weight,
  DeviceTensor1 bias,
  const AccReal epsilon,
  const AccReal momentum,
  DeviceTensor1 runningMean,
  DeviceTensor1 runningVar,
  DeviceTensor1 saveMean,
  DeviceTensor1 saveInvStd,
  const uint32_t flags) {
  const int plane = blockIdx.x;
  const int N = input.getSize(0) * input.getSize(2);

  const AccReal norm = AccReal(1) / N;
  // Compute the mean and variance across (batch, x/y/z)
  const AccReal mean = reduce<AccReal>(
    SumOp<DType, AccReal, DeviceTensor3>(input), input, plane) * norm;
  __syncthreads();
  const AccReal varN = reduce<AccReal>(VarOp<DType, AccReal, DeviceTensor3>(mean, input),
                                       input, plane);
  AccReal invStd = 0;
  if (varN != AccReal(0) || epsilon != AccReal(0)) {
    invStd = AccReal(1.0) / sqrt(varN * norm + epsilon);
  }

  // Save the mean, variance, and moving averages
  if (threadIdx.x == 0) {
    // For one item (0th) per plane (channel), write the per-channel data (ie mean, variance, etc)
    // Momentum based writeback
    saveMean[plane] = ScalarConvert<AccReal, DType>::to(mean);
    saveInvStd[plane] = invStd;
    if ((flags & WRITE_GAMMA_FLAG) != 0 && (flags & FIX_GAMMA_FLAG) != 0
        && weight.numElements() > 0) {
      weight[plane] = AccReal(1);
    }
  }

  // Write normalized and update the output
  const AccReal gamma = weight.numElements() > 0
                        ? ScalarConvert<DType, AccReal>::to(weight[plane])
                        : ScalarConvert<int, AccReal>::to(1);
  const AccReal beta = bias.numElements() > 0 ? ScalarConvert<DType, AccReal>::to(bias[plane])
                                              : ScalarConvert<int, AccReal>::to(0);
  for (int batch = 0, nbatch = input.getSize(0); batch < nbatch; ++batch) {
    for (int x = threadIdx.x, nx = input.getSize(2); x < nx; x += blockDim.x) {
      const DType inp = input(batch, plane, x);
      output(batch, plane, x) =
        ScalarConvert<AccReal, DType>::to(gamma * (inp - mean) * invStd + beta);
    }
  }
}


template<typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor3>
static __global__ void SyncBatchNormalizationGradBackwardKernel(
  const DeviceTensor3 input,
  const DeviceTensor3 gradOutput,
  DeviceTensor1 gradOutputSum,
  DeviceTensor1 dotP,
  const DeviceTensor1 weight,
  const DeviceTensor1 runningMean,
  const DeviceTensor1 runningVar,
  const DeviceTensor1 saveMean,
  const DeviceTensor1 saveInvstd,
  const uint32_t flags,
  const AccReal momentum,
  const double eps) {
  int plane = blockIdx.x;
  int N = gradOutput.getSize(0) * gradOutput.getSize(2);

  const bool is_train_and_not_global_stats =
    (flags & IS_TRAINING_FLAG) != 0 && (flags & USE_GLOBAL_STATS_FLAG) == 0;

  AccReal mean, invstd;
  if (is_train_and_not_global_stats) {
    mean = ScalarConvert<DType, AccReal>::to(saveMean[plane]);
    invstd = saveInvstd[plane];
  } else {
    mean = ScalarConvert<DType, AccReal>::to(runningMean[plane]);
    invstd = VARIANCE_TO_INVSTD(runningVar[plane], eps);
  }

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(gradOutput)
  // 2. DotProduct(input - mean, gradOutput)
  GradOp<DType, AccReal, DeviceTensor3> g(mean, input, gradOutput);
  Float2< DType, AccReal > res = reduce < Float2 < DType, AccReal >,
    GradOp< DType, AccReal, DeviceTensor3 >, DeviceTensor3 > (g, gradOutput, plane);
  const AccReal gradOutputSumVal = res.v1;
  const AccReal dotPVal = res.v2;
  
  if (threadIdx.x == 0) {
    gradOutputSum[plane] = ScalarConvert<AccReal, DType>::to(gradOutputSumVal);
    dotP[plane] = ScalarConvert<AccReal, DType>::to(dotPVal);
  }
}


template<typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor3>
static __global__ void BatchNormalizationBackwardKernel(
  const DeviceTensor3 input,
  const DeviceTensor3 gradOutput,
  DeviceTensor3 gradInput,
  DeviceTensor1 gradWeight,
  DeviceTensor1 gradBias,
  const DeviceTensor1 weight,
  const DeviceTensor1 runningMean,
  const DeviceTensor1 runningVar,
  const DeviceTensor1 saveMean,
  const DeviceTensor1 saveInvstd,
  const uint32_t flags,
  const AccReal momentum,
  const double eps) {
  int plane = blockIdx.x;
  int N = gradOutput.getSize(0) * gradOutput.getSize(2);

  const bool is_train_and_not_global_stats =
    (flags & IS_TRAINING_FLAG) != 0 && (flags & USE_GLOBAL_STATS_FLAG) == 0;

  AccReal mean, invstd;
  if (is_train_and_not_global_stats) {
    mean = ScalarConvert<DType, AccReal>::to(saveMean[plane]);
    invstd = saveInvstd[plane];
  } else {
    mean = ScalarConvert<DType, AccReal>::to(runningMean[plane]);
    invstd = VARIANCE_TO_INVSTD(runningVar[plane], eps);
  }

  const AccReal weightVal = weight.numElements() > 0 ?
                      ScalarConvert<DType, AccReal>::to(weight[plane]) : AccReal(1);
  const AccReal norm = AccReal(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(gradOutput)
  // 2. DotProduct(input - mean, gradOutput)
  GradOp<DType, AccReal, DeviceTensor3> g(mean, input, gradOutput);
  Float2< DType, AccReal > res = reduce < Float2 < DType, AccReal >,
    GradOp< DType, AccReal, DeviceTensor3 >, DeviceTensor3 > (g, gradOutput, plane);
  const AccReal gradOutputSum = res.v1;
  const AccReal dotP = res.v2;

  const AccReal gradMean = gradOutputSum * norm;
  const AccReal projScale = dotP * norm * invstd * invstd;
  const AccReal gradScale = invstd * weightVal;

  if (threadIdx.x == 0 && is_train_and_not_global_stats) {
    const AccReal localVariance = INVSTD_TO_VARIANCE(saveInvstd[plane], eps);
    const AccReal localMean = saveMean[plane];

    // update running averages
    runningMean[plane] = runningMean[plane] * momentum + localMean * (AccReal(1) - momentum);
    runningVar[plane] = runningVar[plane] * momentum + localVariance * (AccReal(1) - momentum);
  }

  if (gradInput.numElements() > 0 && (flags & WRITE_DATA_FLAG) != 0) {
    for (int batch = 0, nbatch = gradOutput.getSize(0); batch < nbatch; ++batch) {
      for (int x = threadIdx.x, nx = gradOutput.getSize(2); x < nx; x += blockDim.x) {
        const DType gradOut = gradOutput(batch, plane, x);
        if (is_train_and_not_global_stats) {
          const DType inp = input(batch, plane, x);
          const AccReal proj = (inp - mean) * projScale;
          gradInput(batch, plane, x) =
            ScalarConvert<AccReal, DType>::to((gradOut - proj - gradMean) * gradScale);
        } else {
          gradInput(batch, plane, x) = ScalarConvert<AccReal, DType>::to(gradOut * gradScale);
        }
      }
    }
  }

  if (gradWeight.numElements() > 0 && threadIdx.x == 0 && (flags & WRITE_GAMMA_FLAG) != 0) {
    if ((flags & FIX_GAMMA_FLAG) == 0) {
      gradWeight[plane] = ScalarConvert<AccReal, DType>::to(dotP * invstd);
    } else {
      gradWeight[plane] = DType(0);
    }
  }

  if (gradBias.numElements() > 0 && threadIdx.x == 0 && (flags & WRITE_BETA_FLAG) != 0) {
    gradBias[plane] = ScalarConvert<AccReal, DType>::to(gradOutputSum);
  }
}


template<typename DType, typename AccReal, typename DeviceTensor1, typename DeviceTensor3>
static __global__ void SyncBatchNormalizationBackwardKernel(
  const DeviceTensor3 input,
  const DeviceTensor3 gradOutput,
  DeviceTensor3 gradInput,
  DeviceTensor1 gradWeight,
  DeviceTensor1 gradBias,
  DeviceTensor1 gradOutputSum,
  DeviceTensor1 dotP,
  const int inst_sum,
  const DeviceTensor1 weight,
  const DeviceTensor1 runningMean,
  const DeviceTensor1 runningVar,
  const DeviceTensor1 saveMean,
  const DeviceTensor1 saveInvstd,
  const uint32_t flags,
  const AccReal momentum,
  const double eps) {
  int plane = blockIdx.x;
  int N = inst_sum;
  AccReal norm = AccReal(1) / N;
  const bool is_train_and_not_global_stats =
    (flags & IS_TRAINING_FLAG) != 0 && (flags & USE_GLOBAL_STATS_FLAG) == 0;

  AccReal mean, invstd;
  if (is_train_and_not_global_stats) {
    mean = ScalarConvert<DType, AccReal>::to(saveMean[plane]);
    invstd = saveInvstd[plane];
  } else {
    mean = ScalarConvert<DType, AccReal>::to(runningMean[plane]);
    invstd = VARIANCE_TO_INVSTD(runningVar[plane], eps);
  }

  const AccReal weightVal = weight.numElements() > 0 ?
                      ScalarConvert<DType, AccReal>::to(weight[plane]) : AccReal(1);

  const AccReal gradOutputSum_val = gradOutputSum[plane];
  const AccReal dotP_val = dotP[plane];
  
  
  const AccReal gradMean = gradOutputSum_val * norm;
  const AccReal projScale = dotP_val * norm * invstd * invstd;
  const AccReal gradScale = invstd  * weightVal;

  if (threadIdx.x == 0 && is_train_and_not_global_stats) {
    const AccReal localVariance = INVSTD_TO_VARIANCE(saveInvstd[plane], eps);
    const AccReal localMean = saveMean[plane];

    // update running averages
    runningMean[plane] = runningMean[plane] * momentum + localMean * (AccReal(1) - momentum);
    runningVar[plane] = runningVar[plane] * momentum + localVariance * (AccReal(1) - momentum);
  }

  if (gradInput.numElements() > 0 && (flags & WRITE_DATA_FLAG) != 0) {
    for (int batch = 0, nbatch = gradOutput.getSize(0); batch < nbatch; ++batch) {
      for (int x = threadIdx.x, nx = gradOutput.getSize(2); x < nx; x += blockDim.x) {
        const DType gradOut = gradOutput(batch, plane, x);
        if (is_train_and_not_global_stats) {
          const DType inp = input(batch, plane, x);
          const AccReal proj = (inp - mean) * projScale;
          gradInput(batch, plane, x) =
            ScalarConvert<AccReal, DType>::to((gradOut - proj - gradMean) * gradScale);
        } else {
          gradInput(batch, plane, x) = ScalarConvert<AccReal, DType>::to(gradOut * gradScale);
        }
      }
    }
  }

  if (gradWeight.numElements() > 0 && threadIdx.x == 0 && (flags & WRITE_GAMMA_FLAG) != 0) {
    if ((flags & FIX_GAMMA_FLAG) == 0) {
      gradWeight[plane] = ScalarConvert<AccReal, DType>::to(dotP_val * invstd);
    } else {
      gradWeight[plane] = DType(0);
    }
  }

  if (gradBias.numElements() > 0 && threadIdx.x == 0 && (flags & WRITE_BETA_FLAG) != 0) {
    gradBias[plane] = ScalarConvert<AccReal, DType>::to(gradOutputSum_val);
  }
}


template<typename DType, int Dim>
struct DeviceTensor {
 public:
  inline DeviceTensor(DType *p, const int *size)
    : dptr_(p) {
    for (int i = 0; i < Dim; ++i) {
      size_[i] = size ? size[i] : 0;
    }
  }

  __host__ __device__
  __forceinline__ unsigned getSize(const int i) const {
    return size_[i];
  }

  __host__ __device__
  __forceinline__ int numElements() const {
    int n = 1;
    for (int i = 0; i < Dim; ++i) {
      n *= size_[i];
    }
    return n;
  }

  __host__ __device__
  __forceinline__ DType &operator()(const size_t batch,
                                    const size_t plane,
                                    const size_t x) const {
    int offset = 0;

    offset *= size_[0];
    offset += batch;

    offset *= size_[1];
    offset += plane;

    offset *= size_[2];
    offset += x;

    return *(const_cast<DType *>(dptr_ + offset));
  }

  __host__ __device__
  __forceinline__ DType &operator[](const size_t x) const {
    return *(dptr_ + x);
  }

  __forceinline__ size_t SpatialSize() const {
    size_t sz = 1;
    for (size_t i = 2; i < Dim; ++i) {
      sz *= size_[i];
    }
    return sz;
  }

  __forceinline__ size_t ChannelCount() const {
    return size_[1];
  }

  DType *dptr_;
  int size_[Dim];
};

template<typename DType, int Dim>
static DeviceTensor<DType, Dim> devicetensor(const TBlob &blob) {
  DType *data = blob.dptr<DType>();
  const int inDim = blob.shape_.ndim();
  if (inDim == Dim) {
    DeviceTensor<DType, Dim> tensor(data, nullptr);
    for (int i = 0; i < Dim; ++i) {
      tensor.size_[i] = blob.size(i);
    }
    return tensor;
  }

  // View in which the last dimensions are collapsed or expanded as needed
  int size[Dim];
  for (int i = 0; i < Dim || i < inDim; ++i) {
    if (i < Dim && i < inDim) {
      size[i] = blob.size(i);
    } else if (i < Dim) {
      size[i] = 1;
    } else {
      size[Dim - 1] *= blob.size(i);
    }
  }
  return DeviceTensor<DType, Dim>(data, &size[0]);
}


#define DeviceTensor1 DeviceTensor<AccReal, 1>
#define DeviceTensor3 DeviceTensor<DType, 3>

template<typename DType, typename AccReal>
static void SyncBatchNormalizationUpdateOutput(mshadow::Stream<gpu> *s,
                                           const OpContext &ctx,
                                           const std::vector<TBlob> &in_data,
                                           const std::vector<TBlob> &out_data,
                                           const std::vector<TBlob> &aux_states,
                                           const uint32_t flags,
                                           double momentum,
                                           double eps,
                                           Sync_Batch_Thread_Comm<gpu, DType, AccReal> * const comm_instance) {
  DeviceTensor3 input = devicetensor<DType, 3>(in_data[sync_batchnorm::kData]);
  DeviceTensor3 output = devicetensor<DType, 3>(out_data[sync_batchnorm::kOut]);
  DeviceTensor1 weight = devicetensor<AccReal, 1>(in_data[sync_batchnorm::kGamma]);
  DeviceTensor1 bias = devicetensor<AccReal, 1>(in_data[sync_batchnorm::kBeta]);
  DeviceTensor1 runningMean = devicetensor<AccReal, 1>(aux_states[sync_batchnorm::kMovingMean]);
  DeviceTensor1 runningVar = devicetensor<AccReal, 1>(aux_states[sync_batchnorm::kMovingVar]);
  DeviceTensor1 saveMean = devicetensor<AccReal, 1>(out_data[sync_batchnorm::kMean]);
  DeviceTensor1 saveInvStd = devicetensor<AccReal, 1>(out_data[sync_batchnorm::kVar]);
  
  //Sync_Batch_Thread_Comm<gpu, DType, AccReal>* comm_instance = (Sync_Batch_Thread_Comm<gpu, DType, AccReal>*)comm_instance_ptr;
  
  AccReal* saveSum_ptr = NULL;
  SYNC_BATCH_NORM_CUDA_CHECK(cudaMalloc(&saveSum_ptr, sizeof(AccReal) * saveMean.size_[0]));
  DeviceTensor1 saveSum(saveSum_ptr, nullptr);
  saveSum.size_[0] = saveMean.size_[0];
  AccReal* saveVarN_ptr = NULL;
  SYNC_BATCH_NORM_CUDA_CHECK(cudaMalloc(&saveVarN_ptr, sizeof(AccReal) * saveInvStd.size_[0]));
  DeviceTensor1 saveVarN(saveVarN_ptr, nullptr);
  saveVarN.size_[0] = saveInvStd.size_[0];
  
  
  DCHECK_GT(weight.numElements(), 0);
  if ((flags & IS_TRAINING_FLAG) == 0 || (flags & USE_GLOBAL_STATS_FLAG) != 0) {
    dim3 blocks(input.ChannelCount());
    dim3 threads(getNumThreads(input.SpatialSize()));
    SyncBatchNormalizationUpdateOutputInferenceKernel<DType, AccReal, DeviceTensor1, DeviceTensor3>
      <<< blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >>> (
      input, output, runningMean, runningVar, saveMean,
        saveInvStd, weight, bias, eps, flags);
  } 
  else {
    dim3 blocks(input.ChannelCount());
    dim3 threads(getNumThreads(input.SpatialSize()));
    
    // compute mean
    SyncBatchNormalizationUpdateSumKernel<DType, AccReal, DeviceTensor1, DeviceTensor3 >
      << < blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >> > (
      input, saveSum);
    
    int saveSum_inst_sum = 0;
    std::vector<AccReal> saveSum_host(saveSum.size_[0]);
    std::vector<AccReal> saveSum_output(saveSum.size_[0]);
    SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(&saveSum_host[0],
                              saveSum.dptr_,
                              sizeof(AccReal) * saveSum.size_[0],
                              cudaMemcpyDeviceToHost));
    
    comm_instance->reduce(saveSum_host, saveSum_output, input.getSize(0) * input.getSize(2), saveSum_inst_sum, std::this_thread::get_id());
    
    SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(saveSum.dptr_, &saveSum_output[0], sizeof(AccReal) * saveSum.size_[0],
                                cudaMemcpyHostToDevice));

    // Compute varN
    int varN_inst_sum = 0;
    SyncBatchNormalizationUpdateVarNKernel<DType, AccReal, DeviceTensor1, DeviceTensor3 >
      << < blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >> > (
      input, saveSum_inst_sum, saveSum, saveVarN);
        
    std::vector<AccReal> saveVarN_host(saveVarN.size_[0]);  
    std::vector<AccReal> saveVarN_output(saveVarN.size_[0]);
    SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(&saveVarN_host[0],
                              saveVarN.dptr_,
                              sizeof(AccReal) * saveVarN.size_[0],
                              cudaMemcpyDeviceToHost));
    
    comm_instance->reduce(saveVarN_host, saveVarN_output, input.getSize(0) * input.getSize(2), varN_inst_sum, std::this_thread::get_id());
                                                                        
    SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(saveVarN.dptr_, &saveVarN_output[0], sizeof(AccReal) * saveVarN.size_[0],
                                cudaMemcpyHostToDevice));
    
    SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(&saveSum_host[0],
                              saveSum.dptr_,
                              sizeof(AccReal) * saveSum.size_[0],
                              cudaMemcpyDeviceToHost));

    // Compute Output
    SyncBatchNormalizationUpdateOutputKernel<DType, AccReal, DeviceTensor1, DeviceTensor3 >
      << < blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >> > (
      input, output, weight, bias, varN_inst_sum, eps, saveMean, saveInvStd, saveSum, saveVarN, flags, comm_instance->comm_key);
    
    std::vector<AccReal> output_host_res(output.size_[0]);
    SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(&output_host_res[0],
                              output.dptr_,
                              sizeof(AccReal) * output.size_[0],
                              cudaMemcpyDeviceToHost));
                              
    std::vector<AccReal> saveMean_host_res(saveMean.size_[0]);
    SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(&saveMean_host_res[0],
                              saveMean.dptr_,
                              sizeof(AccReal) * saveMean.size_[0],
                              cudaMemcpyDeviceToHost));
                              
    std::vector<AccReal> saveInvStd_host_res(saveMean.size_[0]);
    SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(&saveInvStd_host_res[0],
                              saveInvStd.dptr_,
                              sizeof(AccReal) * saveInvStd.size_[0],
                              cudaMemcpyDeviceToHost));
  }
  SYNC_BATCH_NORM_CUDA_CHECK(cudaFree(saveSum_ptr));
  SYNC_BATCH_NORM_CUDA_CHECK(cudaFree(saveVarN_ptr));
  MSHADOW_CUDA_POST_KERNEL_CHECK(SyncBatchNormalizationUpdateOutput);
}


template<typename DType, typename AccReal>
static void SyncBatchNormalizationBackward_bk(mshadow::Stream<gpu> *s,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &out_grad,
                                       const std::vector<TBlob> &in_data,
                                       const std::vector<TBlob> &out_data,
                                       const std::vector<TBlob> &in_grad,
                                       const std::vector<TBlob> &aux_states,
                                       const uint32_t flags,
                                       double momentum,
                                       double eps) {
  DeviceTensor3 input = devicetensor<DType, 3>(in_data[sync_batchnorm::kData]);
  DeviceTensor3 gradOutput = devicetensor<DType, 3>(out_grad[sync_batchnorm::kOut]);
  DeviceTensor3 gradInput = devicetensor<DType, 3>(in_grad[sync_batchnorm::kData]);
  DeviceTensor1 gradWeight = devicetensor<AccReal, 1>(in_grad[sync_batchnorm::kGamma]);
  DeviceTensor1 gradBias = devicetensor<AccReal, 1>(in_grad[sync_batchnorm::kBeta]);
  DeviceTensor1 weight = devicetensor<AccReal, 1>(in_data[sync_batchnorm::kGamma]);
  DeviceTensor1 runningMean = devicetensor<AccReal, 1>(aux_states[sync_batchnorm::kMovingMean]);
  DeviceTensor1 runningVar = devicetensor<AccReal, 1>(aux_states[sync_batchnorm::kMovingVar]);
  DeviceTensor1 saveMean = devicetensor<AccReal, 1>(out_data[sync_batchnorm::kMean]);
  DeviceTensor1 saveInvStd = devicetensor<AccReal, 1>(out_data[sync_batchnorm::kVar]);

  DCHECK_GT(weight.numElements(), 0);

  dim3 blocks(gradOutput.ChannelCount());
  dim3 threads(getNumThreads(gradOutput.SpatialSize()));
  
  BatchNormalizationBackwardKernel<DType, AccReal, DeviceTensor1, DeviceTensor3>
    <<< blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >>> (
    input, gradOutput, gradInput, gradWeight, gradBias, weight, runningMean, runningVar,
      saveMean, saveInvStd, flags, momentum, eps);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SyncBatchNormalizationBackward);
}

template<typename DType, typename AccReal>
static void SyncBatchNormalizationBackward(mshadow::Stream<gpu> *s,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &out_grad,
                                       const std::vector<TBlob> &in_data,
                                       const std::vector<TBlob> &out_data,
                                       const std::vector<TBlob> &in_grad,
                                       const std::vector<TBlob> &aux_states,
                                       const uint32_t flags,
                                       double momentum,
                                       double eps,
                                       Sync_Batch_Thread_Comm<gpu, DType, AccReal> * const comm_instance) {
  DeviceTensor3 input = devicetensor<DType, 3>(in_data[sync_batchnorm::kData]);
  DeviceTensor3 gradOutput = devicetensor<DType, 3>(out_grad[sync_batchnorm::kOut]);
  DeviceTensor3 gradInput = devicetensor<DType, 3>(in_grad[sync_batchnorm::kData]);
  DeviceTensor1 gradWeight = devicetensor<AccReal, 1>(in_grad[sync_batchnorm::kGamma]);
  DeviceTensor1 gradBias = devicetensor<AccReal, 1>(in_grad[sync_batchnorm::kBeta]);
  DeviceTensor1 weight = devicetensor<AccReal, 1>(in_data[sync_batchnorm::kGamma]);
  DeviceTensor1 runningMean = devicetensor<AccReal, 1>(aux_states[sync_batchnorm::kMovingMean]);
  DeviceTensor1 runningVar = devicetensor<AccReal, 1>(aux_states[sync_batchnorm::kMovingVar]);
  DeviceTensor1 saveMean = devicetensor<AccReal, 1>(out_data[sync_batchnorm::kMean]);
  DeviceTensor1 saveInvStd = devicetensor<AccReal, 1>(out_data[sync_batchnorm::kVar]);
  DCHECK_GT(weight.numElements(), 0);

  dim3 blocks(gradOutput.ChannelCount());
  dim3 threads(getNumThreads(gradOutput.SpatialSize()));
  
  AccReal* gradOutputSum_ptr = NULL;
  SYNC_BATCH_NORM_CUDA_CHECK(cudaMalloc(&gradOutputSum_ptr, sizeof(AccReal) * gradOutput.ChannelCount()));
  DeviceTensor1 gradOutputSum(gradOutputSum_ptr, nullptr);
  gradOutputSum.size_[0] = gradOutput.ChannelCount();
    
  AccReal* dotP_ptr = NULL;
  SYNC_BATCH_NORM_CUDA_CHECK(cudaMalloc(&dotP_ptr, sizeof(AccReal) * gradOutput.ChannelCount()));
  DeviceTensor1 dotP(dotP_ptr, nullptr);
  dotP.size_[0] = gradOutput.ChannelCount();
  
  SyncBatchNormalizationGradBackwardKernel<DType, AccReal, DeviceTensor1, DeviceTensor3>
    <<< blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >>> (
    input, gradOutput, gradOutputSum, dotP, weight, runningMean, runningVar,
      saveMean, saveInvStd, flags, momentum, eps);
  
  // Compute reduce of gradOutputSum
  int gradOutputSum_inst_sum = 0;
  std::vector<AccReal> gradOutputSum_host(gradOutputSum.size_[0]);
  std::vector<AccReal> gradOutputSum_output(gradOutputSum.size_[0]);
  SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(&gradOutputSum_host[0],
                              gradOutputSum.dptr_,
                              sizeof(AccReal) * gradOutput.ChannelCount(),
                              cudaMemcpyDeviceToHost));
  comm_instance->reduce(gradOutputSum_host, gradOutputSum_output, 
                        gradOutput.getSize(0) * gradOutput.getSize(2), 
                        gradOutputSum_inst_sum, std::this_thread::get_id());
                            
  SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(gradOutputSum.dptr_, &gradOutputSum_output[0], sizeof(AccReal) * gradOutputSum.size_[0],
                                cudaMemcpyHostToDevice));
   
  
  /*
  for(int i = 0;i < gradOutputSum.size_[0]; i++)
  {
      printf("id:%lld, comm_key:%d, barrier_num:%d, i:%d, inst_cnt:%d, gradOutputSum_host:%.6f, gradOutputSum_output:%.6f\n", 
        std::this_thread::get_id(), comm_key, Sync_Batch_Thread_Comm<gpu, DType, AccReal>::getInstance(comm_key)->get_barrier_max_thread_num(), 
        i, gradOutputSum_inst_sum, gradOutputSum_host[i], gradOutputSum_output[i]);
  }
  */
  
  // Compute reduce of dotP
  int dotP_inst_sum = 0;
  
  std::vector<AccReal> dotP_host(dotP.size_[0]);
  std::vector<AccReal> dotP_output(dotP.size_[0]);

  SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(&dotP_host[0],
                              dotP.dptr_,
                              sizeof(AccReal) * dotP.size_[0],
                              cudaMemcpyDeviceToHost));
  comm_instance->reduce(dotP_host, dotP_output, gradOutput.getSize(0) * gradOutput.getSize(2), dotP_inst_sum, std::this_thread::get_id());
  SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(dotP.dptr_, &dotP_output[0], sizeof(AccReal) * dotP.size_[0],
                                cudaMemcpyHostToDevice));
  
  
  SyncBatchNormalizationBackwardKernel<DType, AccReal, DeviceTensor1, DeviceTensor3>
    <<< blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s) >>> (
    input, gradOutput, gradInput, gradWeight, gradBias, gradOutputSum, dotP, dotP_inst_sum, weight, runningMean, runningVar,
      saveMean, saveInvStd, flags, momentum, eps);
  
  /*
  std::vector<AccReal> gradWeight_host(gradWeight.size_[0]);
  SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(&gradWeight_host[0],
                              gradWeight.dptr_,
                              sizeof(AccReal) * gradWeight.size_[0],
                              cudaMemcpyDeviceToHost));
  
  std::vector<AccReal> gradBias_host(gradBias.size_[0]);
  SYNC_BATCH_NORM_CUDA_CHECK(cudaMemcpy(&gradBias_host[0],
                              gradBias.dptr_,
                              sizeof(AccReal) * gradBias.size_[0],
                              cudaMemcpyDeviceToHost));                             
  for(int i = 0; i < gradWeight.size_[0]; i++)
  {
      printf("id:%lld, comm_key:%d, barrier_num:%d, i:%d, inst_cnt:%d, gradWeight_host:%.6f,gradBias_host:%.6f\n", 
          std::this_thread::get_id(), comm_key,
          Sync_Batch_Thread_Comm<gpu, DType, AccReal>::getInstance(comm_key)->get_barrier_max_thread_num(), 
          i, gradWeight_host[i], gradBias_host[i]);
  }
  printf("------------------\n");
  */
  SYNC_BATCH_NORM_CUDA_CHECK(cudaFree(dotP_ptr));
  SYNC_BATCH_NORM_CUDA_CHECK(cudaFree(gradOutputSum_ptr));
  MSHADOW_CUDA_POST_KERNEL_CHECK(SyncBatchNormalizationBackward);
}


}  // namespace cuda
}  // namespace sync_batchnorm

template<typename xpu, typename DType, typename AccReal>
static inline uint32_t SetupFlags(const OpContext &ctx,
                                  const SyncBatchNormParam& params,
                                  const std::vector<OpReqType> &req) {
  uint32_t flags = 0;
  flags |= ctx.is_train ? IS_TRAINING_FLAG : 0;
  flags |= params.fix_gamma ? FIX_GAMMA_FLAG : 0;
  flags |= params.use_global_stats ? USE_GLOBAL_STATS_FLAG : 0;
  if (SyncBatchNormOp<xpu, DType, AccReal>::IsWriting(req[sync_batchnorm::kData])) {
    flags |= WRITE_DATA_FLAG;
  }
  if (SyncBatchNormOp<xpu, DType, AccReal>::IsWriting(req[sync_batchnorm::kGamma])) {
    flags |= WRITE_GAMMA_FLAG;
  }
  if (SyncBatchNormOp<xpu, DType, AccReal>::IsWriting(req[sync_batchnorm::kBeta])) {
    flags |= WRITE_BETA_FLAG;
  }
  return flags;
}

/*! \brief Forward batch-norm pass on GPU */
template<typename xpu, typename DType, typename AccReal>
void SyncBatchNormOp<xpu, DType, AccReal>::DoForward(mshadow::Stream<gpu> *stream,
                                                 const OpContext &ctx,
                                                 const std::vector<TBlob> &in_data,
                                                 const std::vector<OpReqType> &req,
                                                 const std::vector<TBlob> &out_data,
                                                 const std::vector<TBlob> &aux_states) {

  sync_batchnorm::cuda::SyncBatchNormalizationUpdateOutput<DType, AccReal>(
    stream,
    ctx,
    in_data,
    out_data,
    aux_states,
    SetupFlags<xpu, DType, AccReal>(ctx, param_, req),
    param_.momentum,
    param_.eps,
    comm_instance);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SyncBatchNormOp_DoForward_gpu);
}

/*! \brief Backward batch-norm pass on GPU */
template<typename xpu, typename DType, typename AccReal>
void SyncBatchNormOp<xpu, DType, AccReal>::DoBackward(mshadow::Stream<gpu> *stream,
                                                  const OpContext &ctx,
                                                  const std::vector<TBlob> &out_grad,
                                                  const std::vector<TBlob> &in_data,
                                                  const std::vector<TBlob> &out_data,
                                                  const std::vector<OpReqType> &req,
                                                  const std::vector<TBlob> &in_grad,
                                                  const std::vector<TBlob> &aux_states) {
  sync_batchnorm::cuda::SyncBatchNormalizationBackward<DType, AccReal>(
    stream,
    ctx,
    out_grad,
    in_data,
    out_data,
    in_grad,
    aux_states,
    SetupFlags<xpu, DType, AccReal>(ctx, param_, req),
    param_.momentum,
    param_.eps,
    comm_instance);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SyncBatchNormOp_DoBackward_gpu);
}

/*! \brief Create GPU operator for batch normalization */
template<>
Operator *CreateOp<gpu>(const SyncBatchNormParam& param, const int dtype, const TShape& shape) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype,
                              DType,
                              AccReal,
                              { op = new SyncBatchNormOp<gpu, DType, AccReal>(param); });
  return op;
}

}  // namespace op
}  // namespace mxnet
