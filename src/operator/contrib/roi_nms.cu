/*!
 * Copyright (c) 2015 by Contributors
 * \file roi_nms.cu
 * \brief RoiNms Operator
 * \author Zheng Zhang
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <iostream>

#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./roi_nms-inl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {
namespace roi_nms_utils {
// scores are (n_boxes, 1)
// proposals are (n_boxes, 4)
// workspace_proposals are (n_boxes, 5)
template<typename Dtype>
__global__ void ProposalGridKernel(const int count,
                                   const Dtype* scores,
								   const Dtype* proposals, 
                                   Dtype* workspace_proposals) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    workspace_proposals[index * 5 + 0] = proposals[index * 5 + 1];
    workspace_proposals[index * 5 + 1] = proposals[index * 5 + 2];
    workspace_proposals[index * 5 + 2] = proposals[index * 5 + 3];
    workspace_proposals[index * 5 + 3] = proposals[index * 5 + 4];
    workspace_proposals[index * 5 + 4] = scores[index];
  }
}

// copy score and init order
// dets (n, 5); score (n, ); order (n, )
// count should be n (total anchors or proposals)
template<typename Dtype>
__global__ void CopyScoreKernel(const int count,
                                const Dtype* dets,
                                Dtype* score,
                                int* order) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    score[index] = dets[index * 5 + 4];
    order[index] = index;
  }
}

// reorder proposals according to order and keep the top_n proposals
// prev_dets (n, 5); order (n, ); dets (n, 5)
// count should be output anchor numbers (top_n)
template<typename Dtype>
__global__ void ReorderProposalsKernel(const int count,
                                       const Dtype* prev_dets,
                                       const int* order,
                                       Dtype* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    const int order_i = order[index];
    for (int j = 0; j < 5; j ++) {
      dets[index * 5 + j] = prev_dets[order_i * 5 + j];
    }
  }
}

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, uint64_t *dev_mask) {
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (roi_nms_utils::devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _nms(const mshadow::Tensor<gpu, 2>& boxes,
          const float nms_overlap_thresh,
          int *keep,
          int *num_out) {
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int boxes_num = boxes.size(0);
  const int boxes_dim = boxes.size(1);

  float* boxes_dev = boxes.dptr_;
  uint64_t* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  FRCNN_CUDA_CHECK(cudaMalloc(&mask_dev,
                              boxes_num * col_blocks * sizeof(uint64_t)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  roi_nms_utils::nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);
  FRCNN_CUDA_CHECK(cudaPeekAtLastError());
  std::vector<uint64_t> mask_host(boxes_num * col_blocks);
  FRCNN_CUDA_CHECK(cudaMemcpy(&mask_host[0],
                              mask_dev,
                              sizeof(uint64_t) * boxes_num * col_blocks,
                              cudaMemcpyDeviceToHost));

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep[num_to_keep++] = i;
      uint64_t *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  FRCNN_CUDA_CHECK(cudaFree(mask_dev));
}

// copy proposals to output
// dets (top_n, 5); keep (top_n, ); out (top_n, )
// count should be top_n (total anchors or proposals)
template<typename Dtype>
__global__ void PrepareOutput(const int count,
                              const Dtype* dets,
                              const int* keep,
                              const int out_size,
                              Dtype* out,
                              Dtype* score) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    out[index * 5] = 0;
    if (index < out_size) {
      int keep_i = keep[index];
      for (int j = 0; j < 4; ++j) {
        out[index * 5 + j + 1] = dets[keep_i * 5 + j];
      }
      score[index] = dets[keep_i * 5 + 4];
    } else {
      int keep_i = keep[index % out_size];
      for (int j = 0; j < 4; ++j) {
        out[index * 5 + j + 1] = dets[keep_i * 5 + j];
      }
      score[index] = dets[keep_i * 5 + 4];
    }
  }
}

}  // namespace roi_nms_utils
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu>
class RoiNmsGPUOp : public Operator{
 public:
  explicit RoiNmsGPUOp(RoiNmsParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda;
	
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[roi_nms::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 2> scores = in_data[roi_nms::kClsProb].get<xpu, 2, real_t>(s);
	Tensor<xpu, 2> proposals = in_data[roi_nms::kBBoxPred].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out = out_data[roi_nms::kOut].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out_score = out_data[roi_nms::kScore].get<xpu, 2, real_t>(s);

    int count = scores.shape_[0];
    // set to -1 for max
    int pre_nms_top_n = (param_.pre_nms_top_n > 0) ? param_.pre_nms_top_n : count;
    pre_nms_top_n = std::min(pre_nms_top_n, count);
    int post_nms_top_n = std::min(param_.post_nms_top_n, pre_nms_top_n);

    float* workspace_proposals_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&workspace_proposals_ptr, sizeof(float) * count * 5));
    Tensor<xpu, 2> workspace_proposals(workspace_proposals_ptr, Shape2(count, 5));

    // Copy proposals to a mesh grid
    dim3 dimGrid((count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
    dim3 dimBlock(kMaxThreadsPerBlock);
    CheckLaunchParam(dimGrid, dimBlock, "ProposalGrid");
    roi_nms_utils::ProposalGridKernel<<<dimGrid, dimBlock>>>(
      count, scores.dptr_, proposals.dptr_, workspace_proposals.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // Copy score to a continuous memory
    float* score_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&score_ptr, sizeof(float) * count));
    Tensor<xpu, 1> score(score_ptr, Shape1(count));
    int* order_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&order_ptr, sizeof(int) * count));
    Tensor<xpu, 1, int> order(order_ptr, Shape1(count));

    CheckLaunchParam(dimGrid, dimBlock, "CopyScore");
    roi_nms_utils::CopyScoreKernel<<<dimGrid, dimBlock>>>(
      count, workspace_proposals.dptr_, score.dptr_, order.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // argsort score, save order
    thrust::stable_sort_by_key(thrust::device,
                               score.dptr_,
                               score.dptr_ + score.size(0),
                               order.dptr_,
                               thrust::greater<real_t>());
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // Reorder proposals according to order
    float* workspace_ordered_proposals_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&workspace_ordered_proposals_ptr,
                                sizeof(float) * pre_nms_top_n * 5));
    Tensor<xpu, 2> workspace_ordered_proposals(workspace_ordered_proposals_ptr,
                                               Shape2(pre_nms_top_n, 5));

    dimGrid.x = (pre_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    CheckLaunchParam(dimGrid, dimBlock, "ReorderProposals");
    roi_nms_utils::ReorderProposalsKernel<<<dimGrid, dimBlock>>>(
      pre_nms_top_n, workspace_proposals.dptr_, order.dptr_, workspace_ordered_proposals.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());
	
    FRCNN_CUDA_CHECK(cudaFree(workspace_proposals_ptr));
    FRCNN_CUDA_CHECK(cudaFree(score_ptr));
    FRCNN_CUDA_CHECK(cudaFree(order_ptr));	
	
    // perform nms
    std::vector<int> _keep(workspace_ordered_proposals.size(0));
    
    int out_size = 0;
    roi_nms_utils::_nms(workspace_ordered_proposals,
	  param_.threshold,
	  &_keep[0],
	  &out_size);
	
    // copy nms result to gpu
    int* keep;
    FRCNN_CUDA_CHECK(cudaMalloc(&keep, sizeof(int) * _keep.size()));
    FRCNN_CUDA_CHECK(cudaMemcpy(keep, &_keep[0], sizeof(int) * _keep.size(),
                                cudaMemcpyHostToDevice));

    // copy results after nms
    dimGrid.x = (post_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    CheckLaunchParam(dimGrid, dimBlock, "PrepareOutput");
    roi_nms_utils::PrepareOutput<<<dimGrid, dimBlock>>>(
      post_nms_top_n, workspace_ordered_proposals.dptr_, keep, out_size,
      out.dptr_, out_score.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // free temporary memory
    FRCNN_CUDA_CHECK(cudaFree(keep));
    FRCNN_CUDA_CHECK(cudaFree(workspace_ordered_proposals_ptr));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 3);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[roi_nms::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[roi_nms::kBBoxPred].get<xpu, 4, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[roi_nms::kClsProb], 0);
    Assign(gbbox, req[roi_nms::kBBoxPred], 0);
  }

 private:
  RoiNmsParam param_;
};  // class ProposalGPUOp

template<>
Operator* CreateOp<gpu>(RoiNmsParam param) {
  return new RoiNmsGPUOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
