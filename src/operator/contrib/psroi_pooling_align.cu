/*!
* Copyright (c) 2017 by Contributors
* \file psroi_pooling.cu
* \brief psroi pooling operator
* \author Yi Li, Guodong Zhang
*/
#include "./psroi_pooling_align-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

#define PSROIPOOLINGALIGN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace mshadow {
  namespace cuda {
    template <typename DType>
    __device__ DType bilinear_interp(
      const DType* data,
      const DType x,
      const DType y,
      const int width,
      const int height) {
      int x1 = floor(x);
      int x2 = ceil(x);
      int y1 = floor(y);
      int y2 = ceil(y);
      DType dist_x = static_cast<DType>(x - x1);
      DType dist_y = static_cast<DType>(y - y1);
      DType value11 = data[y1*width + x1];
      DType value12 = data[y2*width + x1];
      DType value21 = data[y1*width + x2];
      DType value22 = data[y2*width + x2];
      DType value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12
        + dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;
      return value;
    }

    template <typename DType>
    __global__ void PSROIPoolAlignForwardKernel(
      const int count,
      const DType* bottom_data,
      const DType spatial_scale,
      const int channels,
      const int height, const int width,
      const int pooled_height, const int pooled_width,
      const DType* bottom_rois,
      const int output_dim,
      const int group_size,
      const int sample_per_part,
      DType* top_data,
      DType* top_count) {
      CUDA_KERNEL_LOOP(index, count) {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const DType* offset_bottom_rois = bottom_rois + n * 5;
        int roi_batch_ind = offset_bottom_rois[0];
        DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
        DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
        DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
        DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

        // Force too small ROIs to be 1x1
        DType roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
        DType roi_height = max(roi_end_h - roi_start_h, 0.1);

        // Compute w and h at bottom
        DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
        DType bin_size_w = roi_width / static_cast<DType>(pooled_width);
        DType sub_bin_size_h = bin_size_h / static_cast<DType>(sample_per_part);
        DType sub_bin_size_w = bin_size_w / static_cast<DType>(sample_per_part);

        const DType* offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
        DType wstart = pw*bin_size_w + roi_start_w;
        DType hstart = ph*bin_size_h + roi_start_h;
        DType sum = 0;
        int count = 0;

        int gw = floor(static_cast<DType>(pw)* group_size / pooled_width);
        int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
        gw = min(max(gw, 0), group_size - 1);
        gh = min(max(gh, 0), group_size - 1);
        for (int ih = 0; ih < sample_per_part; ih++) {
          for (int iw = 0; iw < sample_per_part; iw++) {
            DType w = wstart + iw*sub_bin_size_w;
            DType h = hstart + ih*sub_bin_size_h;
            // bilinear interpolation
            if (w<-0.5 || w>width - 0.5 || h<-0.5 || h>height - 0.5) {
              continue;
            }
            w = min(max(w, 0.), width - 1.);
            h = min(max(h, 0.), height - 1.);
            int c = (ctop*group_size + gh)*group_size + gw;
            DType val = bilinear_interp(offset_bottom_data + c*height*width, w, h, width, height);
            sum += val;
            count++;
          }
        }
        top_data[index] = (count == 0) ? DType(0) : sum / DType(count);
        top_count[index] = count;
      }
    }

    template<typename DType>
    inline void PSROIPoolAlignForward(const Tensor<gpu, 4, DType> &out,
      const Tensor<gpu, 4, DType> &data,
      const Tensor<gpu, 2, DType> &bbox,
      const Tensor<gpu, 4, DType> &top_count,
      const float spatial_scale,
      const int output_dim,
      const int group_size,
      const int sample_per_part) {
      // LOG(INFO) << "PSROIPoolAlignForward";
      const DType *bottom_data = data.dptr_;
      const DType *bottom_rois = bbox.dptr_;
      DType *top_data = out.dptr_;
      DType *top_count_data = top_count.dptr_;
      const int count = out.shape_.Size();
      const int channels = data.size(1);
      const int height = data.size(2);
      const int width = data.size(3);
      const int pooled_height = out.size(2); // in this version, pooled_size == group_size by default
      const int pooled_width = out.size(3);

      cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
      PSROIPoolAlignForwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
        kBaseThreadNum, 0, stream >> >(
          count, bottom_data, spatial_scale, channels, height, width,
          pooled_height, pooled_width, bottom_rois, output_dim, group_size,
          sample_per_part, top_data, top_count_data);
      PSROIPOOLINGALIGN_CUDA_CHECK(cudaPeekAtLastError());
    }


    template <typename DType>
    __global__ void PSROIPoolAlignBackwardAccKernel(
      const int count,
      const DType* top_diff,
      const DType* top_count,
      const int num_rois,
      const DType spatial_scale,
      const int channels,
      const int height, const int width,
      const int pooled_height, const int pooled_width,
      const int output_dim,
      const int group_size,
      const int sample_per_part,
      DType* bottom_diff,
      const DType* bottom_data,
      const DType* bottom_rois) {
      CUDA_KERNEL_LOOP(index, count) {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const DType* offset_bottom_rois = bottom_rois + n * 5;
        int roi_batch_ind = offset_bottom_rois[0];
        DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
        DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
        DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
        DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

        // Force too small ROIs to be 1x1
        DType roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
        DType roi_height = max(roi_end_h - roi_start_h, 0.1);

        // Compute w and h at bottom
        DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
        DType bin_size_w = roi_width / static_cast<DType>(pooled_width);
        DType sub_bin_size_h = bin_size_h / static_cast<DType>(sample_per_part);
        DType sub_bin_size_w = bin_size_w / static_cast<DType>(sample_per_part);

        DType wstart = pw*bin_size_w + roi_start_w;
        DType hstart = ph*bin_size_h + roi_start_h;
        if (top_count[index] <= 0) {
          continue;
        }
        DType diff_val = top_diff[index] / top_count[index];

        int gw = floor(static_cast<DType>(pw)* group_size / pooled_width);
        int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
        gw = min(max(gw, 0), group_size - 1);
        gh = min(max(gh, 0), group_size - 1);
        for (int ih = 0; ih < sample_per_part; ih++) {
          for (int iw = 0; iw < sample_per_part; iw++) {
            DType w = wstart + iw*sub_bin_size_w;
            DType h = hstart + ih*sub_bin_size_h;
            // bilinear interpolation
            if (w<-0.5 || w>width - 0.5 || h<-0.5 || h>height - 0.5) {
              continue;
            }
            w = min(max(w, 0.), width - 1.);
            h = min(max(h, 0.), height - 1.);
            int c = (ctop*group_size + gh)*group_size + gw;
            // backward on feature
            int x0 = floor(w);
            int x1 = ceil(w);
            int y0 = floor(h);
            int y1 = ceil(h);
            DType dist_x = w - x0, dist_y = h - y0;
            DType q00 = (1 - dist_x)*(1 - dist_y);
            DType q01 = (1 - dist_x)*dist_y;
            DType q10 = dist_x*(1 - dist_y);
            DType q11 = dist_x*dist_y;
            DType* offset_bottom_data_diff = bottom_diff + roi_batch_ind * channels * height * width + c * height * width;
            atomicAdd(offset_bottom_data_diff + y0*width + x0, q00*diff_val);
            atomicAdd(offset_bottom_data_diff + y1*width + x0, q01*diff_val);
            atomicAdd(offset_bottom_data_diff + y0*width + x1, q10*diff_val);
            atomicAdd(offset_bottom_data_diff + y1*width + x1, q11*diff_val);
          }
        }
      }
    }


    template<typename DType>
    inline void PSROIPoolAlignBackwardAcc(const Tensor<gpu, 4, DType> &in_grad,
      const Tensor<gpu, 4, DType> &out_grad,
      const Tensor<gpu, 4, DType> &data,
      const Tensor<gpu, 2, DType> &bbox,
      const Tensor<gpu, 4, DType> &top_count,
      const float spatial_scale,
      const int output_dim,
      const int group_size,
      const int sample_per_part) {
      // LOG(INFO) << "PSROIPoolAlignBackward";
      const DType *top_diff = out_grad.dptr_;
      const DType *bottom_data = data.dptr_;
      const DType *bottom_rois = bbox.dptr_;
      DType *bottom_diff = in_grad.dptr_;
      DType *top_count_data = top_count.dptr_;
      const int count = out_grad.shape_.Size();
      const int num_rois = bbox.size(0);
      const int channels = in_grad.size(1);
      const int height = in_grad.size(2);
      const int width = in_grad.size(3);
      const int pooled_height = out_grad.size(2);
      const int pooled_width = out_grad.size(3);

      cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
      PSROIPoolAlignBackwardAccKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
        kBaseThreadNum, 0, stream >> >(
          count, top_diff, top_count_data, num_rois, spatial_scale, channels, height, width,
          pooled_height, pooled_width, output_dim, group_size, sample_per_part,
          bottom_diff, bottom_data, bottom_rois);
      PSROIPOOLINGALIGN_CUDA_CHECK(cudaPeekAtLastError());
    }

  }  // namespace cuda

  template<typename DType>
  inline void PSROIPoolAlignForward(const Tensor<gpu, 4, DType> &out,
    const Tensor<gpu, 4, DType> &data,
    const Tensor<gpu, 2, DType> &bbox,
    const Tensor<gpu, 4, DType> &top_count,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int sample_per_part) {
    cuda::PSROIPoolAlignForward(out, data, bbox, top_count, spatial_scale, output_dim, group_size, sample_per_part);
  }

  template<typename DType>
  inline void PSROIPoolAlignBackwardAcc(const Tensor<gpu, 4, DType> &in_grad,
    const Tensor<gpu, 4, DType> &out_grad,
    const Tensor<gpu, 4, DType> &data,
    const Tensor<gpu, 2, DType> &bbox,
    const Tensor<gpu, 4, DType> &top_count,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int sample_per_part) {
    cuda::PSROIPoolAlignBackwardAcc(in_grad, out_grad, data, bbox, top_count, spatial_scale, output_dim,
      group_size, sample_per_part);
  }

}  // namespace mshadow


namespace mxnet {
  namespace op {

    template<>
    Operator* CreateOp<gpu>(PSROIPoolingAlignParam param, int dtype) {
      Operator* op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new PSROIPoolingAlignOp<gpu, DType>(param);
      });
      return op;
    }

  }  // namespace op
}  // namespace mxnet