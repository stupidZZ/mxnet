#include "./bilinear_resize-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

#define BilinearResize_CUDA_CHECK(condition) \
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
    template <typename Dtype>
    __global__ void ForwardFeatures(const int nthreads, const int num, const int channels, const int bottomwidth, const int bottomheight,
      const int topheight, const int topwidth, const int bot_countpernum, const int bot_numstride, const float widthScale,
      const float heightScale, const int widthKernal, const int heightKernel, const Dtype* src_data, Dtype* dest_data) {
      CUDA_KERNEL_LOOP(index, nthreads) {

        int destx = index % topwidth; //w-pos
        int desty = (index / topwidth) % topheight; //h-pos

        int cn = (index / topwidth / topheight);
        int c = cn % channels; //channel
        int n = cn / channels; //num

        //Compute source center pos in topdiff
        float botx = (float)(destx + 0.5) * widthScale - 0.5; // \in [0.0, (topwidth-1)]
        float boty = (float)(desty + 0.5) * heightScale - 0.5;
        //printf("dest x,y[%d,%d] n:%d c:%d = idx:%d/%d | bot x,y[%d,%d]\n", destx, desty, n, c, index, nthreads, ibotx, iboty);  

        // Accumulate in range around that point:
        int botidxoffcn = (bot_numstride*n) + (bottomwidth*bottomheight*c);

        int start_botx = max(0, (int)ceil(botx - widthKernal));
        int end_botx = min(bottomwidth - 1, (int)floor(botx + widthKernal));
        int start_boty = max(0, (int)ceil(boty - heightKernel));
        int end_boty = min(bottomheight - 1, (int)floor(boty + heightKernel));

        float accum_value = 0;
        float accum_weight = 0;

        for (int by = start_boty; by <= end_boty; by++)  {
          int botidxoffycn = by * bottomwidth + botidxoffcn;
          for (int bx = start_botx; bx <= end_botx; bx++)  {
            float sample = src_data[bx + botidxoffycn];
            float weight = (1.0f - abs((float)bx - botx) / widthKernal) * (1.0f - abs((float)by - boty) / heightKernel);
            accum_value += sample * weight;
            accum_weight += weight;
          }
        }
        dest_data[index] = accum_value / accum_weight;
      }
    }

    template <typename Dtype>
    __global__ void BackwardGradient(const int nthreads, const int num, const int channels, const int bottomwidth, const int bottomheight,
      const int topheight, const int topwidth, const int bot_countpernum, const int bot_numstride, const float widthScale,
      const float heightScale, const int widthKernal, const int heightKernel, Dtype* src_diff, const Dtype* dest_diff) {
      CUDA_KERNEL_LOOP(index, nthreads) {
        int destx = index % topwidth; //w-pos
        int desty = (index / topwidth) % topheight; //h-pos

        int cn = (index / topwidth / topheight);
        int c = cn % channels; //channel
        int n = cn / channels; //num

        //Compute source center pos in topdiff
        float botx = (float)(destx + 0.5) * widthScale - 0.5; // \in [0.0, (topwidth-1)]
        float boty = (float)(desty + 0.5) * heightScale - 0.5;
        //printf("dest x,y[%d,%d] n:%d c:%d = idx:%d/%d | bot x,y[%d,%d]\n", destx, desty, n, c, index, nthreads, ibotx, iboty);  

        // Accumulate in range around that point:
        int botidxoffcn = (bot_numstride*n) + (bottomwidth*bottomheight*c);

        int start_botx = max(0, (int)ceil(botx - widthKernal));
        int end_botx = min(bottomwidth - 1, (int)floor(botx + widthKernal));
        int start_boty = max(0, (int)ceil(boty - heightKernel));
        int end_boty = min(bottomheight - 1, (int)floor(boty + heightKernel));

        float accum_weight = 0;

        for (int by = start_boty; by <= end_boty; by++)  {
          for (int bx = start_botx; bx <= end_botx; bx++)  {
            float weight = (1.0f - abs((float)bx - botx) / widthKernal) * (1.0f - abs((float)by - boty) / heightKernel);
            accum_weight += weight;
          }
        }

        for (int by = start_boty; by <= end_boty; by++)  {
          int botidxoffycn = by * bottomwidth + botidxoffcn;
          for (int bx = start_botx; bx <= end_botx; bx++)  {
            float weight = (1.0f - abs((float)bx - botx) / widthKernal) * (1.0f - abs((float)by - boty) / heightKernel);
            atomicAdd(src_diff + bx + botidxoffycn, dest_diff[index] * weight / accum_weight);
          }
        }
      }
    }

  }  // namespace cuda


  template <typename DType>
  inline void BilinearResizeForward(const Tensor<gpu, 4, DType> &out,
    const Tensor<gpu, 4, DType> &data) {

    DType* top_data = out.dptr_; // dest
    int topwidth = out.size(3);
    int topheight = out.size(2);
    int topchannels = out.size(1);
    int topcount = out.shape_.Size();

    //LOG(INFO) << "Metrics: Tnum " << top[0]->num() << " Tchan " << topchannels << " Tw " << topwidth << " Th " << topheight;

    const DType* bottom_data = data.dptr_; // source
    //LOG(INFO) << "Got ptr to bottom ";

    int bottomnum = data.size(0);
    int bottomchannels = data.size(1);
    int bottomwidth = data.size(3);
    int bottomheight = data.size(2);
    int bottomcount = data.shape_.Size();

    //if (bottomwidth != topwidth || bottomheight != topheight) {

      // From bottom to top

      int bot_countpernum = bottomwidth * bottomheight * bottomchannels;
      int bot_numstride = bottomwidth * bottomheight * bottomchannels;

      float widthScale = (float)(bottomwidth) / (float)(topwidth); // e.g. 2.0 if bottom pixeldist half compared to top. 
      float heightScale = (float)(bottomheight) / (float)(topheight);

      int widthKernal = ceil(widthScale / 2);
      int heightKernal = ceil(heightScale / 2);

      cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);

      cuda::ForwardFeatures<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(topcount), cuda::kBaseThreadNum, 0, stream >> >(
        topcount,
        bottomnum, bottomchannels, bottomwidth, bottomheight,
        topheight, topwidth, bot_countpernum, bot_numstride,
        widthScale, heightScale, widthKernal, heightKernal,
        bottom_data, top_data);

      BilinearResize_CUDA_CHECK(cudaPeekAtLastError());

  }


  template <typename DType>
  inline void BilinearResizeBackward(const Tensor<gpu, 4, DType> &in_grad,
    const Tensor<gpu, 4, DType> &data,
    const Tensor<gpu, 4, DType> &out_grad) {

    const DType* top_diff = out_grad.dptr_; // dest
    int topwidth = out_grad.size(3);
    int topheight = out_grad.size(2);
    int topchannels = out_grad.size(1);
    int topcount = out_grad.shape_.Size();

    //LOG(INFO) << "Metrics: Tnum " << top[0]->num() << " Tchan " << topchannels << " Tw " << topwidth << " Th " << topheight;

    DType* bottom_diff = in_grad.dptr_; // source
    //LOG(INFO) << "Got ptr to bottom ";

    int bottomnum = in_grad.size(0);
    int bottomchannels = in_grad.size(1);
    int bottomwidth = in_grad.size(3);
    int bottomheight = in_grad.size(2);
    int bottomcount = in_grad.shape_.Size();

    //if (bottomwidth != topwidth || bottomheight != topheight) {

      // From bottom to top

      int bot_countpernum = bottomwidth * bottomheight * bottomchannels;
      int bot_numstride = bottomwidth * bottomheight * bottomchannels;

      float widthScale = (float)(bottomwidth) / (float)(topwidth); // e.g. 2.0 if bottom pixeldist half compared to top. 
      float heightScale = (float)(bottomheight) / (float)(topheight);

      int widthKernal = ceil(widthScale / 2);
      int heightKernal = ceil(heightScale / 2);

      cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);

      cuda::BackwardGradient<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(topcount), cuda::kBaseThreadNum, 0, stream >> >(
        topcount,
        bottomnum, bottomchannels, bottomwidth, bottomheight,
        topheight, topwidth, bot_countpernum, bot_numstride,
        widthScale, heightScale, widthKernal, heightKernal,
        bottom_diff, top_diff);

      BilinearResize_CUDA_CHECK(cudaPeekAtLastError());
  }


}  // namespace mshadow

namespace mxnet {
  namespace op {

    template<>
    Operator* CreateOp<gpu>(BilinearResizeParam param, int dtype) {
      Operator* op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new BilinearResizeOp<gpu, DType>(param);
      });
      return op;
    }

  }  // namespace op
}  // namespace mxnet
