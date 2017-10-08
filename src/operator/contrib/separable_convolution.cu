/*!
* Copyright (c) 2016 by Contributors
* \file separable_convolution.cu
* \brief Xception: Deep Learning with Depthwise Separable Convolutions https://arxiv.org/abs/1610.02357
* \brief separable convolution adjusted from tensorflow code: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/depthwise_conv_op_gpu.cu.cc
* \author Han Hu
*/

//TODO: address the dilation issue
//TODO: training speed-up

#include "./separable_convolution-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"

#define SEPARABLE_CONV_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
  namespace cuda {

    template <typename T>
    __global__ void SeparableConv2dGPUKernelNCHW(const mxnet::op::sepconv::DepthwiseArgs args,
      const T* input, const T* filter,
      T* output, int count) {
      const int in_rows = args.in_rows;
      const int in_cols = args.in_cols;
      const int in_depth = args.in_depth;
      const int filter_rows = args.filter_rows;
      const int filter_cols = args.filter_cols;
      const int depth_multiplier = args.depth_multiplier;
      const int stride_y = args.stride_y;
      const int stride_x = args.stride_x;
      const int pad_rows = args.pad_rows;
      const int pad_cols = args.pad_cols;
      const int dilate_y = args.dilate_y;
      const int dilate_x = args.dilate_x;
      const int out_rows = args.out_rows;
      const int out_cols = args.out_cols;
      const int out_depth = args.out_depth;

      for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x * gridDim.y) {

        //CUDA_1D_KERNEL_LOOP(thread_id, num_outputs) {
        // Compute the indexes of this thread in the output.
        //
        // We want coalesced reads so we make sure that each warp reads
        // a contiguous chunk of memory.
        //
        // THIS IS PROBABLY WRONG, we are not doing coalesced reads
        // into the input, because of the depth multiplier division...
        const int OC = index % out_cols;
        const int OR = (index / out_cols) % out_rows;
        const int OD = (index / out_cols / out_rows) % out_depth;
        const int OB = index / out_cols / out_rows / out_depth;

        // Compute the input depth and the index of depth multiplier
        // based off the output depth index that this thread is
        // computing n.
        const int in_d = OD / depth_multiplier;
        const int multiplier = OD % depth_multiplier;

        // Data is stored in the following format (let's assume we
        // flatten the height and width into one contiguous dimension
        // called "P".
        //
        // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
        // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
        //
        // Each row contains in_depth * in_rows * in_cols values
        // for each sample in the batch.
        //
        // We can further flatten it into:
        //
        // B1C1P1 B1C1P2 .....
        // B1C2P1 B1C2P2 ....
        // B2C1P1 B2C1P2 .....
        // B2C2P1 B2C2P2 ....
        //
        // where each row is a contiguous array of all of the spatial
        // pixels for a given batch and input depth.  The following
        // loop unrolls across the filter dimensions for a given thread,
        // indexing into the filter value and the corresponding input
        // patch.
        //
        // We can compute the index into the patch once right here.
        const int input_offset_temp = (OB * in_depth + in_d) * (in_rows * in_cols);

        // Finally, we can iterate over the spatial dimensions and perform the
        // convolution, writing into the output at the end.
        //
        // We perform an additional optimization, where we can determine
        // whether the patch fits within the image indices statically, and
        // avoid boundary checking within the loop.
        const int input_row_start = OR * stride_y - pad_rows;
        const int input_col_start = OC * stride_x - pad_cols;
        const int input_row_end = input_row_start + (filter_rows - 1) * dilate_y;
        const int input_col_end = input_col_start + (filter_cols - 1) * dilate_x;

        T sum = 0;
        const int filter_offset_temp = (multiplier + depth_multiplier * in_d) * filter_rows * filter_cols;

        if (input_row_start >= 0 && input_col_start >= 0 &&
          input_row_end < in_rows && input_col_end < in_cols) {

          // Loop that doesn't need to check for boundary conditions.
          for (int f_r = 0; f_r < filter_rows; ++f_r) {
            const int in_r = input_row_start + f_r * dilate_y;
            //const int filter_offset_temp = filter_cols * f_r;
            for (int f_c = 0; f_c < filter_cols; ++f_c) {
              const int in_c = input_col_start + f_c * dilate_x;

              const int input_offset =
                (input_offset_temp)+(in_r * in_cols) + in_c;
              
              const int filter_offset = filter_offset_temp + filter_cols * f_r + f_c;
              sum += (*(input + input_offset)) * (*(filter + filter_offset));
            }
          }
        }
        else {
          // Loop that needs to check for boundary conditions.
          for (int f_r = 0; f_r < filter_rows; ++f_r) {
            const int in_r = input_row_start + f_r * dilate_y;
            //const int filter_offset_temp = filter_cols * f_r;
            for (int f_c = 0; f_c < filter_cols; ++f_c) {
              const int in_c = input_col_start + f_c * dilate_x;
              // TODO(vrv): the in_r check can be done outside of this loop;
              // benchmark both methods to determine the better decision.
              if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
                //const int in_c = input_col_start + f_c;

                // input_offset_temp indexes into the start of memory
                // where the spatial data starts.
                const int input_offset =
                  (input_offset_temp)+(in_r * in_cols) + in_c;

                //const int filter_offset =
                //  multiplier + depth_multiplier *
                //  (in_d + in_depth * (f_c + filter_offset_temp));
                const int filter_offset = filter_offset_temp + filter_cols * f_r + f_c;
                sum += (*(input + input_offset)) * (*(filter + filter_offset));
              }
            }
          }
        }
        output[index] = sum;
      }
    }

    template <typename T>
    __global__ void 
      SeparableConv2dBackpropInputGPUKernelNCHW(const mxnet::op::sepconv::DepthwiseArgs args,
        const T* out_backprop,
        const T* filter, T* in_backprop, int count) {
      const int in_rows = args.in_rows;
      const int in_cols = args.in_cols;
      const int in_depth = args.in_depth;
      const int filter_rows = args.filter_rows;
      const int filter_cols = args.filter_cols;
      const int depth_multiplier = args.depth_multiplier;
      const int stride_y = args.stride_y;
      const int stride_x = args.stride_x;
      const int pad_rows = args.pad_rows;
      const int pad_cols = args.pad_cols;
      const int dilate_y = args.dilate_y;
      const int dilate_x = args.dilate_x;
      const int out_rows = args.out_rows;
      const int out_cols = args.out_cols;
      const int out_depth = args.out_depth;

      for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x * gridDim.y) {
        
        const int in_c = index % in_cols;
        const int in_r = (index / in_cols) % in_rows;
        const int in_d = (index / in_cols / in_rows) % in_depth;
        const int b = index / in_depth / in_cols / in_rows;

        T sum = 0;
        if (dilate_x == 1 && dilate_y == 1) {
          const int out_d_start = in_d * depth_multiplier;
          const int out_d_end = out_d_start + depth_multiplier;

          int out_r_start = (in_r - (filter_rows - 1) * dilate_y - 1 + pad_rows + stride_y) / stride_y;
          out_r_start = out_r_start < 0 ? 0 : out_r_start;

          int out_r_end = (in_r + pad_rows) / stride_y;
          out_r_end = out_r_end > out_rows - 1 ? out_rows - 1 : out_r_end;

          int out_c_start = (in_c - (filter_cols - 1) * dilate_x - 1 + pad_cols + stride_x) / stride_x;
          out_c_start = out_c_start < 0 ? 0 : out_c_start;

          int out_c_end = (in_c + pad_cols) / stride_x;
          out_c_end = out_c_end > out_cols - 1 ? out_cols - 1 : out_c_end;

          for (int out_d = out_d_start; out_d < out_d_end; ++out_d) {
            const int filter_offset_temp = out_d * filter_rows * filter_cols;

            for (int out_r = out_r_start; out_r <= out_r_end; out_r += dilate_y) {
              const int f_r = (in_r + pad_rows - out_r * stride_y) / dilate_y;

              //const int temp_filter_offset = filter_cols * f_r;
              for (int out_c = out_c_start; out_c <= out_c_end; out_c += dilate_x) {
                const int f_c = (in_c + pad_cols - out_c * stride_x) / dilate_x;
                //const int filter_offset =
                //  filter_dm + args.depth_multiplier *
                //  (in_d + in_depth * (f_c + temp_filter_offset));

                const int filter_offset = filter_offset_temp + filter_cols * f_r + f_c;

                const int out_backprop_offset =
                  (b * out_depth * out_rows * out_cols) +
                  (out_d * out_rows * out_cols) + (out_r * out_cols) + (out_c);

                sum += (*(out_backprop + out_backprop_offset)) *
                  (*(filter + filter_offset));
              }
            }
          }
        }
        else {
          const int out_d_start = in_d * depth_multiplier;
          const int out_d_end = out_d_start + depth_multiplier;

          //const int out_r_start_t = in_r + pad_rows - (filter_rows - 1) * dilate_y;
          //const int out_c_start_t = in_c + pad_cols - (filter_cols - 1) * dilate_x;
          const int out_r_end_t = in_r + pad_rows;
          const int out_c_end_t = in_c + pad_cols;

          for (int out_d = out_d_start; out_d < out_d_end; ++out_d) {
            const int filter_offset_temp = out_d * filter_rows * filter_cols;
            for (int f_r = 0; f_r < filter_rows; ++f_r) {
              int out_r = out_r_end_t - f_r * dilate_y;

              if (out_r % stride_y == 0) {
                int out_r2 = out_r / stride_y;
                if (out_r2 >= 0 && out_r2 <= out_rows - 1) {
                  for (int f_c = 0; f_c < filter_cols; ++f_c) {
                    int out_c = out_c_end_t - f_c * dilate_x;
                    if (out_c % stride_x == 0) {
                      int out_c2 = out_c / stride_x;
                      if (out_c2 >= 0 && out_c2 <= out_cols - 1) {
                        const int filter_offset = filter_offset_temp + filter_cols * f_r + f_c;

                        const int out_backprop_offset =
                          (b * out_depth * out_rows * out_cols) +
                          (out_d * out_rows * out_cols) + (out_r2 * out_cols) + (out_c2);

                        sum += (*(out_backprop + out_backprop_offset)) *
                          (*(filter + filter_offset));

                      }
                    }

                  }
                }
              }

            }
          }

        }
        const int in_backprop_offset = (b * in_rows * in_cols * in_depth) +
          (in_d * in_rows * in_cols) +
          (in_r * in_cols) + (in_c);
        in_backprop[in_backprop_offset] = sum;
      }
    }

    //warp_size + AUX_SIZE to address bank conflict
    template <typename T, int CHA_SIZE, int WARP_ROW, int WARP_COL, int TIME_Y,  int AUX_SIZE>
    __global__ void SeparableConv2dBackpropFilterGPUKernelNCHW_regular64x64(const int batch, const int depth, const int rows, const int cols,
      const T* input, const T* out_backprop, T* filter_backprop) {

      __shared__ float input_s[(WARP_COL+AUX_SIZE)*WARP_ROW*TIME_Y*CHA_SIZE];
      __shared__ float outback_s[(WARP_COL+AUX_SIZE)*WARP_ROW*TIME_Y*CHA_SIZE];
      __shared__ float temp_s[(WARP_COL+AUX_SIZE)*WARP_ROW*TIME_Y*CHA_SIZE];
      __shared__ float filterback_s[9 * CHA_SIZE];
      
      int tx = threadIdx.x;
      int ind_warp_x = tx % WARP_COL;
      int ind_channel = tx / WARP_COL;
      int ind_warp_y = ind_channel % WARP_ROW;
      ind_channel = ind_channel / WARP_ROW; 
      int ind_channel2 = tx % CHA_SIZE;
      int ind_spatial = tx / CHA_SIZE;
      int ind_col = ind_spatial % WARP_COL;
      int ind_row = ind_spatial / WARP_COL;

      int d_start = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * CHA_SIZE;
      if (tx < 9 * CHA_SIZE){
          filterback_s[tx] = 0;
      }

      //int sum_offset = ind_spatial + ind_channel2 * (WARP_COL+AUX_SIZE)*WARP_ROW*TIME_Y;
      //int sm_offset2 = ind_col + ind_row * cols + ind_channel2 * (WARP_COL+AUX_SIZE)*WARP_ROW*TIME_Y;
      for (int b = 0; b < batch; ++b){
        for (int cs = 0; cs < cols; cs+=WARP_COL-1){
          for (int rs = 0; rs < rows; rs+=WARP_ROW*TIME_Y-1){
            for (int t = 0; t < TIME_Y; ++t){
              int input_offset = (b*depth+(d_start+ind_channel))*rows*cols+(rs+ind_warp_y+WARP_ROW*t)*cols+cs+ind_warp_x;
              int sm_offset = ind_warp_x + (ind_warp_y+WARP_ROW*t) * (WARP_COL+AUX_SIZE) +ind_channel*(WARP_COL+AUX_SIZE)*WARP_ROW*TIME_Y; 
              if (rs+ind_warp_y+WARP_ROW*t < rows && cs+ind_warp_x < cols && d_start + ind_channel < depth){
                input_s[sm_offset] = input[input_offset]; 
                outback_s[sm_offset] = out_backprop[input_offset]; 
              }
              else {
                input_s[sm_offset] = 0; 
                outback_s[sm_offset] = 0; 
              } 
            }
            
            for(int f_r = 0; f_r < 3; ++f_r){
              for (int f_c = 0; f_c < 3; ++f_c){
                __syncthreads();
                for (int t = 0; t < TIME_Y; ++t){
                  int input_offset = ind_col + (ind_row + WARP_ROW*t) * (WARP_COL+AUX_SIZE) +  ind_channel2 * (WARP_COL+AUX_SIZE) * WARP_ROW*TIME_Y;
                  int sm_offset = (ind_col + f_c - 1) + (ind_row + WARP_ROW*t + f_r-1) * (WARP_COL+AUX_SIZE) +  ind_channel2 * (WARP_COL+AUX_SIZE) * WARP_ROW*TIME_Y;
                  if (ind_col+f_c-1>=0 && ind_col+f_c-1 < WARP_COL && ind_col*2+f_c-1 <= WARP_COL*2-3){
                    if (ind_row+ WARP_ROW*t+f_r-1 >=0 && ind_row+ WARP_ROW*t + f_r-1 < rows && (ind_row + WARP_ROW*t)*2+f_r-1 <= 2 * WARP_ROW*TIME_Y-3 ) {
                      temp_s[input_offset] = input_s[sm_offset] * outback_s[input_offset];  
                    }
                    else{
                      temp_s[input_offset] = 0;  
                    }
                  }
                  else{
                    temp_s[input_offset] = 0;  
                  }
                }
                int input_offset = ind_col + ind_row * (WARP_COL+AUX_SIZE) +  ind_channel2 * (WARP_COL+AUX_SIZE) * WARP_ROW*TIME_Y;
                 
                __syncthreads();
                for (int t=2; t < TIME_Y; t+=2){
                  //input_offset = ind_spatial + t * (WARP_ROW * WARP_COL) +  ind_channel2 * (WARP_COL+AUX_SIZE) * WARP_ROW*TIME_Y;
                  temp_s[input_offset] += temp_s[input_offset + t * (WARP_ROW *(WARP_COL+AUX_SIZE))];
                }
                for (unsigned int stride = WARP_ROW; stride >= 1; stride >>= 1) {
                    __syncthreads();
                    if (ind_row < stride){
                      temp_s[input_offset] += temp_s[input_offset+stride*(WARP_COL+AUX_SIZE)];
                    }
                }
                for (unsigned int stride = WARP_COL/2; stride >= 1; stride >>=1) {
                    __syncthreads();
                    if (ind_spatial < stride){
                      temp_s[input_offset] += temp_s[input_offset+stride];
                    }
                }
                __syncthreads();
                if (ind_spatial <= 0){
                  filterback_s[f_c+f_r*3+ind_channel2*9] += temp_s[input_offset];
                }

              }
            }

          }
        }
      }
      __syncthreads();
      if (tx < 9 * CHA_SIZE && d_start + tx / 9 < depth){
        filter_backprop[tx+d_start*9] = filterback_s[tx];  
      }
    }


    // A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
    //warp_size + AUX_SIZE to address bank conflict
    template <typename T, int CHA_SIZE, int WARP_SIZE, int COL_SIZE, int AUX_SIZE>
    __global__ void SeparableConv2dBackpropFilterGPUKernelNCHW_regular16x16(const int batch, const int depth, const int rows, const int cols,
      const T* input, const T* out_backprop, T* filter_backprop) {

      __shared__ float input_s[(WARP_SIZE+AUX_SIZE)*CHA_SIZE];
      __shared__ float outback_s[(WARP_SIZE+AUX_SIZE)*CHA_SIZE];
      __shared__ float temp_s[(WARP_SIZE+AUX_SIZE)*CHA_SIZE];
      __shared__ float filterback_s[9 * CHA_SIZE];
      
      int tx = threadIdx.x;
      int ind_warp = tx % (WARP_SIZE / 2);
      int ind_channel = tx / (WARP_SIZE / 2);
      int ind_channel2 = tx % CHA_SIZE;
      int ind_spatial = tx / CHA_SIZE;
      int ind_col = ind_spatial % COL_SIZE;
      int ind_row = ind_spatial / COL_SIZE;

      int d_start = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * CHA_SIZE;
      if (tx < 9 * CHA_SIZE){
          filterback_s[tx] = 0;
      }

      int sum_offset = ind_spatial + ind_channel2 * (WARP_SIZE+AUX_SIZE);
      int sm_offset = ind_warp+ind_channel*(WARP_SIZE+AUX_SIZE); 
      int sm_offset2 = ind_col + ind_row * cols + ind_channel2 * (WARP_SIZE+AUX_SIZE);
      for (int b = 0; b < batch; ++b){
        int input_offset = (b*depth+(d_start+ind_channel))*rows*cols+ind_warp;
        if (ind_warp < rows * cols && d_start + ind_channel < depth){
          input_s[sm_offset] = input[input_offset]; 
          outback_s[sm_offset] = out_backprop[input_offset]; 
        }
        else {
          input_s[sm_offset] = 0; 
          outback_s[sm_offset] = 0; 
        } 
        if (ind_warp + WARP_SIZE/2 < rows * cols && d_start + ind_channel < depth){
          input_s[sm_offset + WARP_SIZE/2] = input[input_offset + WARP_SIZE/2]; 
          outback_s[sm_offset + WARP_SIZE/2] = out_backprop[input_offset + WARP_SIZE/2]; 
        }
        else {
          input_s[sm_offset + WARP_SIZE/2] = 0; 
          outback_s[sm_offset + WARP_SIZE/2] = 0; 
        }
        //__syncthreads();
        for (int f_r = 0; f_r < 3; ++f_r){
          for (int f_c = 0; f_c < 3; ++f_c){
            __syncthreads();
            input_offset = ind_col + f_c - 1 + (ind_row + f_r - 1) * cols +  ind_channel2 * (WARP_SIZE+AUX_SIZE);
            if (ind_col+f_c-1>=0 && ind_col+f_c-1 < cols && ind_col < cols){
              if (ind_row < rows && ind_row+f_r-1 >=0 && ind_row + f_r-1 < rows) {
                temp_s[sum_offset] = input_s[input_offset] * outback_s[sm_offset2];  
              }
              else{
                temp_s[sum_offset] = 0;  
              }
              if (ind_row+WARP_SIZE/COL_SIZE/2 < rows && ind_row+WARP_SIZE/COL_SIZE/2+f_r-1 >=0 && ind_row+WARP_SIZE/COL_SIZE/2 + f_r-1 < rows){
                temp_s[sum_offset+WARP_SIZE/2] = input_s[input_offset+(WARP_SIZE/COL_SIZE/2)*cols] * outback_s[sm_offset2+(WARP_SIZE/COL_SIZE/2)*cols];  
              }
              else{
                temp_s[sum_offset+WARP_SIZE/2] = 0;  
              }
            }
            else{
                temp_s[sum_offset] = 0;  
                temp_s[sum_offset+WARP_SIZE/2] = 0;  
            }

            for (unsigned int stride = WARP_SIZE/2; stride >= 1; stride >>= 1) {
                __syncthreads();
                if (ind_spatial < stride){
                  temp_s[sum_offset] += temp_s[sum_offset+stride];
                }
            }
            __syncthreads();
            if (ind_spatial <= 0){
              filterback_s[f_c+f_r*3+ind_channel2*9] += temp_s[sum_offset];
            }
          }
        }
      }
      __syncthreads();
      if (tx < 9 * CHA_SIZE && d_start + tx / 9 < depth){
        filter_backprop[tx+d_start*9] = filterback_s[tx];  
      }
    }

    // A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
    template <typename T>
    __global__ void SeparableConv2dBackpropFilterGPUKernelNCHW2(
      const mxnet::op::sepconv::DepthwiseArgs args, const T* out_backprop, const T* input,
      T* filter_backprop, int count) {
      const int in_rows = args.in_rows;
      const int in_cols = args.in_cols;
      const int in_depth = args.in_depth;
      const int filter_rows = args.filter_rows;
      const int filter_cols = args.filter_cols;
      const int depth_multiplier = args.depth_multiplier;
      const int stride_y = args.stride_y;
      const int stride_x = args.stride_x;
      const int dilate_y = args.dilate_y;
      const int dilate_x = args.dilate_x;
      const int pad_rows = args.pad_rows;
      const int pad_cols = args.pad_cols;
      const int out_rows = args.out_rows;
      const int out_cols = args.out_cols;
      const int out_depth = args.out_depth;

      const int batch = args.batch;

      for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x * gridDim.y) {
        // Compute the indexes of this thread in the output.
        const int f_c = index % filter_cols;
        const int f_r = (index / filter_cols) % filter_rows;
        const int out_d = (index / filter_cols / filter_rows) % out_depth;

        const int in_d = out_d / depth_multiplier;
        const int dm = out_d % depth_multiplier;

        //int out_r_start = (pad_rows - f_r * dilate_y + stride_y - 1) / stride_y;
        //out_r_start = out_r_start > 0 ? out_r_start : 0;
        //int out_r_end = (in_rows - 1 + pad_rows - f_r * dilate_y) / stride_y;
        //out_r_end = out_r_end < in_rows - 1 ? out_r_end : in_rows - 1;

        //int out_c_start = (pad_cols - f_c * dilate_x + stride_x - 1) / stride_x;
        //out_c_start = out_c_start > 0 ? out_c_start : 0;
        //int out_c_end = (in_cols - 1 + pad_cols - (filter_cols - 1) * dilate_x) / stride_x;
        //out_c_end = out_c_end < in_cols - 1 ? out_c_end : in_cols - 1;

        T sum = 0;
        for (int b = 0; b < batch; ++b) {
          for (int out_r = 0; out_r < out_rows; ++out_r) {
            for (int out_c = 0; out_c < out_cols; ++out_c) {
              int in_r = out_r * stride_y - pad_rows + f_r * dilate_y;
              int in_c = out_c * stride_x - pad_cols + f_c * dilate_x;
              if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols){
                int input_offset = (b * in_depth * in_rows * in_cols) +
                  (in_d * in_rows * in_cols) +
                  (in_r * in_cols) + in_c;
                int out_backprop_offset = (b * out_depth * out_rows * out_cols) +
                  (out_d * out_rows * out_cols) +
                  (out_r * out_cols) + (out_c);
                sum += (*(input + input_offset)) * (*(out_backprop + out_backprop_offset));
              }
          //for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
          //  for (int out_c = out_c_start; out_c < out_c_end; ++out_c) {
          //    const int in_r = out_r * stride_y - pad_rows + f_r * dilate_y;
          //    const int in_c = out_c * stride_x - pad_cols + f_c * dilate_x;
          //    const int input_offset = (b * in_depth * in_rows * in_cols) +
          //      (in_d * in_rows * in_cols) +
          //      (in_r * in_cols) + in_c;
          //    const int out_backprop_offset = (b * out_depth * out_rows * out_cols) +
          //      (out_d * out_rows * out_cols) +
          //      (out_r * out_cols) + (out_c);
          //    sum += (*(input + input_offset)) * (*(out_backprop + out_backprop_offset));
            }
          }
        }
        filter_backprop[index] = sum;
      }
    }

    // A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
    // a bit slow
    template <typename T>
    __global__ void SeparableConv2dBackpropFilterGPUKernelNCHW(
      const mxnet::op::sepconv::DepthwiseArgs args, const T* out_backprop, const T* input,
      T* filter_backprop, int count) {
      const int in_rows = args.in_rows;
      const int in_cols = args.in_cols;
      const int in_depth = args.in_depth;
      const int filter_rows = args.filter_rows;
      const int filter_cols = args.filter_cols;
      const int depth_multiplier = args.depth_multiplier;
      const int stride_y = args.stride_y;
      const int stride_x = args.stride_x;
      const int dilate_y = args.dilate_y;
      const int dilate_x = args.dilate_x;
      const int pad_rows = args.pad_rows;
      const int pad_cols = args.pad_cols;
      const int out_rows = args.out_rows;
      const int out_cols = args.out_cols;
      const int out_depth = args.out_depth;

      for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x * gridDim.y) {
        // Compute the indexes of this thread in the output.
        const int out_c = index % out_cols;
        const int out_r = (index / out_cols) % out_rows;
        const int out_d = (index / out_cols / out_rows) % out_depth;

        const int b = index / out_depth / out_cols / out_rows;
        // Compute the input depth and the index of depth multiplier.
        const int in_d = out_d / depth_multiplier;
        const int dm = out_d % depth_multiplier;

        // Decide if all input is valid, if yes, we can skip the boundary checks
        // for each input.
        const int in_r_start = out_r * stride_y - pad_rows;
        const int in_c_start = out_c * stride_x - pad_cols;
        const int in_r_end = in_r_start + (filter_rows - 1) * dilate_y;
        const int in_c_end = in_c_start + (filter_cols - 1) * dilate_x;

        const int out_backprop_offset = (b * out_depth * out_rows * out_cols) +
          (out_d * out_rows * out_cols) +
          (out_r * out_cols) + (out_c);

        const int filter_offset_temp = out_d * filter_rows * filter_cols;

        const T out_bp = *(out_backprop + out_backprop_offset);
        if (in_r_start >= 0 && in_c_start >= 0 && in_r_end < in_rows &&
          in_c_end < in_cols) {
          for (int f_r = 0; f_r < filter_rows; ++f_r) {
            const int in_r = in_r_start + f_r * dilate_y;
            // Avoid repeated computation.
            const int input_offset_temp = (b * in_depth * in_rows * in_cols) +
              (in_d * in_rows * in_cols) +
              (in_r * in_cols);

            for (int f_c = 0; f_c < filter_cols; ++f_c) {
              const int in_c = in_c_start + f_c * dilate_x;
              const int input_offset = input_offset_temp + in_c;
              T partial_sum = (*(input + input_offset)) * out_bp;
              const int filter_offset = filter_offset_temp + filter_cols * f_r + f_c;
              T* addr = filter_backprop + filter_offset;

              //T* addr = filter_backprop +
              //  (dm + depth_multiplier *
              //  (in_d + in_depth * (f_c + filter_cols * f_r)));
              atomicAdd(addr, partial_sum);
            }
          }
        }
        else {
          for (int f_r = 0; f_r < filter_rows; ++f_r) {
            const int in_r = in_r_start + f_r;
            // Avoid repeated computation.
            const int input_offset_temp = (b * in_depth * in_rows * in_cols) +
              (in_d * in_rows * in_cols) +
              (in_r * in_cols);
            for (int f_c = 0; f_c < filter_cols; ++f_c) {
              const int in_c = in_c_start + f_c;
              const int addr_temp = filter_cols * f_r;

              if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
                const int input_offset = input_offset_temp + in_c;
                T partial_sum = (*(input + input_offset)) * out_bp;

                const int filter_offset = filter_offset_temp + filter_cols * f_r + f_c;
                T* addr = filter_backprop + filter_offset;

                //T* addr =
                //  filter_backprop +
                //  (dm + depth_multiplier * (in_d + in_depth * (f_c + addr_temp)));

                // Potentially many threads can add to the same address so we have
                // to use atomic add here.
                // TODO(jmchen): If atomic add turns out to be slow, we can:
                // 1. allocate multiple buffers for the gradients (one for each
                // example in a batch, for example). This can reduce the
                // contention on the destination; 2. Have each thread compute one
                // gradient for an element in the filters. This should work well
                // when the input depth is big and filter size is not too small.
                atomicAdd(addr, partial_sum);
              }
            }
          }
        }
      }
    }
    

    template<typename Dtype>
    inline void SeparableConv2dForward(const Tensor<gpu, 4, Dtype> &out,
      const Tensor<gpu, 4, Dtype> &data,
      const Tensor<gpu, 3, Dtype> &wmat,
      const mxnet::op::sepconv::DepthwiseArgs args) {
      // LOG(INFO) << "PSROIPoolForward";
      const Dtype *bottom_data = data.dptr_;
      const Dtype *weight_data = wmat.dptr_;
      Dtype *top_data = out.dptr_;

      int count = out.shape_.Size();

      const int grid_size = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
      dim3 dimGrid(kMaxGridNum, (grid_size + kMaxGridNum - 1) / kMaxGridNum);
      dim3 dimBlock(kMaxThreadsPerBlock);
      CheckLaunchParam(dimGrid, dimBlock, "SeparableConvolution Forward");
      cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
      SeparableConv2dGPUKernelNCHW<Dtype> <<<dimGrid, dimBlock, 0, stream >>>(args, bottom_data, weight_data, top_data, count);
      SEPARABLE_CONV_CUDA_CHECK(cudaPeekAtLastError());
    }

    template<typename Dtype>
    inline void SeparableConv2dBackwardInput(const Tensor<gpu, 4, Dtype> &in_grad,
      const Tensor<gpu, 4, Dtype> &out_grad,
      const Tensor<gpu, 3, Dtype> &wmat,
      const mxnet::op::sepconv::DepthwiseArgs args) {
      // LOG(INFO) << "PSROIPoolForward";
      const Dtype *top_diff = out_grad.dptr_;
      Dtype *bottom_diff = in_grad.dptr_;
      const Dtype *weight_data = wmat.dptr_;

      int count = in_grad.shape_.Size();

      const int grid_size = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
      dim3 dimGrid(kMaxGridNum, (grid_size + kMaxGridNum - 1) / kMaxGridNum);
      dim3 dimBlock(kMaxThreadsPerBlock);
      CheckLaunchParam(dimGrid, dimBlock, "SeparableConvolution Forward");
      cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
      SeparableConv2dBackpropInputGPUKernelNCHW<Dtype> <<<dimGrid, dimBlock, 0, stream >>>(args, top_diff, weight_data, bottom_diff, count);
      SEPARABLE_CONV_CUDA_CHECK(cudaPeekAtLastError());
    }

    template<typename Dtype>
    inline void SeparableConv2dBackwardFilter(const Tensor<gpu, 4, Dtype> &out_grad,
      const Tensor<gpu, 4, Dtype> &input,
      const Tensor<gpu, 3, Dtype> &wmat_diff,
      const mxnet::op::sepconv::DepthwiseArgs args) {
      // LOG(INFO) << "PSROIPoolForward";
      const Dtype *top_diff = out_grad.dptr_;
      const Dtype *bottom_data = input.dptr_;
      Dtype *weight_diff = wmat_diff.dptr_;

      //int count = out_grad.shape_.Size();
      int count = wmat_diff.shape_.Size();

      const int grid_size = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
      dim3 dimGrid(kMaxGridNum, (grid_size + kMaxGridNum - 1) / kMaxGridNum);
      dim3 dimBlock(kMaxThreadsPerBlock);
      CheckLaunchParam(dimGrid, dimBlock, "SeparableConvolution Forward");
      cudaStream_t stream = Stream<gpu>::GetStream(wmat_diff.stream_);
      //SeparableConv2dBackpropFilterGPUKernelNCHW<Dtype> <<<dimGrid, dimBlock, 0, stream >>>(args, top_diff, bottom_data, weight_diff, count);
      SeparableConv2dBackpropFilterGPUKernelNCHW2<Dtype> << <dimGrid, dimBlock, 0, stream >> >(args, top_diff, bottom_data, weight_diff, count);
      SEPARABLE_CONV_CUDA_CHECK(cudaPeekAtLastError());
    }

    template<typename Dtype>
    inline void SeparableConv2dBackwardFilter_share(const Tensor<gpu, 4, Dtype> &out_grad,
      const Tensor<gpu, 4, Dtype> &input,
      const Tensor<gpu, 3, Dtype> &wmat_diff,
      const mxnet::op::sepconv::DepthwiseArgs args) {
      // LOG(INFO) << "PSROIPoolForward";
      const Dtype *top_diff = out_grad.dptr_;
      const Dtype *bottom_data = input.dptr_;
      Dtype *weight_diff = wmat_diff.dptr_;

      //int count = out_grad.shape_.Size();
      if (args.pad_rows==1 && args.dilate_x==1 && args.filter_rows==3 && args.stride_x == 1 && kMaxThreadsPerBlock == 1024){
        if (args.in_rows <= 8 && args.in_cols <=8){
          int WARP_SIZE=64;
          int count = args.in_depth * WARP_SIZE / 2;
          const int grid_size = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
          dim3 dimGrid(grid_size);
          dim3 dimBlock(kMaxThreadsPerBlock);
          CheckLaunchParam(dimGrid, dimBlock, "SeparableConvolution Forward");
          cudaStream_t stream = Stream<gpu>::GetStream(wmat_diff.stream_);
          SeparableConv2dBackpropFilterGPUKernelNCHW_regular16x16<Dtype, 32, 64, 8, 1> << <dimGrid, dimBlock, 0, stream >> >(args.batch, args.in_depth, args.in_rows, args.in_cols, bottom_data, top_diff, weight_diff);
          SEPARABLE_CONV_CUDA_CHECK(cudaPeekAtLastError());
        }
        else if (args.in_rows <= 16 && args.in_cols <=16) {
          int WARP_SIZE=256;
          int count = args.in_depth * WARP_SIZE / 2;
          const int grid_size = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
          dim3 dimGrid(grid_size);
          dim3 dimBlock(kMaxThreadsPerBlock);
          CheckLaunchParam(dimGrid, dimBlock, "SeparableConvolution Forward");
          cudaStream_t stream = Stream<gpu>::GetStream(wmat_diff.stream_);
          SeparableConv2dBackpropFilterGPUKernelNCHW_regular16x16<Dtype, 8, 256, 16, 4> << <dimGrid, dimBlock, 0, stream >> >(args.batch, args.in_depth, args.in_rows, args.in_cols, bottom_data, top_diff, weight_diff);
          SEPARABLE_CONV_CUDA_CHECK(cudaPeekAtLastError());
        }
        else {
          int WARP_COL=32;
          int WARP_ROW=4;
          int TIME_Y = 2;
          int count = args.in_depth * WARP_COL * WARP_ROW;
          const int grid_size = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
          dim3 dimGrid(grid_size);
          dim3 dimBlock(kMaxThreadsPerBlock);
          CheckLaunchParam(dimGrid, dimBlock, "SeparableConvolution Forward");
          cudaStream_t stream = Stream<gpu>::GetStream(wmat_diff.stream_);
          SeparableConv2dBackpropFilterGPUKernelNCHW_regular64x64<Dtype, 8, 4, 32, 2, 1> << <dimGrid, dimBlock, 0, stream >> >(args.batch, args.in_depth, args.in_rows, args.in_cols, bottom_data, top_diff, weight_diff);
          SEPARABLE_CONV_CUDA_CHECK(cudaPeekAtLastError());
        }
      }
      else {
        int count = wmat_diff.shape_.Size();

        const int grid_size = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
        dim3 dimGrid(kMaxGridNum, (grid_size + kMaxGridNum - 1) / kMaxGridNum);
        dim3 dimBlock(kMaxThreadsPerBlock);
        CheckLaunchParam(dimGrid, dimBlock, "SeparableConvolution Forward");
        cudaStream_t stream = Stream<gpu>::GetStream(wmat_diff.stream_);
        SeparableConv2dBackpropFilterGPUKernelNCHW2<Dtype> << <dimGrid, dimBlock, 0, stream >> >(args, top_diff, bottom_data, weight_diff, count);
        SEPARABLE_CONV_CUDA_CHECK(cudaPeekAtLastError());
      }
    }

  }  // namespace


  template<typename Dtype>
  inline void SeparableConv2dForward(const Tensor<gpu, 4, Dtype> &out,
    const Tensor<gpu, 4, Dtype> &data,
    const Tensor<gpu, 3, Dtype> &wmat,
    const mxnet::op::sepconv::DepthwiseArgs args) {
    cuda::SeparableConv2dForward(out, data, wmat, args);
  }

  template<typename Dtype>
  inline void SeparableConv2dBackwardInput(const Tensor<gpu, 4, Dtype> &in_grad,
    const Tensor<gpu, 4, Dtype> &out_grad,
    const Tensor<gpu, 3, Dtype> &wmat,
    const mxnet::op::sepconv::DepthwiseArgs args) {
    cuda::SeparableConv2dBackwardInput(in_grad, out_grad, wmat, args);
  }

  template<typename Dtype>
  inline void SeparableConv2dBackwardFilter(const Tensor<gpu, 4, Dtype> &out_grad,
    const Tensor<gpu, 4, Dtype> &input,
    const Tensor<gpu, 3, Dtype> &wmat_diff,
    const mxnet::op::sepconv::DepthwiseArgs args) {

    cuda::SeparableConv2dBackwardFilter(out_grad, input, wmat_diff, args);
  }

  template<typename Dtype>
  inline void SeparableConv2dBackwardFilter_share(const Tensor<gpu, 4, Dtype> &out_grad,
    const Tensor<gpu, 4, Dtype> &input,
    const Tensor<gpu, 3, Dtype> &wmat_diff,
    const mxnet::op::sepconv::DepthwiseArgs args) {

    cuda::SeparableConv2dBackwardFilter_share(out_grad, input, wmat_diff, args);
  }

}  // namespace mshadow


namespace mxnet {
  namespace op {

    template<>
    Operator* CreateOp<gpu>(SeparableConvolutionParam param, int dtype,
      std::vector<TShape> *in_shape,
      std::vector<TShape> *out_shape,
      Context ctx) {
      Operator *op = NULL;

      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new SeparableConvolutionOp<gpu, DType>(param);
      })

        return op;
    }

  }  // namespace op
}  // namespace mxnet
