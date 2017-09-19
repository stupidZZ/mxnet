/*!
 * Copyright (c) 2017 by Contributors
 * \file batch_norm-inl.h
 * \brief
 * \author Bing Xu, Chris Olivier
 */
#ifndef MXNET_OPERATOR_SYNC_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_SYNC_BATCH_NORM_INL_H_

#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <sys/shm.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unistd.h>
#include <errno.h>
#include <typeinfo>

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

namespace {
template <typename xpu, typename DType, typename AccReal>
class Sync_Batch_Thread_Comm {
    private:
        static std::map<int, void*> instance_set;
        
        // reduce related
        std::mutex reduce_mutex;
        static std::mutex instance_set_mutex;
        std::vector<AccReal> buffer;
        std::atomic_bool buffer_init;
        std::atomic_int sum_inst_cnt;

        // barrier related
        std::mutex barrier_mutex;
        std::condition_variable barrier_cv;
        std::atomic_int barrier_count, barrier_generation, barrier_max_thread_num;
        
        Sync_Batch_Thread_Comm(const Sync_Batch_Thread_Comm&) {};
        Sync_Batch_Thread_Comm& operator=(const Sync_Batch_Thread_Comm&){};
        
    protected:
        Sync_Batch_Thread_Comm() {
            barrier_count = 0;
            barrier_generation = 0;
            barrier_max_thread_num = 0;
            buffer_init = false;
            sum_inst_cnt = 0;
        }
        
    public:
        void reduce(const std::vector<AccReal> & input_array, std::vector<AccReal> & output_array, const int inst_cnt, 
                    int & output_sum_inst_cnt, const std::thread::id debug_id) {
            if(input_array.size() != output_array.size() && inst_cnt > 0) {
                exit(-1);
            }
            
            reduce_mutex.lock();
            if(buffer.size() < input_array.size()) 
            {
                buffer.resize(input_array.size());
            }
            if(buffer_init == false) {
                sum_inst_cnt = 0;
                memset(&buffer[0], 0, sizeof(AccReal) * buffer.size());
                buffer_init = true;
            }
            
            sum_inst_cnt += inst_cnt;
            for(int i = 0;i < input_array.size(); i++) {
                buffer[i] += input_array[i];
            }
            reduce_mutex.unlock();
            int barrier_max_thread_num_t = barrier_max_thread_num;
            //printf("id:%lld barrier_max_thread_num:%d Good\n", debug_id, barrier_max_thread_num_t);
            barrier();
            //printf("id:%lld barrier_max_thread_num:%d Done\n", debug_id, barrier_max_thread_num_t);
            //for(int i = 0;i < buffer.size(); i++) {
            //    printf("id:%lld, i:%d, buffer_addr:%d, buffer:%.6f\n", debug_id, i, buffer, buffer[i]);
            //}
            
            buffer_init = false;
            memcpy(&output_array[0], &buffer[0], sizeof(AccReal) * input_array.size());
            
            for(int i = 0;i < buffer.size(); i++) {
                output_array[i] = buffer[i];
            }
            
            output_sum_inst_cnt = sum_inst_cnt;
            barrier();
        }
        
        void barrier() {
            std::unique_lock<std::mutex> ulock(barrier_mutex);
            int gen = barrier_generation;
            //printf("thread_id:%lld, comm_key:%lld, barrier_count:%d\n", std::this_thread::get_id(), comm_key, barrier_count+1);
            if(++barrier_count == barrier_max_thread_num) {
                barrier_generation++;
                barrier_count = 0;
                barrier_cv.notify_all();
                return;
            }
            
            while(gen == barrier_generation) {
                barrier_cv.wait(ulock);
            }
        }
        
        void set_barrier_max_thread_num(int n) {
            barrier_max_thread_num = n;
            //printf("barrier_max_thread_num:%d\n", n);
        }
        
        int get_barrier_max_thread_num() {
            return barrier_max_thread_num;
        }
        static Sync_Batch_Thread_Comm* getInstance(int comm_key) {
            instance_set_mutex.lock();
            void *tmp = NULL;
            if(instance_set[comm_key] == NULL) {
                tmp = new Sync_Batch_Thread_Comm();
                instance_set[comm_key] = tmp;
                ((Sync_Batch_Thread_Comm*) tmp)->comm_key = comm_key;
            }
            else {   
                tmp = instance_set[comm_key];
            }
            instance_set_mutex.unlock();
            return (Sync_Batch_Thread_Comm*)tmp;
        }
        
        int comm_key;
};
template <typename xpu, typename DType, typename AccReal>
std::map<int, void*> Sync_Batch_Thread_Comm<xpu, DType, AccReal>::instance_set = std::map<int, void*>();

template <typename xpu, typename DType, typename AccReal>
std::mutex Sync_Batch_Thread_Comm<xpu, DType, AccReal>::instance_set_mutex;

//template <typename xpu, typename DType, typename AccReal>
//Sync_Batch_Thread_Comm<xpu, DType, AccReal>* Sync_Batch_Thread_Comm<xpu, DType, AccReal>::instance = new Sync_Batch_Thread_Comm<xpu, DType, AccReal>();
}

namespace mxnet {
namespace op {

namespace sync_batchnorm {
enum SyncBatchNormOpInputs {kData, kGamma, kBeta, kPrevOp};  // kGamma: weights, kBeta: biases
enum SyncBatchNormOpOutputs {kOut, kMean, kVar};  // req, out_data
enum SyncBatchNormOpAuxiliary {kMovingMean, kMovingVar};  // aux_states
}  // namespace sync_batchnorm

/*! \brief Parameters for BatchNoram operator */
struct SyncBatchNormParam : public dmlc::Parameter<SyncBatchNormParam> {
  float eps;
  float momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  int ctx_number;
  int comm_key;
  int dev_id;
  DMLC_DECLARE_PARAMETER(SyncBatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0. "
              "Must be bigger than CUDNN_BN_MIN_EPSILON "
              "defined in cudnn.h when using cudnn (usually 1e-5)");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output All,normal mean and var");
    DMLC_DECLARE_FIELD(ctx_number).set_default(0)
      .describe("ctx_number");     
    DMLC_DECLARE_FIELD(comm_key).set_default(0)
      .describe("comm_key");     
  }
  
  void Copy(const SyncBatchNormParam& param) {
    eps = param.eps;
    momentum = param.momentum;
    use_global_stats = param.use_global_stats;
    output_mean_var = param.output_mean_var;
    ctx_number = param.ctx_number;
    comm_key = param.comm_key;
    fix_gamma = param.fix_gamma;
  }
};

/*! \brief Batch normalization operator */
template <typename xpu, typename DType, typename AccReal>
class SyncBatchNormOp : public Operator {
 public:
  /*! \brief Batch normalization operator parameters */
  SyncBatchNormParam param_;
  Sync_Batch_Thread_Comm<gpu, DType, AccReal> * comm_instance;
  explicit SyncBatchNormOp(SyncBatchNormParam param) {
    this->param_ = param;
    comm_instance = Sync_Batch_Thread_Comm<gpu, DType, AccReal>::getInstance(this->param_.comm_key);
    comm_instance->set_barrier_max_thread_num(this->param_.ctx_number);
  }

  static inline bool IsWriting(const OpReqType ort) {
    return ort == kWriteTo || ort == kWriteInplace;
  }

  /*!
   * \brief perform a forward operation of Operator, save the output to TBlob.
   * \param ctx runtime context available to this call
   * \param in_data array of input data, it is const
   * \param req the request types of saving operation, can only be kWriteTo or kWriteInplace.
   * \param out_data array of output data, pointer is used to indicate that this is holder
   *        the space of TBlob in out_data must be pre-allocated with InferShape
   * \param aux_states Auxiliary states of operator. Normally operator doesn't
   *        need, epecial case like Batch Norm requires.
   * \sa OpReqType, OpContext
   */
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(in_data.size(), 4U);
    CHECK_EQ(aux_states.size(), 2U);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3U);
      CHECK_EQ(req.size(), 3U);
    } else {
      CHECK_GE(out_data.size(), 1U);
      CHECK_GE(req.size(), 1U);
      CHECK_EQ(req[sync_batchnorm::kOut], kWriteTo);
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    //printf("thread_num:%lld, Forward\n", std::this_thread::get_id());

    DoForward(s, ctx, in_data, req, out_data, aux_states);
  }

  /*!
   * \brief Perform a Backward Operation, write gradient to the in_grad.
   *
   * \note
   * Convention:
   *   out_grad.size() == OperatorProperty.NumVisibleOutputs()
   *   out_data.size() == OperatorProperty.NumOutputs()
   * out_data can contain additional invisible returns that remembers the
   * state carried from the Forward pass. For example mask in the dropout.
   * The gradients are passed from visible returns in this function.
   *
   * \par
   * Not all the TBlobs in the arguments will be available
   * if you override the DeclareBackwardDependency of corresponding OperatorProperty class.
   * Only the dependencies you declared will be available at corresponding position,
   * the rest of the parameters are simply dummy where you will get a nullptr.
   * You will be safe if you use the default DeclareBackwardDependency.
   * But only declare what you need will give engine more chance for optimization.
   *
   * \param ctx runtime context available to this call
   * \param out_grad the gradient value we get from of the Operator.
   * \param in_data the array of input data.
   * \param out_data the array of output data.
   * \param req request types of the saving operation, can be all types.
   * \param in_grad the array of gradient we need to write to.
   * \param aux_states Auxiliary states of operator. Normally operator doesn't need
   * \sa OperatorProperty, OpReqType, OpContext
   */
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    CHECK_EQ(out_grad.size(), param_.output_mean_var ? 3U : 1U);
    CHECK_EQ(in_data.size(), 4U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK_EQ(in_grad.size(), 4U);
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    //printf("thread_num:%lld, Backward\n", std::this_thread::get_id());
    DoBackward(s, ctx, out_grad, in_data,
               out_data, req, in_grad, aux_states);
  }

 private:
  void DoForward(mshadow::Stream<cpu> *stream,
                 const OpContext &ctx,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data,
                 const std::vector<TBlob> &aux_states);

  void DoBackward(mshadow::Stream<cpu> *stream,
                  const OpContext &ctx,
                  const std::vector<TBlob> &out_grad,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &in_grad,
                  const std::vector<TBlob> &aux_states);

#if MXNET_USE_CUDA
  void DoForward(mshadow::Stream<gpu> *stream,
                 const OpContext &ctx,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data,
                 const std::vector<TBlob> &aux_states);
  void DoBackward(mshadow::Stream<gpu> *stream,
                  const OpContext &ctx,
                  const std::vector<TBlob> &out_grad,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &in_grad,
                  const std::vector<TBlob> &aux_states);
#endif  // MXNET_USE_CUDA
};  // class SyncBatchNormOp

template<typename xpu>
Operator *CreateOp(const SyncBatchNormParam& param, const int dtype, const TShape& shape);

#if DMLC_USE_CXX11
class SyncBatchNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 4U) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);

    if (dshape.ndim() == 0) {
      return false;
    }

    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));

    out_shape->clear();
    out_shape->push_back(dshape);             // kOut
    out_shape->push_back(Shape1(dshape[1]));  // kMean
    out_shape->push_back(Shape1(dshape[1]));  // kVar

    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));  // kMovingMean
    aux_shape->push_back(Shape1(dshape[1]));  // kMovingVar
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    using namespace mshadow;
    CHECK_GE(in_type->size(), 1U);
    const int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    // NOTE: This requirement is from cuDNN (v. 4 and 5)
    int dtype_param;
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
         dtype_param = mshadow::DataType<AccRealX>::kFlag; });
    for (index_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        CHECK_EQ((*in_type)[i], dtype_param) << "This layer requires uniform type. "
                                             << "Expected " << dtype_param << " v.s. given "
                                             << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    for (index_t i = 0; i < aux_type->size(); ++i) {
      if ((*aux_type)[i] != -1) {
        CHECK_EQ((*aux_type)[i], dtype_param) << "This layer requires uniform type. "
                                              << "Expected " << dtype_param << " v.s. given "
                                              << (*aux_type)[i] << " at " << ListArguments()[i];
      }
    }
    const size_t n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (size_t i = 0; i < n_aux; ++i) {
      aux_type->push_back(dtype_param);
    }
    const size_t n_out = this->ListOutputs().size();
    out_type->clear();
    out_type->push_back(dtype);
    for (size_t i = 1; i < n_out; ++i) {
      out_type->push_back(dtype_param);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SyncBatchNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SyncBatchNorm";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[sync_batchnorm::kOut],
            out_data[sync_batchnorm::kMean],
            out_data[sync_batchnorm::kVar],
            in_data[sync_batchnorm::kData],
            in_data[sync_batchnorm::kGamma]
           };
  }

  int NumVisibleOutputs() const override {
    if (param_.output_mean_var) {
      return 3;
    }
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta", "prev_op"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_var"};
  }

  Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const override;

  inline const SyncBatchNormParam& getParam() const {
    return param_;
  }

 private:
  SyncBatchNormParam param_;
};  // class SyncBatchNormProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif

#endif  // MXNET_OPERATOR_SYNC_BATCH_NORM_INL_H_

