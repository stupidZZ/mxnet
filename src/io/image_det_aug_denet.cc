/*!
 *  Copyright (c) 2015 by Contributors
 * \file image_det_aug_denet.cc
 * \brief Default augmenter.
 */
#include <mxnet/base.h>
#include <utility>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>
#include <exception>  
#include <iostream>
#include "./image_augmenter.h"
#include "../common/utils.h"


#define DEBUG_FLAG 0

namespace mxnet {
namespace io {

using nnvm::Tuple;

namespace image_det_aug_denet_enum {
enum ImageDetAugDeNetCropEmitMode {kCenter, kOverlap, kIou};
enum ImageDetAugDeNetResizeMode {kForce, kShrink, kFit, kForceFit};
}

/*! \brief image detection augmentation parameters*/
struct DenetImageDetAugmentParam : public dmlc::Parameter<DenetImageDetAugmentParam> {
  /*! \brief resize shorter edge to size before applying other augmentations */
  int resize;
  /*! \brief probability we do random cropping, use prob <= 0 to disable */
  float rand_crop_prob;
  /*! \brief min crop scales */
  Tuple<float> min_crop_scales;
  /*! \brief max crop scales */
  Tuple<float> max_crop_scales;
  /*! \brief min crop aspect ratios */
  Tuple<float> min_crop_aspect_ratios;
  /*! \brief max crop aspect ratios */
  Tuple<float> max_crop_aspect_ratios;
  /*! \brief min IOUs between ground-truths and crop boxes */
  Tuple<float> min_crop_overlaps;
  /*! \brief max IOUs between ground-truths and crop boxes */
  Tuple<float> max_crop_overlaps;
  /*! \brief min itersection/gt_area between ground-truths and crop boxes */
  Tuple<float> min_crop_sample_coverages;
  /*! \brief max itersection/gt_area between ground-truths and crop boxes */
  Tuple<float> max_crop_sample_coverages;
  /*! \brief min itersection/crop_area between ground-truths and crop boxes */
  Tuple<float> min_crop_object_coverages;
  /*! \brief max itersection/crop_area between ground-truths and crop boxes */
  Tuple<float> max_crop_object_coverages;
  /*! \brief number of crop samplers, skip random crop if <= 0 */
  int num_crop_sampler;
  /*! \beief 0-emit ground-truth if center out of crop area
   * 1-emit if overlap < emit_overlap_thresh
   */
  int crop_emit_mode;
  /*! \brief ground-truth emition threshold specific for crop_emit_mode == 1 */
  float emit_overlap_thresh;
  /*! \brief maximum trials for cropping, skip cropping if fails exceed this number */
  Tuple<int> max_crop_trials;
  /*! \brief random padding prob */
  float rand_pad_prob;
  /*!< \brief maximum padding scale */
  float max_pad_scale;
  /*! \brief max random in H channel */
  int max_random_hue;
  /*! \brief random H prob */
  float random_hue_prob;
  /*! \brief max random in S channel */
  int max_random_saturation;
  /*! \brief random saturation prob */
  float random_saturation_prob;
  /*! \brief max random in L channel */
  int max_random_illumination;
  /*! \brief random illumination change prob */
  float random_illumination_prob;
  /*! \brief max random contrast */
  float max_random_contrast;
  /*! \brief random contrast prob */
  float random_contrast_prob;
  /*! \brief random mirror prob */
  float rand_mirror_prob;
  /*! \brief filled color while padding */
  int fill_value;
  /*! \brief interpolation method 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand  */
  int inter_method;
  /*! \brief shape of the image data */
  TShape data_shape;
  /*! \brief resize mode, 0-force
   * 1-Shrink to data_shape, preserve ratio,
   * 2-fit to data_shape, preserve ratio
   */
  int resize_mode;
  // declare parameters
  DMLC_DECLARE_PARAMETER(DenetImageDetAugmentParam) {
    DMLC_DECLARE_FIELD(resize).set_default(-1)
        .describe("Augmentation Param: scale shorter edge to size "
                  "before applying other augmentations, -1 to disable.");
    DMLC_DECLARE_FIELD(rand_crop_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability of random cropping, <= 0 to disable");
    DMLC_DECLARE_FIELD(min_crop_scales).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: Min crop scales.");
    DMLC_DECLARE_FIELD(max_crop_scales).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Max crop scales.");
    DMLC_DECLARE_FIELD(min_crop_aspect_ratios).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Min crop aspect ratios.");
    DMLC_DECLARE_FIELD(max_crop_aspect_ratios).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Max crop aspect ratios.");
    DMLC_DECLARE_FIELD(min_crop_overlaps).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: Minimum crop IOU between crop_box and ground-truths.");
    DMLC_DECLARE_FIELD(max_crop_overlaps).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Maximum crop IOU between crop_box and ground-truth.");
    DMLC_DECLARE_FIELD(min_crop_sample_coverages).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: Minimum ratio of intersect/crop_area "
                  "between crop box and ground-truths.");
    DMLC_DECLARE_FIELD(max_crop_sample_coverages).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Maximum ratio of intersect/crop_area "
                  "between crop box and ground-truths.");
    DMLC_DECLARE_FIELD(min_crop_object_coverages).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: Minimum ratio of intersect/gt_area "
                  "between crop box and ground-truths.");
    DMLC_DECLARE_FIELD(max_crop_object_coverages).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Maximum ratio of intersect/gt_area "
                  "between crop box and ground-truths.");
    DMLC_DECLARE_FIELD(num_crop_sampler).set_default(1)
        .describe("Augmentation Param: Number of crop samplers.");
    DMLC_DECLARE_FIELD(crop_emit_mode)
        .add_enum("center", image_det_aug_denet_enum::kCenter)
        .add_enum("overlap", image_det_aug_denet_enum::kOverlap)
        .add_enum("iou", image_det_aug_denet_enum::kIou)
        .set_default(image_det_aug_denet_enum::kCenter)
        .describe("Augmentation Param: Emition mode for invalid ground-truths after crop. "
                  "center: emit if centroid of object is out of crop region; "
                  "overlap: emit if overlap is less than emit_overlap_thresh. ");
    DMLC_DECLARE_FIELD(emit_overlap_thresh).set_default(0.3f)
        .describe("Augmentation Param: Emit overlap thresh for emit mode overlap only.");
    DMLC_DECLARE_FIELD(max_crop_trials).set_default(Tuple<int>({25}))
        .describe("Augmentation Param: Skip cropping if fail crop trail count "
                  "exceeds this number.");
    DMLC_DECLARE_FIELD(rand_pad_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability for random padding.");
    DMLC_DECLARE_FIELD(max_pad_scale).set_default(1.0f)
        .describe("Augmentation Param: Maximum padding scale.");
    DMLC_DECLARE_FIELD(max_random_hue).set_default(0)
        .describe("Augmentation Param: Maximum random value of H channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_hue_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability to apply random hue.");
    DMLC_DECLARE_FIELD(max_random_saturation).set_default(0)
        .describe("Augmentation Param: Maximum random value of S channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_saturation_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability to apply random saturation.");
    DMLC_DECLARE_FIELD(max_random_illumination).set_default(0)
        .describe("Augmentation Param: Maximum random value of L channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_illumination_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability to apply random illumination.");
    DMLC_DECLARE_FIELD(max_random_contrast).set_default(0)
        .describe("Augmentation Param: Maximum random value of delta contrast.");
    DMLC_DECLARE_FIELD(random_contrast_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability to apply random contrast.");
    DMLC_DECLARE_FIELD(rand_mirror_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability to apply horizontal flip aka. mirror.");
    DMLC_DECLARE_FIELD(fill_value).set_default(127)
        .describe("Augmentation Param: Filled color value while padding.");
    DMLC_DECLARE_FIELD(inter_method).set_default(1)
        .describe("Augmentation Param: 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand.");
    DMLC_DECLARE_FIELD(data_shape)
        .set_expect_ndim(3).enforce_nonzero()
        .describe("Dataset Param: Shape of each instance generated by the DataIter.");
    DMLC_DECLARE_FIELD(resize_mode)
      .add_enum("force", image_det_aug_denet_enum::kForce)
      .add_enum("shrink", image_det_aug_denet_enum::kShrink)
      .add_enum("fit", image_det_aug_denet_enum::kFit)
      .add_enum("forcefit", image_det_aug_denet_enum::kForceFit)
      .set_default(image_det_aug_denet_enum::kForce)
      .describe("Augmentation Param: How image data fit in data_shape. "
                "force: force reshape to data_shape regardless of aspect ratio; "
                "shrink: ensure each side fit in data_shape, preserve aspect ratio; "
                "fit: fit image to data_shape, preserve ratio, will upscale if applicable."
                "forcefit: fit image to data_shape, preserve ratio, will fill border with fill_value if the shape is smaller than data_shape.");
  }
};

DMLC_REGISTER_PARAMETER(DenetImageDetAugmentParam);

std::vector<dmlc::ParamFieldInfo> ListDenetDetAugParams() {
  return DenetImageDetAugmentParam::__FIELDS__();
}

#if MXNET_USE_OPENCV
using Rect = cv::Rect_<float>;

#ifdef _MSC_VER
#define M_PI CV_PI
#endif

/*! \brief helper class for better detection label handling */
class ImageDetLabel {
 public:
  /*! \brief Helper struct to store the coordinates and id for each object */
  struct ImageDetObject {
    float id;
    float left;
    float top;
    float right;
    float bottom;
    std::vector<float> extra;  // store extra info other than id and coordinates

    /*! \brief Return converted Rect object */
    Rect ToRect() const {
      return Rect(left, top, right - left, bottom - top);
    }

     /*! \brief Return projected coordinates according to new region */
     ImageDetObject Project(Rect box) const {
       ImageDetObject ret = *this;
       ret.left = std::max(0.f, (ret.left - box.x) / box.width);
       ret.top = std::max(0.f, (ret.top - box.y) / box.height);
       ret.right = std::min(1.f, (ret.right - box.x) / box.width);
       ret.bottom = std::min(1.f, (ret.bottom - box.y) / box.height);
       return ret;
     }

     /*! \brief Return Horizontally fliped coordinates */
     ImageDetObject HorizontalFlip() const {
       ImageDetObject ret = *this;
       ret.left = 1.f - this->right;
       ret.right = 1.f - this->left;
       return ret;
     }
  };  // struct ImageDetObject

  /*! \brief constructor from raw array of detection labels */
  explicit ImageDetLabel(const std::vector<float> &raw_label) {
      try {
        FromArray(raw_label);
      }
      catch (...) {
        printf("raw_label.size():%d\n", raw_label.size());
        throw;
      }
  }

  /*! \brief construct from raw array with following format
   * header_width, object_width, (extra_headers...),
   * [id, xmin, ymin, xmax, ymax, (extra_object_info)] x N
   */
  void FromArray(const std::vector<float> &raw_label) {
    int label_width = static_cast<int>(raw_label.size());
    CHECK_GE(label_width, 7);  // at least 2(header) + 5(1 object)
    int header_width = static_cast<int>(raw_label[0]);
    CHECK_GE(header_width, 2);
    object_width_ = static_cast<int>(raw_label[1]);
    CHECK_GE(object_width_, 5);  // id, x1, y1, x2, y2...
    header_.assign(raw_label.begin(), raw_label.begin() + header_width);
    int num = (label_width - header_width) / object_width_;
    CHECK_EQ((label_width - header_width) % object_width_, 0);
    objects_.reserve(num);
    for (int i = header_width; i < label_width; i += object_width_) {
      ImageDetObject obj;
      auto it = raw_label.cbegin() + i;
      obj.id = *(it++);
      obj.left = *(it++);
      obj.top = *(it++);
      obj.right = *(it++);
      obj.bottom = *(it++);
      obj.extra.assign(it, it - 5 + object_width_);
      objects_.push_back(obj);
      try {
          CHECK_GT(obj.right, obj.left);
          CHECK_GT(obj.bottom, obj.top);
      }
      catch (...) {
        printf("obj box:(i:%d object_width:%d header_width:%d %f, %f, %f, %f)\n", i, object_width_, header_width, obj.right, obj.left, obj.bottom, obj.top);
        throw;
      }
    }
  }

  /*! \brief Convert back to raw array */
  std::vector<float> ToArray() const {
    std::vector<float> out(header_);
    out.reserve(out.size() + objects_.size() * object_width_);
    for (auto& obj : objects_) {
      out.push_back(obj.id);
      out.push_back(obj.left);
      out.push_back(obj.top);
      out.push_back(obj.right);
      out.push_back(obj.bottom);
      out.insert(out.end(), obj.extra.begin(), obj.extra.end());
    }
    return out;
  }

  /*! \brief Intersection over Union between two rects */
  static float RectIOU(Rect a, Rect b) {
    float intersect = (a & b).area();
    if (intersect <= 0.f) return 0.f;
    return intersect / (a.area() + b.area() - intersect);
  }

  /*! \brief try crop image with given crop_box
   * return false if fail to meet any of the constraints
   * convert all objects if success
   */
  bool TryCrop(const Rect crop_box,
    const float min_crop_overlap, 
    const int crop_emit_mode, const float emit_overlap_thresh) {
     
    
    if(DEBUG_FLAG) {
        printf("min_crop_overlap:%f, should be 0.5\n", min_crop_overlap);
        printf("crop_emit_mode:%d, should be %d\n", crop_emit_mode, image_det_aug_denet_enum::kOverlap);
        printf("emit_overlap_thresh:%f, should be 0.5\n", emit_overlap_thresh); 
    }
    
    if (objects_.size() < 1) {
      return true;  // no object, raise error or just skip?
    }
    
    Rect unit_rect(0, 0, 1, 1);
    float ovp = RectIOU(crop_box, unit_rect);
    bool valid = ovp > min_crop_overlap;

    if (!valid) return false;
    // transform ground-truth labels
    std::vector<ImageDetObject> new_objects;
    for (auto iter = objects_.begin(); iter != objects_.end(); ++iter) {
      if (image_det_aug_denet_enum::kCenter == crop_emit_mode) {
        float center_x = (iter->left + iter->right) * 0.5f;
        float center_y = (iter->top + iter->bottom) * 0.5f;
        if (!crop_box.contains(cv::Point2f(center_x, center_y))) {
          continue;
        }
        new_objects.push_back(iter->Project(crop_box));
      } else if (image_det_aug_denet_enum::kOverlap == crop_emit_mode) {
        Rect gt_box = iter->ToRect();
        float overlap = (crop_box & gt_box).area() / gt_box.area();
        if (overlap > emit_overlap_thresh) {
          new_objects.push_back(iter->Project(crop_box));
        }
      } else if (image_det_aug_denet_enum::kIou == crop_emit_mode) {
        Rect gt_box = iter->ToRect();
        float overlap = RectIOU(crop_box, gt_box);
        if (overlap > emit_overlap_thresh) {
          new_objects.push_back(iter->Project(crop_box));
        } 
      }
    }
    if (new_objects.size() < 1) return false;
    objects_ = new_objects;  // replace the old objects
    return true;
  }

  /*! \brief try pad image with given pad_box
   * convert all objects afterwards
   */
  bool TryPad(const Rect pad_box) {
    // update all objects inplace
    for (auto it = objects_.begin(); it != objects_.end(); ++it) {
      *it = it->Project(pad_box);
    }
    return true;
  }

  /*! \brief flip image and object coordinates horizontally */
  bool TryMirror() {
    // flip all objects horizontally
    for (auto it = objects_.begin(); it != objects_.end(); ++it) {
      *it = it->HorizontalFlip();
    }
    return true;
  }

 private:
  /*! \brief width for each object information, 5 at least */
  int object_width_;
  /*! \brief vector to store original header info */
  std::vector<float> header_;
  /*! \brief storing objects in more convenient formats */
  std::vector<ImageDetObject> objects_;
};  // class ImageDetLabel

/*! \brief helper class to do image augmentation */
class DenetImageDetAugmenter : public ImageAugmenter {
 public:
  // contructor
  DenetImageDetAugmenter() {}

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    kwargs_left = param_.InitAllowUnknown(kwargs);

    CHECK((param_.inter_method >= 1 && param_.inter_method <= 4) ||
     (param_.inter_method >= 9 && param_.inter_method <= 10))
      << "invalid inter_method: valid value 0,1,2,3,9,10";
    
    
    CHECK_LE(param_.num_crop_sampler, 1);
    
    // validate crop parameters
    ValidateCropParameters(&param_.min_crop_scales, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_scales, param_.num_crop_sampler);
    ValidateCropParameters(&param_.min_crop_aspect_ratios, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_aspect_ratios, param_.num_crop_sampler);
    ValidateCropParameters(&param_.min_crop_overlaps, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_overlaps, param_.num_crop_sampler);
    ValidateCropParameters(&param_.min_crop_sample_coverages, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_sample_coverages, param_.num_crop_sampler);
    ValidateCropParameters(&param_.min_crop_object_coverages, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_object_coverages, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_trials, param_.num_crop_sampler);
    for (int i = 0; i < param_.num_crop_sampler; ++i) {
      CHECK_GE(param_.min_crop_scales[i], 0.0f);
      CHECK_LE(param_.max_crop_scales[i], 1.0f);
      CHECK_GT(param_.max_crop_scales[i], param_.min_crop_scales[i]);
      CHECK_GE(param_.min_crop_aspect_ratios[i], 0.0f);
      CHECK_GE(param_.max_crop_aspect_ratios[i], param_.min_crop_aspect_ratios[i]);
      CHECK_GE(param_.max_crop_overlaps[i], param_.min_crop_overlaps[i]);
      CHECK_GE(param_.max_crop_sample_coverages[i], param_.min_crop_sample_coverages[i]);
      CHECK_GE(param_.max_crop_object_coverages[i], param_.min_crop_object_coverages[i]);
    }
    CHECK_GE(param_.emit_overlap_thresh, 0.0f);
  }
  /*!
   * \brief get interpolation method with given inter_method, 0-CV_INTER_NN 1-CV_INTER_LINEAR 2-CV_INTER_CUBIC
   * \ 3-CV_INTER_AREA 4-CV_INTER_LANCZOS4 9-AUTO(cubic for enlarge, area for shrink, bilinear for others) 10-RAND
   */
  int GetInterMethod(int inter_method, int old_width, int old_height, int new_width,
    int new_height, common::RANDOM_ENGINE *prnd) {
    if (inter_method == 9) {
      if (new_width > old_width && new_height > old_height) {
        return 2;  // CV_INTER_CUBIC for enlarge
      } else if (new_width <old_width && new_height < old_height) {
        return 3;  // CV_INTER_AREA for shrink
      } else {
        return 1;  // CV_INTER_LINEAR for others
      }
      } else if (inter_method == 10) {
      std::uniform_int_distribution<size_t> rand_uniform_int(0, 4);
      return rand_uniform_int(*prnd);
    } else {
      return inter_method;
    }
  }

  /*! \brief Check number of crop samplers and given parameters */
  template<typename DType>
  void ValidateCropParameters(nnvm::Tuple<DType> *param, const int num_sampler) {
    if (num_sampler == 1) {
      CHECK_EQ(param->ndim(), 1);
    } else if (num_sampler > 1) {
      if (param->ndim() == 1) {
        std::vector<DType> vec(num_sampler, (*param)[0]);
        param->assign(vec.begin(), vec.end());
      } else {
        CHECK_EQ(param->ndim(), num_sampler) << "# of parameters/crop_samplers mismatch ";
      }
    }
  }
    
    
   /*! \brief Generate crop box region given cropping parameters */
  Rect GenerateCropBox(const float min_crop_scale,
    const float max_crop_scale, const float min_crop_aspect_ratio,
    common::RANDOM_ENGINE *prnd,
    const int im_size) {
    if(DEBUG_FLAG) {
        printf("min_crop_scale:%f, should be 0.08\n", min_crop_scale);
        printf("max_crop_scale:%f, should be 1\n", max_crop_scale);
        printf("min_crop_aspect_ratio:%f, should be 0.75\n", min_crop_aspect_ratio);
    }
    float target_area_ratio = std::uniform_real_distribution<float>(
        min_crop_scale, max_crop_scale)(*prnd);
    float target_area = target_area_ratio * im_size * im_size + 1e-12f;
    
    
    float aspect_ratio = std::pow(min_crop_aspect_ratio, std::uniform_real_distribution<float>(
        -1.0, 1.0)(*prnd));
        
    float w = std::sqrt(target_area * aspect_ratio) / im_size;
    float h = std::sqrt(target_area / aspect_ratio) / im_size;
    
    float x0 = std::uniform_real_distribution<float>(0.f, 1 - w)(*prnd);
    float y0 = std::uniform_real_distribution<float>(0.f, 1 - h)(*prnd);
    
    if(DEBUG_FLAG) {
        printf("target_area_ratio:%f, aspect_ratio:%f, x0:%f, y0:%f, w:%f, h:%f\n", target_area_ratio, aspect_ratio, x0, y0, w, h);
    }
    return Rect(x0, y0, w, h);
  }
  

  /*! \brief Generate padding box region given padding parameters */
  Rect GeneratePadBox(const float max_pad_scale,
    common::RANDOM_ENGINE *prnd, const float threshold = 1.05f) {
      float new_scale = std::uniform_real_distribution<float>(
        1.f, max_pad_scale)(*prnd);
      if (new_scale < threshold) return Rect(0, 0, 0, 0);
      auto rand_uniform = std::uniform_real_distribution<float>(0.f, new_scale - 1);
      float x0 = rand_uniform(*prnd);
      float y0 = rand_uniform(*prnd);
      return Rect(-x0, -y0, new_scale, new_scale);
    }
  
  
  cv::Mat Process(const cv::Mat &src, std::vector<float> *label,
                  common::RANDOM_ENGINE *prnd) override {
    // init hyper params
    float rgb_eigen_val_arr[3] = {0.2175,  0.0188, 0.0045};
    
    float rgb_eigen_vec_arr[3][3] = {{-0.5675,  0.7192,  0.4009}, 
                                    {-0.5808, -0.0045, -0.8140},
                                    {-0.5836, -0.6948,  0.4203}};
                                    
    const cv::Mat rgb_eigen_val_mat(3, 1, CV_32F, &rgb_eigen_val_arr);
    const cv::Mat rgb_eigen_vec_mat(3, 3, CV_32F, &rgb_eigen_vec_arr);

    const float color_space_random_value = 0.1;
    const float photometric_random_value = 0.4;
    
    
    // init internal resource   
    cv::Mat res = src;
    ImageDetLabel det_label(*label);
    std::uniform_real_distribution<float>  norm_rand_uniform(0, 1);

    
    // padding
    int max_size = std::max(res.cols, res.rows);
    int offset_x = -1 * (max_size - res.cols) / 2;
    int offset_y = -1 * (max_size - res.rows) / 2;
    Rect pad_box(1.0*offset_x/res.cols, 1.0*offset_y/res.rows, 1.0*max_size/res.cols, 1.0*max_size/res.rows);
    temp_ = res;
    
    
    int magic_index = (int)(norm_rand_uniform(*prnd) * 2310941);
    int left = static_cast<int>(-pad_box.x * res.cols + 0.5);
    int top = static_cast<int>(-pad_box.y * res.rows + 0.5);
    //int right = static_cast<int>((pad_box.width + pad_box.x - 1) * res.cols + 0.5);
    //int bot = static_cast<int>((pad_box.height + pad_box.y - 1) * res.rows + 0.5);
    int right = max_size - res.cols - left;
    int bot = max_size - res.rows - top;
    CHECK_GE(right, 0);
    CHECK_GE(bot, 0);
    CHECK_LE(std::abs(right - left), 1);
    CHECK_LE(std::abs(bot - top), 1);

    //printf("magic_index:%d res.cols:%d, res.rows:%d\n", magic_index,res.cols, res.rows);
    //printf("magic_index:%d offset_x:%d, offset_y:%d\n", magic_index,offset_x, offset_y);
    //printf("magic_index:%d pad_box=(%f, %f, %f, %f)\n", magic_index,pad_box.x, pad_box.y, pad_box.width, pad_box.height);
    //printf("magic_index:%d pad:(%d, %d, %d, %d)\n", magic_index,top, bot, left, right);
   
    cv::copyMakeBorder(temp_, res, top, bot, left, right, cv::BORDER_ISOLATED,
                        cv::Scalar(param_.fill_value, param_.fill_value, param_.fill_value));
    //printf("magic_index:%d res.cols:%d, res.rows:%d\n",magic_index, res.cols, res.rows);

    det_label.TryPad(pad_box);

    CHECK_EQ(res.rows, max_size);
    CHECK_EQ(res.cols, max_size);
    
    //printf("magic_index:%d starting crop augmentation\n", magic_index);

    // crop augmentation
    CHECK_LE(param_.num_crop_sampler, 1);
    bool crop_success = false;
    if (param_.rand_crop_prob > 0 && param_.num_crop_sampler > 0) {
        for (int t = 0; t < param_.max_crop_trials[0]; ++t) {
            Rect crop_box = GenerateCropBox(param_.min_crop_scales[0],
                                              param_.max_crop_scales[0], param_.min_crop_aspect_ratios[0],
                                              prnd, max_size);
            
            
            int left = static_cast<int>(crop_box.x * res.cols);
            int top = static_cast<int>(crop_box.y * res.rows);
            int width = static_cast<int>(crop_box.width * res.cols);
            int height = static_cast<int>(crop_box.height * res.rows);
            if(DEBUG_FLAG) {
                printf("crop_box abs:(%d, %d, %d, %d), rel:(%f, %f, %f, %f)\n", left, top, width, height, 
                        crop_box.x, crop_box.y, crop_box.width, crop_box.height);
            }
              
              
            if(crop_box.width <= 1 && 
               crop_box.height <= 1 && 
               det_label.TryCrop(crop_box, param_.min_crop_overlaps[0],
                param_.crop_emit_mode,
                param_.emit_overlap_thresh)) {
              // crop image
              
              res = res(cv::Rect(left, top, width, height));
              
              crop_success = true;
              break;
            }
      }
    }
    
    if(DEBUG_FLAG) {
        printf("crop_success:%d\n", crop_success);
    }
    
    //printf("magic_index:%d starting rescale processing\n", magic_index);
    // do rescale processing
    if (image_det_aug_denet_enum::kForce == param_.resize_mode) {
      // force resize to specified data_shape, regardless of aspect ratio
      int new_height = param_.data_shape[1];
      int new_width = param_.data_shape[2];
      int interpolation_method = GetInterMethod(param_.inter_method,
                   res.cols, res.rows, new_width, new_height, prnd);
      cv::resize(res, res, cv::Size(new_width, new_height),
                   0, 0, interpolation_method);
    } else if (image_det_aug_denet_enum::kShrink == param_.resize_mode) {
      // try to keep original size, shrink if too large
      float h = param_.data_shape[1];
      float w = param_.data_shape[2];
      if (res.rows > h || res.cols > w) {
        float ratio = std::min(h / res.rows, w / res.cols);
        int new_height = ratio * res.rows;
        int new_width = ratio * res.cols;
        int interpolation_method = GetInterMethod(param_.inter_method,
                     res.cols, res.rows, new_width, new_height, prnd);
        cv::resize(res, res, cv::Size(new_width, new_height),
                    0, 0, interpolation_method);
      }
    } else if (image_det_aug_denet_enum::kFit == param_.resize_mode) {
      float h = param_.data_shape[1];
      float w = param_.data_shape[2];
      float ratio = std::min(h / res.rows, w / res.cols);
      int new_height = ratio * res.rows;
      int new_width = ratio * res.cols;
      int interpolation_method = GetInterMethod(param_.inter_method,
                   res.cols, res.rows, new_width, new_height, prnd);
      cv::resize(res, res, cv::Size(new_width, new_height),
                  0, 0, interpolation_method);
    } else if (image_det_aug_denet_enum::kForceFit == param_.resize_mode) {
      float h = param_.data_shape[1];
      float w = param_.data_shape[2];
      
      int new_width = w;
      int new_height = h;
      if (((float)w/res.cols) < ((float)h/res.rows)) {
        new_width = w;
        new_height = (res.rows * w)/res.cols;
      } else {
        new_height = h;
        new_width = (res.cols * h)/res.rows;
      }
      int interpolation_method = GetInterMethod(param_.inter_method,
                   res.cols, res.rows, new_width, new_height, prnd);
      cv::resize(res, res, cv::Size(new_width, new_height),
                  0, 0, interpolation_method);

      temp_ = res;

      int left = static_cast<int>(((int)w-new_width)/2);
      int top = static_cast<int>(((int)h-new_height)/2);
      int right = static_cast<int>(((int)w-new_width) - ((int)w-new_width)/2);
      int bot = static_cast<int>(((int)h-new_height) - ((int)h-new_height)/2);
      cv::copyMakeBorder(temp_, res, top, bot, left, right, cv::BORDER_ISOLATED,
            cv::Scalar(param_.fill_value, param_.fill_value, param_.fill_value));
      
      Rect pad_box(-(float)left / new_width, -(float)top / new_height, w / new_width, h / new_height);
      det_label.TryPad(pad_box);
    }
    
    try {
        //printf("magic_index:%d starting photometric augmentation\n", magic_index);
        // photometric augmentation
        std::uniform_real_distribution<float> photometric_rand_uniform(1-photometric_random_value, 1+photometric_random_value);
        int photometric_types[3] = {0,1,2};
        std::random_shuffle(photometric_types, photometric_types+3);
        for(int type_idx = 0;type_idx < 3; type_idx++) {
            float alpha = photometric_rand_uniform(*prnd);
            if(photometric_types[type_idx] == 0) { 
                // brightness
                res = res * alpha;
            } else if(photometric_types[type_idx] == 1) {
                // contrast
                std::vector<cv::Mat> channel_imgs;
                cv::split(res, channel_imgs);
                cv::Mat gray_img = 0.114*channel_imgs[0] + 0.587*channel_imgs[1] + 0.299*channel_imgs[2];
                res = res * alpha + (1-alpha) * cv::mean(gray_img).val[0];
            } else if(photometric_types[type_idx] == 2) {
                // saturation
                std::vector<cv::Mat> channel_imgs;
                cv::split(res, channel_imgs);
                cv::Mat gray_img = 0.114*channel_imgs[0] + 0.587*channel_imgs[1] + 0.299*channel_imgs[2];
                std::vector<cv::Mat> merged_img = {gray_img, gray_img, gray_img};
                cv::merge(merged_img, gray_img);
                res = res*alpha + (1.0 - alpha)*gray_img;
            }
        }
    }
    catch (cv::Exception& e) {
        std::cout << "magic_index:" << magic_index << ", photometric, Standard exception: " << e.what() << std::endl;
        throw;
    }        
    
    
    try {
        //printf("magic_index:%d starting color space augmentation\n", magic_index);
        // color space augmentation
        cv::Mat aug_rgb_mat(3, 1, CV_32F);
        std::normal_distribution<float> color_rand_normal(0, color_space_random_value);
        
        //printf("magic_index:%d, generating aug_rgb_mat start\n", magic_index);

        for(int c = 0; c < 3; c++) {
            aug_rgb_mat.at<float>(c, 0) = rgb_eigen_val_mat.at<float>(c, 0) * color_rand_normal(*prnd);
        }

        //printf("magic_index:%d, generating aug_rgb_mat running\n", magic_index);

        aug_rgb_mat = rgb_eigen_vec_mat * aug_rgb_mat;
        
        if (DEBUG_FLAG) {
            printf("aug_rgb_mat:(%f, %f, %f)\n", aug_rgb_mat.at<float>(0, 0), aug_rgb_mat.at<float>(1, 0), aug_rgb_mat.at<float>(2, 0));
        }
        
        
        //printf("magic_index:%d, generating aug_rgb_mat done\n", magic_index);
        for (int i = 0; i < res.rows; ++i) {
            for (int j = 0; j < res.cols; ++j) {
                for (int k = 0; k < 3; ++k) {
                  int v = res.at<cv::Vec3b>(i, j)[k];
                  v += aug_rgb_mat.at<float>(2 - k, 0) * 255;
                  res.at<cv::Vec3b>(i, j)[k] = std::max(0, std::min(v, 254));
                }
            }
        }
    }
    catch (cv::Exception& e) {
        std::cout << "magic_index:" << magic_index << ", color space, Standard exception: " << e.what() << std::endl;
        throw;
    }   
    
    //printf("magic_index:%d starting mirror logic\n", magic_index);
    // random mirror logic
    if (param_.rand_mirror_prob > 0 && norm_rand_uniform(*prnd) < param_.rand_mirror_prob) {
      if (det_label.TryMirror()) {
        // flip image
        cv::flip(res, temp_, 1);
        res = temp_;
      }
    }

    *label = det_label.ToArray();  // put back processed labels
    //printf("magic_index:%d done\n", magic_index);
    
    return res;
  }
  

 private:
  // temporal space
  cv::Mat temp_;
  // parameters
  DenetImageDetAugmentParam param_;
};

MXNET_REGISTER_IMAGE_AUGMENTER(det_aug_denet)
.describe("denet detection augmenter")
.set_body([]() {
    return new DenetImageDetAugmenter();
  });
#endif  // MXNET_USE_OPENCV
}  // namespace io
}  // namespace mxnet
