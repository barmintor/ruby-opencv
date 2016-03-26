/************************************************************

   cascadeclassifier.cpp -

   $Author: barmintor $

   Copyright (C) 2016-2017 Benjamin Armintor

************************************************************/
#include "cascadeclassifier.h"
/*
 * Document-class: OpenCV::CascadeClassifier
 *
 * Haar/LBP Cascade Classifier for Object Detection
 */
__NAMESPACE_BEGIN_OPENCV
__NAMESPACE_BEGIN_CASCADECLASSIFIER

VALUE rb_klass;

VALUE
rb_class()
{
  return rb_klass;
}

VALUE
rb_allocate(VALUE klass)
{
  cv::CascadeClassifier* cc = new cv::CascadeClassifier;
  return Data_Wrap_Struct(klass, mark_root_object, release_object, cc);
}

void
rb_release(void *ptr)
{
  if (ptr) {
    // strange but borrowed from CascadeClassifier
    CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade*)ptr;
    cvReleaseHaarClassifierCascade(&cascade);
  }
}

VALUE
rb_empty_q(VALUE self)
{
  cv::CascadeClassifier *self_ptr = CASCADECLASSIFIER(self);
  return CASCADECLASSIFIER(self)->empty() ? Qtrue : Qfalse;
}

/*
 * Load trained cascade from file.
 * May be LBP or Haar trained features.
 */
VALUE
rb_load(VALUE self, VALUE path)
{
  try {
    Check_Type(path, T_STRING);
    char* s = StringValueCStr(path);
    CASCADECLASSIFIER(self)->load(std::string(s));
  }
  catch (cv::Exception& e) {
    raise_cverror(e);
  }

  return Qnil;
}

/*
 * Detects objects of different sizes in the input image.
 *
 * @overload detect_objects(image, options = nil)
 *   @param image [CvMat,IplImage] an image where objects are detected.
 *   @param options [Hash] Options
 *   @option options [Number] :scale_factor
 *     Parameter specifying how much the image size is reduced at each image scale.
 *   @option options [Number] :storage
 *      Memory storage to store the resultant sequence of the object candidate rectangles
 *   @option options [Number] :min_neighbors
 *      Parameter specifying how many neighbors each candidate rectangle should have to retain it.
 *   @option options [CvSize] :min_size
 *      Minimum possible object size. Objects smaller than that are ignored.
 *   @option options [CvSize] :max_size
 *      Maximum possible object size. Objects larger than that are ignored.
 * @return [Array<CvRect>] Detected objects as a list of rectangles
 * @opencv_func CascadeClassifier::detectMultiScale
 */
VALUE
rb_detect_objects(int argc, VALUE *argv, VALUE self)
{ 
  VALUE image, options;
  rb_scan_args(argc, argv, "11", &image, &options);

  double scale_factor;
  int flags, min_neighbors;
  CvSize min_size, max_size;

  if (NIL_P(options)) {
    scale_factor = 1.1;
    flags = 0;
    min_neighbors = 3;
    min_size = max_size = cvSize(0, 0);
  }
  else {
    scale_factor = IF_DBL(LOOKUP_HASH(options, "scale_factor"), 1.1);
    flags = IF_INT(LOOKUP_HASH(options, "flags"), 0);
    min_neighbors = IF_INT(LOOKUP_HASH(options, "min_neighbors"), 3);
    VALUE min_size_val = LOOKUP_HASH(options, "min_size");
    min_size = NIL_P(min_size_val) ? cvSize(0, 0) : VALUE_TO_CVSIZE(min_size_val);
    VALUE max_size_val = LOOKUP_HASH(options, "max_size");
    max_size = NIL_P(max_size_val) ? cvSize(0, 0) : VALUE_TO_CVSIZE(max_size_val);
  }
  std::vector<cv::Rect> rect_vector;
  VALUE result = Qnil;
  try {
    CASCADECLASSIFIER(self)->detectMultiScale(CVMAT_WITH_CHECK(image), rect_vector,
      scale_factor, min_neighbors, flags, min_size, max_size);
    int len = rect_vector.size();
    result = rb_ary_new2(len);
    for(int i = 0; i < len; ++i) {
      cv::Rect r = rect_vector[i];
      CvRect *rect;
      rect = RB_CVALLOC(CvRect);
      rect->x = r.x;
      rect->y = r.y;
      rect->width = r.width;
      rect->height = r.height;
      VALUE rect_val = Data_Wrap_Struct(cCvRect::rb_class(), mark_root_object, release_object, rect);
      if (rb_block_given_p()) {
        rb_yield(rect_val);
      }
      rb_ary_store(result, i, rect_val);
    }
  }
  catch (cv::Exception& e) {
    raise_cverror(e);
  }
  return result;
}

/*
 * Groups the object candidate rectangles.
 *
 * @overload group_rectangles(image, options = nil)
 *   @param rectangles [Array<CvRect>] Rectangles of detected features
 *   @param groupThreshold [Fixnum] Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
 *   @param eps [Number] Relative difference between sides of the rectangles to merge them into a group. Default 0.2
 * @return [Array<CvRect>, Array<int>] Grouped rectangles, weights
 * @opencv_func CascadeClassifier.group_rectangles
 */
VALUE
rb_group_rectangles(int argc, VALUE *argv, VALUE self)
{
  VALUE detected, threshold_val, eps_val, weights;
  rb_scan_args(argc, argv, "22", &detected, &threshold_val, &eps_val, &weights);
  double eps;
  int group_threshold = NUM2INT(threshold_val);

  if (NIL_P(eps_val)) {
    eps = 0.2;
  }
  else {
    eps = NUM2DBL(eps_val);
  }
  std::vector<cv::Rect> rect_vector;
  // copy the detected rectangles into the vector
  CvRect* cv_rect;
  for(int i = 0; i < RARRAY_LEN(detected); i++) {
    cv_rect = CVRECT(rb_ary_entry(detected, i));
    rect_vector.push_back(cv::Rect(*cv_rect));
  }
  // dispatch to OpenCV
  if (NIL_P(weights)) {
    weights = Qnil;
    cv::groupRectangles(rect_vector, group_threshold, eps);
  } else {
    std::vector<int> weights_vector;
    for(int i = 0; i < RARRAY_LEN(weights); i++) {
      weights_vector.push_back(NUM2INT(rb_ary_entry(weights,i)));
    }
    cv::groupRectangles(rect_vector, weights_vector, group_threshold, eps);
  }
  // copy grouped rectangles to result
  VALUE rect_array = rb_ary_new2(rect_vector.size());
  for (std::vector<cv::Rect>::iterator it = rect_vector.begin() ; it != rect_vector.end(); ++it) {
    CvRect *rect;
    rect = RB_CVALLOC(CvRect);
    rect->x = it->x;
    rect->y = it->y;
    rect->width = it->width;
    rect->height = it->height;
    VALUE rect_val = Data_Wrap_Struct(cCvRect::rb_class(), mark_root_object, release_object, rect);
    if (rb_block_given_p()) {
      rb_yield(rect_val);
    }
    rb_ary_push(rect_array, rect_val);
  }

  VALUE result = rb_ary_new2(2);
  rb_ary_store(result, 0, rect_array);
  rb_ary_store(result, 1, weights);
  return result;
}

void
init_ruby_class()
{
#if 0
  // For documentation using YARD
  VALUE opencv = rb_define_module("OpenCV");
#endif

  if (rb_klass)
    return;

  VALUE opencv = rb_module_opencv();
  VALUE cascade_classifier = cCascadeClassifier::rb_class();
  rb_klass = rb_define_class_under(opencv, "CascadeClassifier", rb_cObject);
  rb_define_alloc_func(rb_klass, rb_allocate);
  rb_define_method(rb_klass, "empty?", RUBY_METHOD_FUNC(rb_empty_q), 0);
  rb_define_method(rb_klass, "load", RUBY_METHOD_FUNC(rb_load), 1);
  rb_define_method(rb_klass, "detect_objects", RUBY_METHOD_FUNC(rb_detect_objects), -1);
  rb_define_module_function(rb_klass, "group_rectangles", RUBY_METHOD_FUNC(rb_group_rectangles), -1);
}

__NAMESPACE_END_CASCADECLASSIFIER
__NAMESPACE_END_OPENCV
