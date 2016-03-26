/************************************************************

   cascadeclassifier.h -

   $Author: barmintor $

   Copyright (C) 2016-2017 Benjamin Armintor

************************************************************/
#ifndef RUBY_OPENCV_CASCADECLASSIFIER_H
#define RUBY_OPENCV_CASCADECLASSIFIER_H

#include "opencv.h"

#define __NAMESPACE_BEGIN_CASCADECLASSIFIER namespace cCascadeClassifier {
#define __NAMESPACE_END_CASCADECLASSIFIER }

__NAMESPACE_BEGIN_OPENCV
__NAMESPACE_BEGIN_CASCADECLASSIFIER

VALUE rb_class();

void init_ruby_class();

VALUE rb_allocate(VALUE klass);

VALUE rb_empty_q(VALUE self);
VALUE rb_load(VALUE self, VALUE path);
VALUE rb_detect_objects(int argc, VALUE *argv, VALUE self);
VALUE rb_group_rectangles(int argc, VALUE *argv, VALUE klass);
void rb_release(void *ptr);

__NAMESPACE_END_CASCADECLASSIFIER

inline cv::CascadeClassifier*
CASCADECLASSIFIER(VALUE object)
{
  cv::CascadeClassifier* ptr;
  Data_Get_Struct(object, cv::CascadeClassifier, ptr);
  return ptr;
}

__NAMESPACE_END_OPENCV

#endif // RUBY_OPENCV_CASCADECLASSIFIER_H

