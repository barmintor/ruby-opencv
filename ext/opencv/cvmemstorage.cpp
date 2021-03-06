/************************************************************

   cvmemstorage.cpp -

   $Author: lsxi $

   Copyright (C) 2005 Masakazu Yonekura

************************************************************/
#include "cvmemstorage.h"
/*
 * Document-class: OpenCV::CvMemStorage
 *
 * Internal memory management class used by CvSeq.
 */
__NAMESPACE_BEGIN_OPENCV
__NAMESPACE_BEGIN_CVMEMSTORAGE

VALUE rb_klass;

VALUE
rb_class()
{
  return rb_klass;
}

VALUE
rb_allocate(VALUE klass)
{
  CvMemStorage *storage = rb_cvCreateMemStorage(0);
  return Data_Wrap_Struct(klass, 0, cvmemstorage_free, storage);
}

void
cvmemstorage_free(void *ptr)
{
  try {
    cvReleaseMemStorage((CvMemStorage**)&ptr);
  }
  catch (cv::Exception& e) {
    raise_cverror(e);
  }
}

VALUE
new_object(int blocksize)
{
  CvMemStorage *storage = rb_cvCreateMemStorage(blocksize);
  return Data_Wrap_Struct(rb_klass, 0, cvmemstorage_free, storage);
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
  /* 
   * opencv = rb_define_module("OpenCV");
   * 
   * note: this comment is used by rdoc.
   */
  VALUE opencv = rb_module_opencv();
  rb_klass = rb_define_class_under(opencv, "CvMemStorage", rb_cObject);
}

__NAMESPACE_END_CVMEMSTORAGE
__NAMESPACE_END_OPENCV

