#!/usr/bin/env ruby
# -*- mode: ruby; coding: utf-8 -*-
require 'test/unit'
require 'opencv'
require File.expand_path(File.dirname(__FILE__)) + '/helper'

include OpenCV

# Tests for OpenCV::CascadeClassifier
class TestCascadeClassifier < OpenCVTestCase
  def setup
    @cascade = CascadeClassifier.new
    assert_equal(@cascade.empty?, true)
    @cascade.load(HAARCASCADE_FRONTALFACE_ALT)
  end
  
  def test_detect_objects
    img = CvMat.load(FILENAME_LENA256x256)

    detected = @cascade.detect_objects(img)
    assert_equal(Array, detected.class)
    assert_equal(1, detected.size)
    assert_equal(CvRect, detected[0].class)
    assert_equal(89, detected[0].width)
    assert_equal(89, detected[0].height)
    assert_equal(106, detected[0].x)
    assert_equal(100, detected[0].y)

    detected = @cascade.detect_objects(img) { |face|
      assert_equal(106, face.x)
      assert_equal(100, face.y)
      assert_equal(89, face.width)
      assert_equal(89, face.height)
    }
    assert_equal(Array, detected.class)
    assert_equal(1, detected.size)
    assert_equal(CvRect, detected[0].class)

    detected = @cascade.detect_objects(img, :scale_factor => 2.0, :flags => CV_HAAR_DO_CANNY_PRUNING,
                                       :min_neighbors => 5, :min_size => CvSize.new(10, 10),
                                       :max_size => CvSize.new(100, 100))
    assert_equal(Array, detected.class)
    assert_equal(1, detected.size)
    assert_equal(CvRect, detected[0].class)
    assert_equal(109, detected[0].x)
    assert_equal(102, detected[0].y)
    assert_equal(80, detected[0].width)
    assert_equal(80, detected[0].height)

    assert_raise(TypeError) {
      @cascade.detect_objects('foo')
    }
  end

  def test_group_rectangles
    rectangles = [10,12,14,16,18,20].map {|x| CvRect.new(x, x, 80, 80) }
    group_threshold = 2
    eps = 0.2
    weights = [1]
    grouped, weights = CascadeClassifier.group_rectangles(rectangles, group_threshold, eps, weights)
    assert_equal(15, grouped[0].x)
    assert_equal(15, grouped[0].y)
    assert_equal(80, grouped[0].width)
    assert_equal(80, grouped[0].height)
    assert_equal(1, weights[0])

    grouped, weights = CascadeClassifier.group_rectangles(rectangles, group_threshold, eps) do |rect|
      assert_equal(15, rect.x)
      assert_equal(15, rect.y)
      assert_equal(80, rect.width)
      assert_equal(80, rect.height)
    end
    assert_equal(nil, weights)
  end
end

