#include "core/mat.h"
#include "image_proc/rank_filter.h"
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgcodecs.hpp>

using namespace fish::image_proc::rank_filter;
using namespace fish::core;

int main() {
    double radius = 4.27;   // test for large radius
    // test for small radius
    // double      radius     = 0.27;   // but the small kernel have some errors!
    std::string image_path = "/mnt/d/images/test_images/test_01.tif";
    cv::Mat     image      = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    image.convertTo(image, CV_32FC1);
    float*          img_ptr  = reinterpret_cast<float*>(image.data);
    int             height   = image.rows;
    int             width    = image.cols;
    int             channels = image.channels();
    ImageMat<float> input_mat(height, width, channels, img_ptr, MatMemLayout::LayoutRight, false);
    ImageMat<float> output_mat(height, width, channels);
    constexpr FilterType filter_type = FilterType::VARIANCE;
    rank_filter(input_mat, output_mat, filter_type, radius);
    std::cout << output_mat << std::endl;

    rank_filter(input_mat, input_mat, filter_type, radius);
    std::cout << input_mat << std::endl;
    return 0;
}