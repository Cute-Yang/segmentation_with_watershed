#include "core/mat.h"
#include "image_proc/guassian_blur.h"
#include "opencv2/imgcodecs.hpp"
#include <chrono>
#include <iostream>

using namespace fish::core;
using namespace fish::image_proc::guassian_blur;
int main() {
    std::string image_path = "/mnt/d/images/test_images/test_01.tif";
    cv::Mat     image      = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    int         height     = image.rows;
    int         width      = image.cols;
    int         channels   = image.channels();
    image.convertTo(image, CV_32FC1);
    float*          image_ptr = reinterpret_cast<float*>(image.data);
    ImageMat<float> input_mat(height, width, channels, image_ptr);
    ImageMat<float> output_mat(height, width, channels);
    double          sigma = 0.43;
    auto            start = std::chrono::steady_clock::now();
    guassian_blur_2d(input_mat, output_mat, sigma);
    auto   end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    std::cout << "elased " << elapsed << " ms" << std::endl;
    std::cout << output_mat << std::endl;
    return 0;
}