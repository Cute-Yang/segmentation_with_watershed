#include "core/mat.h"
#include "image_proc/convolution.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "utils/logging.h"
#include "utils/tick_toc.h"
#include <iostream>

using namespace fish::image_proc::convolution;

int main() {
    fish::utils::TickTok tick_toker;
    std::string          image_path = "/mnt/d/images/test_images/test_01.tif";
    cv::Mat              image      = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    int                  height     = image.rows;
    int                  width      = image.cols;
    int                  channels   = image.channels();
    image.convertTo(image, CV_32FC1);
    float* image_ptr = reinterpret_cast<float*>(image.data);

    ImageMat<float> input_mat(height, width, channels, image_ptr, MatMemLayout::LayoutRight, false);
    ImageMat<float> output_mat(height, width, channels);
    std::vector<float> kernel = {
        0.235f, 0.47f, 0.13f, -0.026f, 0.03f, 0.15f, 0.35f, 0.072f, 0.423f};
    int kh = 3;
    int kw = 3;
    tick_toker.tick();
    convolution_2d(input_mat, output_mat, kernel.data(), kh, kw);
    tick_toker.tock();
    double elapsed = tick_toker.compute_elapsed_milli();
    LOG_INFO("elapsed {} ms", elapsed);
    std::cout << output_mat << std::endl;
}