#include "core/mat.h"
#include "image_proc/distance_transform.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "utils/logging.h"
#include <chrono>
#include <cstdint>
#include <iostream>

using namespace fish::image_proc::distance_transform;

int main() {
    std::string             image_path = "/mnt/d/images/test_images/test_01.tif";
    cv::Mat                 image      = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    int                     height     = image.rows;
    int                     width      = image.cols;
    int                     channels   = image.channels();
    unsigned char*          image_ptr  = image.data;
    ImageMat<unsigned char> input_mat(
        height, width, channels, image_ptr, MatMemLayout::LayoutRight, false);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (input_mat(y, x, 0) < 20) {
                input_mat(y, x, 0) = 0;
            }
        }
    }

    ImageMat<float> output_mat(height, width, 1, MatMemLayout::LayoutRight);
    uint8_t         backgroup_value = 0;
    auto            start           = std::chrono::steady_clock::now();
    distance_transform<uint8_t>(input_mat, output_mat, false, backgroup_value);
    auto   end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_INFO("distance transform elapsed {}ms", elapsed);
    std::cout << output_mat << std::endl;
}