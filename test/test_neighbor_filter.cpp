#include "core/mat.h"
#include "image_proc/neighbor_filter.h"
#include "opencv2/imgcodecs.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>

using namespace fish::core;
using namespace fish::image_proc::neighbor_filter;
int main() {
    std::string image_path = "/mnt/d/images/test_images/test_01.tif";
    cv::Mat     image      = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    int         height     = image.rows;
    int         width      = image.cols;
    int         channels   = image.channels();
    // image.convertTo(image, CV_32FC1);
    uint8_t*           image_ptr = image.data;
    ImageMat<uint8_t>  input_mat(height, width, channels, image_ptr);
    ImageMat<uint8_t>  output_mat(height, width, channels);
    bool               pad_edges    = false;
    int                binary_count = 0;
    auto               start        = std::chrono::steady_clock::now();
    NeighborFilterType filter_type  = NeighborFilterType::BLUR_MORE;
    neighbor_filter_with_3x3_window(input_mat, output_mat, filter_type, pad_edges, binary_count);
    auto   end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    std::cout << "elased " << elapsed << " ms" << std::endl;
    std::cout << output_mat << std::endl;
    return 0;
}