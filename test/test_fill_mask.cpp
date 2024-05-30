#include "common/fishdef.h"
#include "core/mat.h"
#include "image_proc/fill_mask.h"
#include "utils/logging.h"
#include <chrono>
#include <cstdint>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

using namespace fish::image_proc::fill_mask;
using namespace fish::core;
void fill_polygon_test(const std::vector<Coordinate2d>& points,
                       const std::string&               image_output_path) {
    // the point we already move the left upper to (0,0)
    // find the max x/y coor,use x+1/y+1 as the width and height
    int xmax = 0;
    int ymax = 0;

    for (size_t i = 0; i < points.size(); ++i) {
        xmax = FISH_MAX(points[i].x, xmax);
        ymax = FISH_MAX(points[i].y, ymax);
    }
    int height = ymax + 1;
    int width  = xmax + 1;
    LOG_INFO("generate a mask with height {} width {}", height, width);
    cv::Mat           cv_mask(height, width, CV_8UC1);
    uint8_t*          data_ptr = cv_mask.data;
    ImageMat<uint8_t> mask(height, width, 1, data_ptr, MatMemLayout::LayoutRight, false);
    mask.set_zero();

    PolygonFiller poly_filler;
    auto          start = std::chrono::steady_clock::now();
    poly_filler.fill_polygon(points.data(), points.size(), mask, 255);
    auto   end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_INFO("elapsed {}ms", elapsed);
    cv::imwrite(image_output_path, cv_mask);
}


int main() {
    // std::vector<Coordinate2d> points = {{10, 50}, {300, 200}, {50, 500}};
    std::vector<Coordinate2d> points_transilation = {
        {2827, 41},   {3052, 144},  {3114, 144},  {3237, 226},  {3667, 594},  {4118, 1332},
        {4220, 1578}, {4282, 1783}, {4287, 1836}, {5121, 1783}, {5101, 2930}, {4957, 3319},
        {4835, 3483}, {4609, 3749}, {4241, 3995}, {3544, 4261}, {3114, 4363}, {2090, 4363},
        {1352, 4200}, {1066, 4077}, {574, 3708},  {410, 3524},  {62, 2889},   {0, 2663},
        {0, 1967},    {41, 1721},   {410, 1045},  {820, 615},   {1107, 369},  {1352, 226},
        {1926, 21},   {2499, 0}};
    std::string image_path = "../test_images/lazydog_transilation.png";
    fill_polygon_test(points_transilation, image_path);

    std::vector<Coordinate2d> points{
        {10433, 12577}, {10658, 12680}, {10720, 12680}, {10843, 12762}, {11273, 13130},
        {11724, 13868}, {11826, 14114}, {11888, 14319}, {11893, 14372}, {12727, 14319},
        {12707, 15466}, {12563, 15855}, {12441, 16019}, {12215, 16285}, {11847, 16531},
        {11150, 16797}, {10720, 16899}, {9696, 16899},  {8958, 16736},  {8672, 16613},
        {8180, 16244},  {8016, 16060},  {7668, 15425},  {7606, 15199},  {7606, 14503},
        {7647, 14257},  {8016, 13581},  {8426, 13151},  {8713, 12905},  {8958, 12762},
        {9532, 12557},  {10105, 12536}};
    image_path = "../test_images/lazydog.png";
    fill_polygon_test(points, image_path);
    // using the original point to do a transilation!

    std::vector<Coordinate2d> points_with_holes;
    return 0;
}