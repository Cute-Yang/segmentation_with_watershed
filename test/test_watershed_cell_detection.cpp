#include "opencv2/opencv.hpp"
#include "segmentation/watershed_cell_detection.h"
#include "utils/logging.h"
#include <limits>

using namespace fish::segmentation::watershed_cell_detection;

int main() {
    std::string image_path = "/mnt/d/images/test_images/test_01.tif";
    cv::Mat     image      = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    int         height     = image.rows;
    int         width      = image.cols;
    int         channels   = image.channels();
    image.convertTo(image, CV_32FC1);
    float* image_ptr = reinterpret_cast<float*>(image.data);

    ImageMat<float> original_image(
        height, width, channels, image_ptr, MatMemLayout::LayoutRight, false);

    WatershedCellDetector cell_detector;

    double background_radius            = 0.16;
    double max_background               = std::numeric_limits<double>::infinity();
    double median_radius                = 0.0;
    double sigma                        = 0.03;
    double threhsold                    = 25.0;
    double min_area                     = 0.001;
    double max_area                     = 0.05;
    double cell_expansion               = 0.0;
    double pixel_size_microns           = 0.010417;
    double requested_pixel_size         = 0.01;
    bool   have_pixel_size_microns      = true;
    bool   merge_all                    = true;
    bool   watershed_postprocess        = true;
    bool   exclude_DAB                  = false;
    bool   smooth_boundaries            = true;
    bool   make_measurements            = true;
    bool   background_by_reconstruction = true;
    int    z                            = 0;
    int    t                            = 0;

    cell_detector.set_background_radius(background_radius);
    cell_detector.set_max_background(max_background);
    cell_detector.set_median_radius(median_radius);
    cell_detector.set_sigma(sigma);
    cell_detector.set_threshold(threhsold);
    cell_detector.set_min_area(min_area);
    cell_detector.set_max_area(max_area);
    cell_detector.set_merge_all(merge_all);
    cell_detector.set_watershed_postprocess(watershed_postprocess);
    cell_detector.set_exclude_DAB(exclude_DAB);
    cell_detector.set_smooth_boundaries(smooth_boundaries);
    cell_detector.set_make_measurements(make_measurements);
    cell_detector.set_background_by_reconstruction(background_by_reconstruction);
    cell_detector.set_have_pixle_size_microns(have_pixel_size_microns);
    cell_detector.set_pixel_size_microns(pixel_size_microns, pixel_size_microns);
    cell_detector.set_requested_pixel_size(requested_pixel_size);
    cell_detector.cell_detection(original_image, 0);
    return 0;
}