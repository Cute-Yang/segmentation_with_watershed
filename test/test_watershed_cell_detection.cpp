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

    // set the detect params...
    // double backgound_radius = 15.36;
    // double max_background   = std::numeric_limits<double>::infinity();
    // double median_radius    = 0.0;
    // double sigma            = 2.88;
    // double threhsold        = 25.0;
    // double min_area         = 9.216;
    // double max_area         = 460.8;
    // double cell_expansion   = 0.0;
    // if zoom in the image,the specified prefered pixel size < image pixel size,we will use the
    // maximum value,so will get image pixel size,
    // also the downsample factor is also 1.0

    // if zoom out,the specified prefered pixel size  > imae pixel,we will do smapling for the data
    // so downsample factor > 1.0

    double background_radius       = 0.16;
    double max_background          = std::numeric_limits<double>::infinity();
    double median_radius           = 0.0;
    double sigma                   = 0.03;
    double threhsold               = 25.0;
    double min_area                = 0.001;
    double max_area                = 0.05;
    double cell_expansion          = 0.0;
    double pixel_size_microns      = 0.010417;
    bool   have_pixel_size_microns = true;
    if (have_pixel_size_microns) {
        if (pixel_size_microns == 0.0) {
            LOG_ERROR("the pixel size macrons can not be zero...");
            return -1;
        }
        LOG_INFO("we will scale these params by pixel size microns....");
        background_radius = background_radius / pixel_size_microns;
        median_radius     = median_radius / pixel_size_microns;
        sigma             = sigma / pixel_size_microns;
        min_area          = min_area / (pixel_size_microns * pixel_size_microns);
        max_area          = max_area / (pixel_size_microns * pixel_size_microns);
        cell_expansion    = cell_expansion / pixel_size_microns;
    }
    bool merge_all             = true;
    bool watershed_postprocess = true;
    bool exclude_DAB           = false;
    bool smooth_boundaries     = true;
    bool make_measurements     = true;
    int  z                     = 0;
    int  t                     = 0;

    LOG_INFO("cell "
             "detection "
             "params\n********************************^_^*********************************"
             "\nbackground_raidus:{}"
             "\nmedian_radius:"
             "{}\nsigma:{}\nthreshold:"
             "{}\nmin_area:{}\nmax_"
             "area:{}\ncell_expansion:{}\nmax_background:{}\nmerge_all:{}\nwatershed_postprocess:{}"
             "\nexclude_DAB:{}\nsmooth_boundaries:{}\nmake_measurements:{}\n***********************"
             "**********^_^*******************"
             "**************",
             background_radius,
             median_radius,
             sigma,
             threhsold,
             min_area,
             max_area,
             cell_expansion,
             max_background,
             merge_all,
             watershed_postprocess,
             exclude_DAB,
             smooth_boundaries,
             make_measurements);
    cell_detector.set_background_radius(background_radius);
    cell_detector.set_median_radius(median_radius);
    cell_detector.set_sigma(sigma);
    cell_detector.set_threshold(threhsold);
    cell_detector.set_min_area(min_area);
    cell_detector.set_max_area(max_area);
    cell_detector.set_merge_all(merge_all);
    cell_detector.set_watershed_postprocess(watershed_postprocess);
    cell_detector.set_exclude_DAB(exclude_DAB);
    cell_detector.set_smooth_boundaries(smooth_boundaries);
    cell_detector.set_maek_measurements(make_measurements);
    cell_detector.cell_detection(original_image, 0);
    return 0;
}