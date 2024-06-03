#pragma once
#include "core/mat.h"
#include "image_proc/polygon.h"
#include <vector>
namespace fish {
namespace segmentation {
namespace watershed_cell_detection {
using namespace fish::core::mat;
using namespace fish::image_proc::polygon;

namespace WatershedCellDetectionParam {
constexpr bool   refine_boundary       = true;
constexpr double background_radius     = 15;
constexpr double max_background        = 0.3;
constexpr int    z                     = 0;
constexpr int    t                     = 0;
constexpr double cell_expansion        = 0;
constexpr double min_area              = 0.0;
constexpr double max_area              = 0.0;
constexpr double median_radius         = 2;
constexpr double sigma                 = 2.5;
constexpr double threshold             = 0.3;
constexpr bool   merge_all             = true;
constexpr bool   watershed_postprocess = true;
constexpr bool   exclude_DAB           = false;
constexpr bool   smooth_boundaries     = false;
}   // namespace WatershedCellDetectionParam

class WatershedCellDetector {
private:
    double background_radius;
    double max_background;
    double median_radius;
    double sigma;
    double threshold;
    double min_area;
    double max_area;
    double merge_all;
    bool   watershed_postprocess;
    bool   exclude_DAB;
    double cell_expansion;
    bool   smooth_boundaries;
    bool   make_measurements;
    bool   background_by_reconstruction;
    bool   refine_boundary;

    // the params of image
    bool   have_pixel_size_microns;
    double pixel_size_microns_h;
    double pixel_size_microns_w;
    double requested_pixel_size;

    // this is the polygon of nuclei
    std::vector<PolygonTypef32> nuclei_rois;
    // this is the polygon of cell!
    std::vector<PolygonTypef32> cell_rois;
    void                        transform_params_by_microns();


    // should add 3 statics nuclei_stat/cell_stat/Cytoplasm stat(the area between nuclei and cell)
    // ^_^
public:
    WatershedCellDetector() {
        // here just set the default value...
        refine_boundary       = WatershedCellDetectionParam::refine_boundary;
        background_radius     = WatershedCellDetectionParam::background_radius;
        max_background        = WatershedCellDetectionParam::max_background;
        cell_expansion        = WatershedCellDetectionParam::cell_expansion;
        min_area              = WatershedCellDetectionParam::min_area;
        max_area              = WatershedCellDetectionParam::max_area;
        median_radius         = WatershedCellDetectionParam::median_radius;
        sigma                 = WatershedCellDetectionParam::sigma;
        threshold             = WatershedCellDetectionParam::threshold;
        merge_all             = WatershedCellDetectionParam::merge_all;
        watershed_postprocess = WatershedCellDetectionParam::watershed_postprocess;
        exclude_DAB           = WatershedCellDetectionParam::exclude_DAB;
        smooth_boundaries     = WatershedCellDetectionParam::smooth_boundaries;
    }

    void set_background_radius(double background_radius_) {
        background_radius = background_radius_;
    }

    void set_median_radius(double median_radius_) { median_radius = median_radius_; }

    void set_max_background(double max_background_) { max_background = max_background_; }

    void set_sigma(double sigma_) { sigma = sigma_; }

    void set_threshold(double threshold_) { threshold = threshold_; }

    void set_min_area(double min_area_) { min_area = min_area_; }

    void set_max_area(double max_area_) { max_area = max_area_; }

    void set_merge_all(bool merge_all_) { merge_all = merge_all_; }

    void set_watershed_postprocess(bool watershed_postprocess_) {
        watershed_postprocess = watershed_postprocess_;
    }

    void set_exclude_DAB(bool exclude_DAB_) { exclude_DAB = exclude_DAB_; }

    void set_cell_expansion(double cell_expansion_) { cell_expansion = cell_expansion_; }

    void set_smooth_boundaries(bool smooth_boundaries_) { smooth_boundaries = smooth_boundaries_; }

    void set_make_measurements(bool make_measurements_) { make_measurements = make_measurements_; }

    void set_background_by_reconstruction(bool background_by_reconstruction_) {
        background_by_reconstruction = background_by_reconstruction_;
    }

    void set_refine_boundary(bool refine_boundary_) { refine_boundary = refine_boundary_; }

    void set_have_pixle_size_microns(bool have_pixel_size_microns_) {
        have_pixel_size_microns = have_pixel_size_microns_;
    }

    void set_pixel_size_microns(double pixel_size_microns_h_, double pixel_size_microns_w_) {
        pixel_size_microns_h = pixel_size_microns_h_;
        pixel_size_microns_w = pixel_size_microns_w_;
    }

    // this param will be effective only set have pixel size microns
    void set_requested_pixel_size(double requested_pixel_size_) {
        requested_pixel_size = requested_pixel_size_;
    }

    // this function for image which have microns,be sure your image is sampled with speicfy
    // ratio!
    bool cell_detection(const ImageMat<float>& original_image, int detect_channel,
                        int Hematoxylin_channel, int DAB_channel);

    // for single channel image maybe!
    bool cell_detection(const ImageMat<float>& original_image, int detect_channel);

    // this function for image which do not have micron info..

    bool cell_detection(const ImageMat<uint8_t>& original_image, int detect_channel,
                        int Hematoxylin_channel, int DAB_channel);

    bool cell_detection(const ImageMat<uint8_t>& original_image, int detect_channel);


    std::vector<PolygonTypef32>& get_nuclei_rois_ref() { return nuclei_rois; }

    const std::vector<PolygonTypef32>& get_nuclei_rois_cref() const { return nuclei_rois; }

    std::vector<PolygonTypef32> get_nuclei_rois() const { return nuclei_rois; }

    std::vector<PolygonTypef32>& get_cell_rois_ref() { return cell_rois; }

    const std::vector<PolygonTypef32>& get_cell_rois_cref() const { return cell_rois; }

    std::vector<PolygonTypef32> get_cell_rois() const { return cell_rois; }
};
}   // namespace watershed_cell_detection
}   // namespace segmentation
}   // namespace fish