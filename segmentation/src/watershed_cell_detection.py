#include "core/base.h"
#include "core/mat.h"
#include "core/mat_ops.h"
#include "image_proc/convolution.h"
#include "image_proc/distance_transform.h"
#include "image_proc/fill_mask.h"
#include "image_proc/find_contour.h"
#include "image_proc/find_maximum.h"
#include "image_proc/guassian_blur.h"
#include "image_proc/neighbor_filter.h"
#include "image_proc/polygon.h"
#include "image_proc/rank_filter.h"
#include "image_proc/roi_labeling.h"
#include "segmentation/estimate_backgroud.h"
#include "segmentation/morphological_transform.h"
#include "segmentation/shape_simplifier.h"
#include "segmentation/watershed.h"
#include "utils/logging.h"
#include <vector>

namespace fish {
namespace segmentation {
namespace cell_detection {
using namespace fish::image_proc::rank_filter;
using namespace fish::segmentation::estimate;
using namespace fish::core::mat_ops;
using namespace fish::image_proc::guassian_blur;
using namespace fish::image_proc::convolution;
using namespace fish::image_proc::roi_labeling;
using namespace fish::segmentation::watershed;
using namespace fish::image_proc::neighbor_filter;
using namespace fish::image_proc::distance_transform;
using namespace fish::image_proc::find_maximum;

namespace cell_detection_params {
constexpr bool   refine_boundary  = true;
constexpr double backgroun_radius = 15;
constexpr double max_background   = 0.3;
constexpr int    z_dim            = 0;
constexpr int    t_dim            = 0;
constexpr bool   include_nuclei   = true;
constexpr bool   cell_expansion   = true;
constexpr double min_area         = 0.0;
constexpr double max_area         = 0.0;

constexpr double median_radius = 2.0;
constexpr double sigam         = 2.5;
constexpr double threshold     = 0.3;

constexpr bool merge_all = true;

constexpr bool watershed_post_process = true;

constexpr bool exclude_DAB       = false;
constexpr bool smooth_boundaries = false;

constexpr bool background_by_reconstruction = true;

constexpr bool make_measurements = true;



}   // namespace cell_detection_params
class WatershedCellDetector {
private:
    bool   refine_boundary;
    double backgroud_radius;
    double max_background;
    bool   include_nuclei;
    double cell_expansion;
    double min_area;
    double max_area;
    double median_radius;
    double sigma;
    double threshold;
    bool   merge_all;
    bool   watershed_postprocess;
    bool   exclude_DAB;
    bool   smooth_boundaries;
    bool   background_by_reconstruction;

public:
    WatershedCellDetector()
        : refine_boundary(cell_detection_params::refine_boundary)
        , backgroud_radius(cell_detection_params::backgroun_radius)
        , max_background(cell_detection_params::max_background)
        , include_nuclei(cell_detection_params::include_nuclei)
        , cell_expansion(cell_detection_params::cell_expansion)
        , min_area(cell_detection_params::min_area)
        , max_area(cell_detection_params::max_area)
        , median_radius(cell_detection_params::median_radius)
        , sigma(cell_detection_params::sigam)
        , threshold(cell_detection_params::threshold)
        , merge_all(cell_detection_params::merge_all)
        , watershed_postprocess(cell_detection_params::watershed_post_process)
        , exclude_DAB(cell_detection_params::exclude_DAB)
        , smooth_boundaries(cell_detection_params::smooth_boundaries)
        , background_by_reconstruction(cell_detection_params::smooth_boundaries) {}

    void cell_detection(ImageMat<float>& input_mat) {
        int height   = input_mat.get_height();
        int width    = input_mat.get_width();
        int channels = input_mat.get_channels();
        if (channels != 1) {
            LOG_ERROR("we only support single channel image now...");
            return;
        }

        if (median_radius > 0.0) {
            LOG_INFO("apply median filter...");
        }
        Status::ErrorCode run_status;

        ImageMat<float> detection_mat(height, width, channels, MatMemLayout::LayoutRight);
        run_status = rank_filter(input_mat, detection_mat, FilterType::MEDIAN, median_radius);
        if (run_status != Status::ErrorCode::Ok) {
            LOG_ERROR("fail to apply median rank filter...");
            return;
        }

        // something to do...
        if (exclude_DAB) {}

        ImageMat<float> background_mat = detection_mat;
        if (backgroud_radius > 0) {
            LOG_INFO("apply the background estimate...");
            // here must copy,we will chang the value of background_mat
            estimate_background(detection_mat,
                                background_mat,
                                backgroud_radius,
                                max_background,
                                background_by_reconstruction);
            copy_image_mat(background_mat, detection_mat, ValueOpKind::SUBSTRACT);
        }

        // guassian blur
        // here we copy the line to a cache,so we can apply detection_mat inplace!
        guassian_blur_2d(detection_mat, detection_mat, sigma);
        float conv_kernel[9] = {0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0};

        ImageMat<float> conv_mat(height, width, channels, MatMemLayout::LayoutRight);

        convolution_2d(detection_mat, conv_mat, conv_kernel, 3, 3);

        detection_mat.swap(conv_mat);
        conv_mat.release_mat();

        ImageMat<uint8_t> detection_mask = threshold_above(detection_mat, 0.0f);

        // here we should resuse the memory!
        ImageMat<float>    temp = morphological::find_regional_maxima(detection_mat, threshold);
        ImageMat<uint16_t> detect_labels;
        label_image<float, uint16_t>(temp, detect_labels, 0.0f);
        watershed_process<float, uint16_t, NeighborConnectiveType::Conn4>(
            detection_mat, detect_labels, 0.0f);

        std::vector<PolygonType> rois =
            get_filled_polygon_rois(detect_labels, WandMode::FOUR_CONNECTED);

        PolygonFiller     poly_filler;
        ImageMat<uint8_t> roi_mask;
        for (size_t i = 0; i < rois.size(); ++i) {
            // compute the man value
            double mean_value = 0.0f;
            if (mean_value <= threshold) {
                continue;
            }

            double background_value = 0.0f;
            if (background_value > 0) {
                continue;
            }
            // check the background!
            poly_filler.fill_mask(roi_mask, 255);
        }

        // set threshold 127 to roi_mask;
        if (merge_all) {
            neighbor_filter_with_3x3_window(roi_mask, roi_mask, NeighborFilterType::MAX, true, 0);
            copy_image_mat(detection_mask, roi_mask, ValueOpKind::AND);
            if (watershed_postprocess) {
                std::vector<PolygonType> rois2 =
                    get_filled_polygon_rois(roi_mask, WandMode::FOUR_CONNECTED);
                for (size_t i = 0; i < rois2.size(); ++i) {
                    poly_filler.fill_mask(roi_mask, 255);
                }

                ImageMat<float> dist_mat;
                distance_transform<uint8_t>(roi_mask, dist_mat, true, 0, 0);
                ImageMat<uint8_t> s;
                auto              max_mask = find_maxima(
                    dist_mat, s, true, 0.5, NO_THRESHOLD, EDMOutputType::SEGMENTED, false, true);
                if (max_mask.not_empty()) {
                    copy_image_mat(max_mask, roi_mask, ValueOpKind::AND);
                }
            }
        }
        // can fill out side here...
        if (refine_boundary && sigma > 1.5) {
            ImageMat<float> refine_boundary_mat = detection_mat;
            guassian_blur_2d(refine_boundary_mat, refine_boundary_mat, 1.0);
            float           conv_kernel[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
            ImageMat<float> conv_mat(height, width, 1, MatMemLayout::LayoutRight);
            convolution_2d(refine_boundary_mat, conv_mat, conv_kernel, 3, 3);
            auto binary_mask = threshold_above(conv_mat, 0.0f);
            copy_image_mat(roi_mask, binary_mask, ValueOpKind::MIN);
            ImageMat<uint8_t> filter_roi_mask(height, width, 1, MatMemLayout::LayoutRight);
            neighbor_filter_with_3x3_window(
                roi_mask, filter_roi_mask, NeighborFilterType::MIN, true, 0);
            copy_image_mat(binary_mask, filter_roi_mask, ValueOpKind::MAX);
            roi_mask.swap(filter_roi_mask);
        }

        std::vector<PolygonType> rois_nuclei =
            get_filled_polygon_rois(roi_mask, WandMode::FOUR_CONNECTED);

        if (min_area > 0 || max_area > 0) {
            // filter rois...
        }
        ImageMat<uint16_t> roi_labels(height, width, 1, MatMemLayout::LayoutRight);
        // set label for each roi...

        // if (make_measurements) {

        // }

        float                       downsample_sqrt;
        std::vector<F32PolygonType> gg;
        for (int i = 0; i < rois_nuclei.size(); ++i) {
            auto roi_f = get_float_polygon(rois_nuclei[i]);
            if (smooth_boundaries) {
                auto p1           = get_interpolated_polygon(roi_f, 1, false, false);
                auto smoothed_roi = smooth_polygon_roi(p1);
                auto g            = get_interpolated_polygon(
                    smoothed_roi, FISH_MIN(2.0, smoothed_roi.size() * 0.1), false, false);

                // transform the point to original in image...
                if (smooth_boundaries) {
                    auto expected_roi =
                        shape_simplier::simplify_polygon_points_better(g, downsample_sqrt / 2);
                    gg.push_back(expected_roi);
                } else {
                    gg.push_back(g);
                }
            }
        }

        if (cell_expansion > 0) {
            LOG_INFO("apply cell expansion with distance {}", cell_expansion);
            ImageMat<float> fp_EDM;
            distance_transform<uint8_t>(roi_mask, fp_EDM, false, 255, 0);
            float* ptr_EDM = fp_EDM.get_data_ptr();
            for (int i = 0; i < height * width; ++i) {
                ptr_EDM[i] *= -1.0f;
            }
            double cell_expansion_threshold = -1.0 * cell_expansion;
            auto   label_cells              = detect_labels;

            watershed_process<float, uint16_t, NeighborConnectiveType::Conn4>(
                fp_EDM, label_cells, cell_expansion_threshold);
            std::vector<PolygonType> rois_cells =
                labels_to_filled_rois(label_cells, rois_nuclei.size());

            for (int i = 0; i < rois_cells.size(); ++i) {
                if (rois_cells.size() == 0) {
                    continue;
                }
                //这里可以优化和复用内存
                auto r1 = get_float_polygon(rois_cells[i]);
                auto r2 = get_interpolated_polygon(r1, 1, false, false);
                auto r3 = smooth_polygon_roi(r2);
                auto r4 =
                    get_interpolated_polygon(r3, FISH_MIN(2.0, r3.size() * 0.1), false, false);
                if (smooth_boundaries) {
                    auto r5 =
                        shape_simplier::simplify_polygon_points_better(r4, downsample_sqrt / 2.0);
                } else {
                    // push back r4
                }
            }
        }
    }
};

}   // namespace cell_detection
}   // namespace segmentation
}   // namespace fish