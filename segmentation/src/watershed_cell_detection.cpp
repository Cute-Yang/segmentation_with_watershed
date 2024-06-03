#include "segmentation/watershed_cell_detection.h"
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "core/mat_ops.h"
#include "image_proc/convolution.h"
#include "image_proc/distance_transform.h"
#include "image_proc/fill_mask.h"
#include "image_proc/find_contour.h"
#include "image_proc/find_maximum.h"
#include "image_proc/guassian_blur.h"
#include "image_proc/img_stat.h"
#include "image_proc/neighbor_filter.h"
#include "image_proc/polygon.h"
#include "image_proc/rank_filter.h"
#include "image_proc/roi_labeling.h"
#include "segmentation/estimate_backgroud.h"
#include "segmentation/morphological_transform.h"
#include "segmentation/shape_simplifier.h"
#include "segmentation/watershed.h"
#include "utils/logging.h"
#include <asm-generic/errno.h>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace fish {
namespace segmentation {
namespace watershed_cell_detection {
using namespace fish::core;
using namespace fish::core::mat;
using namespace fish::image_proc::rank_filter;
using namespace fish::image_proc::guassian_blur;
using namespace fish::image_proc::convolution;
using namespace fish::image_proc::find_maximum;
using namespace fish::core::mat_ops;
using namespace fish::segmentation::estimate;
using namespace fish::segmentation::morphological;
using namespace fish::image_proc::roi_labeling;
using namespace fish::segmentation::watershed;
using namespace fish::image_proc::contour;
using namespace fish::image_proc::statistic;
using namespace fish::image_proc::fill_mask;
using namespace fish::image_proc::neighbor_filter;
using namespace fish::image_proc::distance_transform;
using namespace fish::segmentation::shape_simplier;
/**
 * @brief
 *
 * @param original_image the mat of image!
 * @param detect_channel,the channel you want to use
 * @param Hematoxylin_channel
 * @param DAB_channel
 * @param background_radius
 * @param max_background
 * @param median_radius
 * @param sigma
 * @param threshold
 * @param min_area
 * @param max_area
 * @param merge_all
 * @param watershed_postprocess
 * @param exclude_DAB
 * @param cell_expansion
 * @param smooth_boundaries
 * @param make_measurements
 * @param background_by_reconstruction
 * @param downsample_factor
 * @return void
 */
namespace internal {
// return the max of requested size and real size as perfered size!
FISH_ALWAYS_INLINE double compute_averaged_pixel_size_microns(double pixel_size_microns_h,
                                                              double pixel_size_microns_w) {
    return (pixel_size_microns_h + pixel_size_microns_w) * 0.5;
}

double compute_preferred_pixel_size_microns(double pixel_size_microns_h,
                                            double pixel_size_microns_w,
                                            double requested_pixel_size) {
    double averaged_pixel_size =
        compute_averaged_pixel_size_microns(pixel_size_microns_h, pixel_size_microns_w);
    if (requested_pixel_size < 0.0) {
        LOG_INFO("the given requested pixel size microns < 0.0 which is unexpected!so we will "
                 "multiply -1.0");
        requested_pixel_size = averaged_pixel_size * (-requested_pixel_size);
    }
    // use the max value as final prefered pixel size!
    requested_pixel_size = FISH_MAX(requested_pixel_size, averaged_pixel_size);
    return requested_pixel_size;
}

double compute_preferred_donwsample_macrons(double pixel_size_microns,
                                            double requested_pixel_size_microns, bool apply_log2) {
    double downsample_factor;
    if (apply_log2) {
        downsample_factor =
            std::pow(2.0,
                     std::round(std::log(requested_pixel_size_microns / pixel_size_microns) /
                                std::log(2.0)));
    } else {
        downsample_factor = requested_pixel_size_microns / pixel_size_microns;
    }
    return downsample_factor;
}

double compute_downsample_factor(double pixel_size_microns_h, double pixel_size_microns_w,
                                 double preferred_pixel_size_microns, bool apply_log2) {
    double pixel_size_microns =
        compute_averaged_pixel_size_microns(pixel_size_microns_h, pixel_size_microns_w);
    double downsample_factor = compute_preferred_donwsample_macrons(
        pixel_size_microns, preferred_pixel_size_microns, false);
    if (downsample_factor < 1.0) {
        downsample_factor = 1.0;
    }
    return downsample_factor;
}

template<class T1, class T2, typename = dtype_limit<T1>, typename = dtype_limit<T2>>
ImageMat<T2> convert_mat_dtype(const ImageMat<T1>& mat) {
    if constexpr (std::is_same_v<T1, T2>) {
        return mat;
    }
    int          height   = mat.get_height();
    int          width    = mat.get_width();
    int          channels = mat.get_channels();
    ImageMat<T2> converted_mat(height, width, channels, MatMemLayout::LayoutRight);
    T2*          converted_mat_ptr = converted_mat.get_data_ptr();
    const T1*    mat_ptr           = mat.get_data_ptr();
    size_t       data_size         = height * width * channels;
    for (int i = 0; i < data_size; ++i) {
        converted_mat_ptr[i] = static_cast<T2>(mat_ptr[i]);
    }
    return converted_mat;
}


Status::ErrorCode cell_detection_impl(
    const ImageMat<float>& original_image, int detect_channel, int Hematoxylin_channel,
    int DAB_channel, double background_radius, double max_background, double median_radius,
    double sigma, double threshold, double min_area, double max_area, double merge_all,
    bool watershed_postprocess, bool exclude_DAB, double cell_expansion, bool smooth_boundaries,
    bool make_measurements, bool background_by_reconstruction, bool refine_boundary,
    double downsample_factor, std::vector<PolygonTypef32>& out_nuclei_rois,
    std::vector<PolygonTypef32>& out_cell_rois) {
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
             threshold,
             min_area,
             max_area,
             cell_expansion,
             max_background,
             merge_all,
             watershed_postprocess,
             exclude_DAB,
             smooth_boundaries,
             make_measurements);
    if (original_image.empty()) {
        LOG_ERROR("the origianl image is empty,so noting to do....");
        return Status::ErrorCode::InvalidMatShape;
    }
    int height   = original_image.get_height();
    int width    = original_image.get_width();
    int channels = original_image.get_channels();
    if (detect_channel < 0 || detect_channel >= channels) {
        LOG_ERROR("out original image has channels {},but you speicfy channel {} to detecit,this "
                  "is invalid!",
                  channels,
                  detect_channel);
        return Status::ErrorCode::InvalidMatChannle;
    }
    LOG_INFO("we will use channel {} to detect...", detect_channel);
    // for the fill func,we pass the mask to reuse the memory!
    LOG_INFO("genreate two placeholder to reuse the memory...");
    ImageMat<uint8_t> image_u8_placeholder(height, width, 1, MatMemLayout::LayoutRight);
    ImageMat<float>   image_f32_placeholder(height, width, 1, MatMemLayout::LayoutRight);

    // for memory report!
    float image_f32_memory_size =
        static_cast<float>(height * width * sizeof(float)) / 1024.0f / 1024.0f;
    float image_u8_memory_size = static_cast<float>(height * width) / 1024.0f / 1024.0f;
    // the background mask need to allocate a buffer!
    ImageMat<uint8_t> background_mask;
    // the image to compute the measurements!
    ImageMat<float> measurement_image;

    int data_size = height * width;
    // apply copy from detect_image!
    LOG_INFO("copying the detect channel data to a new mat!");
    ImageMat<float> transform_image(height, width, 1, MatMemLayout::LayoutRight);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            transform_image(y, x) = original_image(y, x, detect_channel);
        }
    }

    Status::ErrorCode invoke_status;
    if (median_radius > 0) {
        LOG_INFO("apply median filter with radius {}", median_radius);
        // attention,here our func can invoke inplace!
        // add nodiscard to
        invoke_status =
            rank_filter(transform_image, transform_image, FilterType::MEDIAN, median_radius);
        if (invoke_status != Status::ErrorCode::Ok) {
            const char* error_msg = Status::get_error_msg(invoke_status);
            LOG_ERROR("apply rank filter fail,the error msg is {}", error_msg);
            return invoke_status;
        }
    }
    if (exclude_DAB) {
        bool Hematoxylin_valid = is_valid_channel(Hematoxylin_channel, channels);
        bool DAB_valid         = is_valid_channel(DAB_channel, channels);
        // pass by ref!
        auto& DAB_mask = image_u8_placeholder;
        if (Hematoxylin_valid && DAB_valid && Hematoxylin_channel != DAB_channel) {
            LOG_INFO("exclude the DAB...");
            constexpr uint8_t DAB_fill_value = 1;
            simple_threshold::greater_equal_than(
                original_image, DAB_mask, Hematoxylin_channel, DAB_channel, DAB_fill_value);
            constexpr double DAB_rank_radius = 2.5;
            rank_filter(DAB_mask, DAB_mask, FilterType::MEDIAN, DAB_rank_radius);
            rank_filter(DAB_mask, DAB_mask, FilterType::MAX, DAB_rank_radius);
            // if the mask == 0,set the pixel value to zero!
            uint8_t* DAB_mask_ptr        = DAB_mask.get_data_ptr();
            float*   transform_image_ptr = transform_image.get_data_ptr();
            // make sure all of our data have same layout!
            // while use the pointer will be a little faster than access with our index...
            for (int i = 0; i < data_size; ++i) {
                if (DAB_mask_ptr[i] == 0) {
                    transform_image_ptr[i] = 0.0f;
                }
            }
        }
    }

    // allocate memory for measurement image!while measurement_image can not use
    // image_f32_placeholder!
    LOG_INFO("allocate memory {}Mb for measurments image,also for filter...",
             image_f32_memory_size);
    measurement_image.resize(height, width, 1, true);

    if (background_radius > 0) {
        auto&  background_image     = image_f32_placeholder;
        float* background_image_ptr = background_image.get_data_ptr();
        float* transform_image_ptr  = transform_image.get_data_ptr();
        // copy the transfomr image value to background...
        for (int i = 0; i < data_size; ++i) {
            background_image_ptr[i] = transform_image_ptr[i];
        }

        // allocate memory for background mask...,while this can not use the image_u8_placeholder!
        estimate_background(transform_image,
                            background_image,
                            background_mask,
                            background_radius,
                            max_background,
                            background_by_reconstruction);
        copy_image_mat(background_image, transform_image, ValueOpKind::SUBSTRACT);

        LOG_INFO("using the image after background estimate as the image to make measurements!");
        float* measurement_image_ptr = measurement_image.get_data_ptr();
        // now the buffer
        for (int i = 0; i < data_size; ++i) {
            measurement_image_ptr[i] = transform_image_ptr[i];
        }
    } else {
        // we can copy the original buffer to temp image buffer!
        //  use the original image to make measuremetns..
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                measurement_image(y, x) = original_image(y, x);
            }
        }
    }

    // first step,generate the rois...
    auto& guassian_blur_result = image_f32_placeholder;
    invoke_status              = guassian_blur_2d(transform_image, guassian_blur_result, sigma);
    if (invoke_status != Status::ErrorCode::Ok) {
        LOG_ERROR("apply guassian transform occur error {}", Status::get_error_msg(invoke_status));
        return invoke_status;
    }

    // apply conv!
    constexpr int conv_kh                               = 3;
    constexpr int conv_kw                               = 3;
    float         convolution_kernel[conv_kh * conv_kw] = {
                0.0f, -1.0f, 0.0f, -1.0f, 4.0f, -1.0f, 0.0f, -1.0f, 0.0f};
    // now need to allocate a buffer to restore the conv result1;
    invoke_status =
        convolution_2d(guassian_blur_result, transform_image, convolution_kernel, conv_kh, conv_kw);
    if (invoke_status != Status::ErrorCode::Ok) {
        LOG_INFO("apply convolution failed...");
        return invoke_status;
    }

    LOG_INFO("apply binarize with thresold 0.0f");
    ImageMat<uint8_t> transform_image_mask(height, width, 1, MatMemLayout::LayoutRight);
    // compare a matrix with scalr
    threshold_above(transform_image, transform_image_mask, 0.0f);

    // reuse memory
    LOG_INFO("binding image placeholder to morphological image...");
    ImageMat<float>& morphological_image = image_f32_placeholder;

    LOG_INFO("apply morphological transform...");
    find_regional_maxima(transform_image, morphological_image, 0.001f);

    LOG_INFO("compute the image label....");

    // support 2^32 -1 polygons...
    using image_label_t = uint32_t;
    ImageMat<image_label_t> label_image(height, width, 1, MatMemLayout::LayoutRight);
    invoke_status = compute_image_label(morphological_image, label_image, 0.0f, false);
    if (invoke_status != Status::ErrorCode::Ok) {
        LOG_ERROR("fail to compute image label...");
        return invoke_status;
    }

    LOG_INFO("apply watershed transform...");
    invoke_status = watershed_transform(transform_image, label_image, 0.0f, false);
    if (invoke_status != Status::ErrorCode::Ok) {
        LOG_ERROR("fail to apply watershed transform...");
        return invoke_status;
    }

    // generate the rois...
    std::vector<PolygonType> rois;
    std::vector<PolyMask>    roi_masks;

    // in qupath,they set min value to 0.5,but image type is short,will apply int(0.5 + 0.5) -> 1.0
    constexpr image_label_t lower_thresh  = 1;
    constexpr image_label_t higher_thresh = std::numeric_limits<image_label_t>::max();
    LOG_INFO("find filled rois...");
    {
        LOG_INFO("binding image_u8_palceholder to temp_fill_mask...");
        auto& temp_fill_mask = image_u8_placeholder;
        invoke_status        = get_filled_polygon(label_image,
                                           temp_fill_mask,
                                           WandMode::FOUR_CONNECTED,
                                           rois,
                                           roi_masks,
                                           lower_thresh,
                                           higher_thresh,
                                           false);
        if (invoke_status != Status::ErrorCode::Ok) {
            LOG_INFO("fill poly fail...");
            return invoke_status;
        }
        LOG_INFO("find {} polygon...", rois.size());
    }

    // filter the rois by mean value,fill the image and use this image to apply transform!
    ImageMat<uint8_t> filled_image(height, width, 1, MatMemLayout::LayoutRight);
    filled_image.set_zero();
    PolygonFiller poly_filler;
    if (background_mask.empty()) {
        for (size_t i = 0; i < rois.size(); ++i) {
            auto& roi       = rois[i];
            auto& roi_mask  = roi_masks[i].mask;
            int   x1        = roi_masks[i].x1;
            int   y1        = roi_masks[i].y1;
            int   rh        = roi_mask.get_height();
            int   rw        = roi_mask.get_width();
            float poly_mean = compute_roi_mean(measurement_image, roi_mask, x1, y1);
            if (poly_mean <= threshold) {
                continue;
            }
            fill_image_with_mask<uint8_t>(filled_image, roi_mask, x1, y1, 255);
        }
    } else {
        // also check the background!
        for (size_t i = 0; i < rois.size(); ++i) {
            auto& roi       = rois[i];
            auto& roi_mask  = roi_masks[i].mask;
            int   x1        = roi_masks[i].x1;
            int   y1        = roi_masks[i].y1;
            int   rh        = roi_mask.get_height();
            int   rw        = roi_mask.get_width();
            float poly_mean = compute_roi_mean(measurement_image, roi_mask, x1, y1);
            if (poly_mean <= threshold) {
                continue;
            }
            float background_poly_mean = compute_roi_mean(background_mask, roi_mask, x1, y1);
            if (background_poly_mean > 0) {
                continue;
            }
            fill_image_with_mask<uint8_t>(filled_image, roi_mask, x1, y1, 255);
        }
    }

    if (merge_all) {
        LOG_INFO("binding image_u8_palceholder to neigh_filter_3x3_result...");
        ImageMat<uint8_t>& neigh_filter_3x3_result = image_u8_placeholder;
        bool               pad_edges               = true;
        int                binary_count            = 3;
        neighbor_filter_with_3x3_window(filled_image,
                                        neigh_filter_3x3_result,
                                        NeighborFilterType::MAX,
                                        pad_edges,
                                        binary_count);
        LOG_INFO("swap the neight_filter_3x3_result to filled_image...");
        filled_image.swap(neigh_filter_3x3_result);
        copy_image_mat(transform_image_mask, filled_image, ValueOpKind::AND);
        if (watershed_postprocess) {
            std::vector<PolygonType> postprocess_rois;
            std::vector<PolyMask>    postprocess_roi_masks;
            constexpr uint8_t        lower_thresh   = 127;
            constexpr uint8_t        higher_thresh  = 255;
            ImageMat<uint8_t>&       temp_fill_mask = image_u8_placeholder;
            invoke_status                           = get_filled_polygon(filled_image,
                                               temp_fill_mask,
                                               WandMode::FOUR_CONNECTED,
                                               postprocess_rois,
                                               postprocess_roi_masks,
                                               lower_thresh,
                                               higher_thresh,
                                               true);
            if (invoke_status != Status::ErrorCode::Ok) {
                LOG_ERROR("invoke fill polygon fail");
                return invoke_status;
            }
            for (size_t i = 0; i < postprocess_roi_masks.size(); ++i) {
                auto& poly_mask = postprocess_roi_masks[i].mask;
                // got the left upper coor of the poly!
                int x1 = postprocess_roi_masks[i].x1;
                int y1 = postprocess_roi_masks[i].y1;
                int rh = poly_mask.get_height();
                int rw = poly_mask.get_width();
                fill_image_with_mask<uint8_t>(filled_image, poly_mask, x1, y1, 255);
            }

            {
                LOG_INFO("binding image_f32_placeholder to distance image...");
                ImageMat<float>& distance_image = image_f32_placeholder;
                distance_transform<uint8_t>(filled_image, distance_image, false, 0);
                ImageMat<uint8_t>  distance_mask;   // if empty,we think all value is valid!
                constexpr float    MAXFINDER_TOLERANCE = 0.5f;
                ImageMat<uint8_t>& maximum_mask        = image_u8_placeholder;
                bool               strict              = false;
                bool               exclude_on_edges    = false;
                bool               is_EDM              = true;
                invoke_status                          = find_maxima(distance_image,
                                            distance_mask,
                                            maximum_mask,
                                            strict,
                                            MAXFINDER_TOLERANCE,
                                            NO_THRESHOLD,
                                            EDMOutputType::SEGMENTED,
                                            exclude_on_edges,
                                            true);
                if (invoke_status != Status::ErrorCode::Ok) {
                    LOG_ERROR("find maximum mask fail....");
                    return invoke_status;
                }
                copy_image_mat(maximum_mask, filled_image, ValueOpKind::AND);
            }
        }
    }
    // if only the part of image
    if (refine_boundary && sigma > 1.5) {
        LOG_INFO("refine boundary....");
        // copy the original image value to transform image
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                transform_image(y, x) = original_image(y, x, detect_channel);
            }
        }
        LOG_INFO("binding the image_f32_placeholder to guassian_blur_result");
        ImageMat<float>& guassian_blur_result = image_f32_placeholder;
        constexpr float  refine_sigma         = 1.0f;
        invoke_status = guassian_blur_2d(transform_image, guassian_blur_result, refine_sigma);
        if (invoke_status != Status::ErrorCode::Ok) {
            LOG_ERROR("refine guassian blur fail...");
            return invoke_status;
        }
        // swap the data to transform image...
        // conv
        int   conv_kh        = 3;
        int   conv_kw        = 3;
        float conv_kernel[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
        invoke_status =
            convolution_2d(guassian_blur_result, transform_image, conv_kernel, conv_kh, conv_kw);
        if (invoke_status != Status::ErrorCode::Ok) {
            LOG_ERROR("refine conv_kernel blur fail...");
            return invoke_status;
        }

        // here we can reuse the transform image mask!
        threshold_above(transform_image, transform_image_mask, 0.0f);
        copy_image_mat(filled_image, transform_image_mask, ValueOpKind::MIN);

        // apply 3x3 filter!
        auto& neigh_filter_3x3_result = image_u8_placeholder;
        int   binary_count            = 0;
        bool  pad_edges               = false;
        neighbor_filter_with_3x3_window(filled_image,
                                        neigh_filter_3x3_result,
                                        NeighborFilterType::MIN,
                                        pad_edges,
                                        binary_count);
        filled_image.swap(neigh_filter_3x3_result);
        copy_image_mat(transform_image_mask, filled_image, ValueOpKind::MAX);
    }

    std::vector<PolygonType> nuclei_rois;
    std::vector<PolyMask>    nuclei_roi_masks;
    std::vector<uint8_t>     nuclei_roi_keep_flags(nuclei_rois.size(), 1);
    int                      remove_roi_size = 0;
    {
        LOG_INFO("binding image_u8_placeholder to temp_fill_mask...");
        ImageMat<uint8_t>& temp_fill_mask = image_u8_placeholder;
        constexpr uint8_t  thresh_lower   = 127;
        constexpr uint8_t  thresh_higher  = 255;
        get_filled_polygon(filled_image,
                           temp_fill_mask,
                           WandMode::FOUR_CONNECTED,
                           nuclei_rois,
                           nuclei_roi_masks,
                           thresh_lower,
                           thresh_higher,
                           false);
        LOG_INFO("find {} polygon....", nuclei_rois.size());
    }
    if (min_area > 0.0 || max_area > 0.0) {
        constexpr uint8_t current_fill_value = 0;
        for (size_t i = 0; i < nuclei_roi_masks.size(); ++i) {
            auto&      poly_mask = nuclei_roi_masks[i].mask;
            int        x1        = nuclei_roi_masks[i].x1;
            int        y1        = nuclei_roi_masks[i].y1;
            StatResult poly_stat = compute_roi_stat(measurement_image, poly_mask, x1, y1);
            double     poly_area = poly_stat.pixel_count;
            double     poly_mean = (poly_stat.sum) / (poly_area + 1e-7);
            if (poly_area < threshold || (min_area > 0.0 && poly_area < min_area) ||
                (max_area > 0.0 && poly_area > max_area)) {
                // fill it to zero!
                fill_image_with_mask(filled_image, poly_mask, x1, y1, current_fill_value);
                nuclei_roi_keep_flags[i] = 0;
                ++remove_roi_size;
            }
        }
        if (remove_roi_size > 0) {
            LOG_INFO("after area filter,we remove {} rois...", remove_roi_size);
            std::vector<PolygonType> temp_nuclei_rois;
            temp_nuclei_rois.reserve(nuclei_rois.size() - remove_roi_size);
            std::vector<PolyMask> temp_nuclei_roi_masks;
            temp_nuclei_roi_masks.reserve(nuclei_rois.size() - remove_roi_size);
            for (size_t i = 0; i < nuclei_rois.size(); ++i) {
                if (nuclei_roi_keep_flags[i] != 0) {
                    temp_nuclei_rois.push_back(std::move(nuclei_rois[i]));
                    temp_nuclei_roi_masks.push_back(std::move(temp_nuclei_roi_masks[i]));
                }
            }
            // then swap the buffer
            nuclei_rois.swap(temp_nuclei_rois);
            nuclei_roi_masks.swap(temp_nuclei_roi_masks);
        }
    }

    // rois label to image...
    // only when make measurements...
    label_image.set_zero();
    for (size_t i = 0; i < nuclei_rois.size(); ++i) {
        int   fill_value = i + 1;
        auto& poly_mask  = nuclei_roi_masks[i].mask;
        int   x1         = nuclei_roi_masks[i].x1;
        int   y1         = nuclei_roi_masks[i].y1;
        fill_image_with_mask<uint32_t>(label_image, poly_mask, x1, y1, fill_value);
    }
    double downsample_sqrt = std::sqrt(downsample_factor);

    std::vector<PolygonTypef32> smoothed_nuclei_rois;
    smoothed_nuclei_rois.reserve(nuclei_rois.size());
    LOG_INFO("apply smooth for nuclei_rois...");
    if (smooth_boundaries) {
        for (size_t i = 0; i < nuclei_rois.size(); ++i) {
            PolygonType&   roi     = nuclei_rois[i];
            PolygonTypef32 roi_f32 = convert_polygon_to_float(roi);

            constexpr float interval_s1 = 1.0f;
            PolygonTypef32  roi_f32_s1 =
                get_interpolated_polygon(roi_f32, interval_s1, false, RoiType::POLYGON);
            PolygonTypef32 roi_f32_s2  = smooth_polygon_roi(roi_f32_s1);
            float          interval_s3 = FISH_MIN(2.0, roi_f32_s2.size());
            PolygonTypef32 roi_f32_s3 =
                get_interpolated_polygon(roi_f32_s2, interval_s3, false, RoiType::POLYGON);
            PolygonTypef32 roi_f32_s4 =
                simplify_polygon_points_better(roi_f32_s3, downsample_sqrt / 2.0);
            smoothed_nuclei_rois.push_back(std::move(roi_f32_s4));
        }
    } else {
        // just convert the
        for (size_t i = 0; i < nuclei_rois.size(); ++i) {
            PolygonType&   roi     = nuclei_rois[i];
            PolygonTypef32 roi_f32 = convert_polygon_to_float(roi);
            smoothed_nuclei_rois.push_back(std::move(roi_f32));
        }
    }
    out_nuclei_rois.swap(smoothed_nuclei_rois);

    // firstly,compute the nuclei! if not apply expansion,just use nuclei
    if (cell_expansion > 0.0) {
        LOG_INFO("apply cell expansion!");
        ImageMat<float>& distance_image = image_f32_placeholder;
        distance_transform<uint8_t>(filled_image, distance_image, false, 255);
        float* distance_image_ptr = distance_image.get_data_ptr();
        for (int i = 0; i < data_size; ++i) {
            distance_image_ptr[i] *= 1.0f;
        }
        double             cell_expansion_threshold = -1.0 * cell_expansion;
        ImageMat<uint32_t> cell_label_image         = label_image;
        watershed_transform<float, uint32_t>(
            distance_image, cell_label_image, cell_expansion_threshold, false);
        ImageMat<uint8_t>&       temp_fill_mask = image_u8_placeholder;
        std::vector<PolygonType> cell_rois;
        std::vector<PolyMask>    cell_roi_masks;
        labels_to_filled_polygon(
            cell_label_image, temp_fill_mask, nuclei_rois.size(), cell_rois, cell_roi_masks, true);

        std::vector<PolygonTypef32> smoothed_cell_rois;
        smoothed_cell_rois.reserve(cell_rois.size());
        if (smooth_boundaries) {
            for (size_t i = 0; i < cell_rois.size(); ++i) {
                PolygonType&   roi     = cell_rois[i];
                PolygonTypef32 roi_f32 = convert_polygon_to_float(roi);
                if (smooth_boundaries) {
                    constexpr float interval_s1 = 1.0f;
                    PolygonTypef32  roi_f32_s1 =
                        get_interpolated_polygon(roi_f32, interval_s1, false, RoiType::POLYGON);
                    PolygonTypef32 roi_f32_s2  = smooth_polygon_roi(roi_f32_s1);
                    float          interval_s3 = FISH_MIN(2.0, roi_f32_s2.size());
                    PolygonTypef32 roi_f32_s3 =
                        get_interpolated_polygon(roi_f32_s2, interval_s3, false, RoiType::POLYGON);
                    PolygonTypef32 roi_f32_s4 =
                        simplify_polygon_points_better(roi_f32_s3, downsample_sqrt / 2.0);
                    smoothed_cell_rois.push_back(std::move(roi_f32_s4));
                }
            }
        } else {
            // also convert to float!
            for (size_t i = 0; i < cell_rois.size(); ++i) {
                PolygonType&   roi     = cell_rois[i];
                PolygonTypef32 roi_f32 = convert_polygon_to_float(roi);
                smoothed_cell_rois.push_back(std::move(roi_f32));
            }
        }
        out_cell_rois.swap(smoothed_cell_rois);
    }
    return Status::ErrorCode::Ok;
}
}   // namespace internal


bool WatershedCellDetector::cell_detection(const ImageMat<float>& original_image,
                                           int detect_channel, int Hematoxylin_channel,
                                           int DAB_channel) {
    LOG_INFO("apply cell detection without pixel size microns,so we will set the downsample factor "
             "to 1.0 as default...");
    double downsample;
    if (have_pixel_size_microns) {
        LOG_INFO("compute the downsample factor wiht pixel size macrons");
        double preferred_pixel_size_microns = internal::compute_preferred_pixel_size_microns(
            pixel_size_microns_h, pixel_size_microns_w, requested_pixel_size);
        downsample = internal::compute_downsample_factor(
            pixel_size_microns_h, pixel_size_microns_w, preferred_pixel_size_microns, false);
    } else {
        // if do not have the macrons,just use 1.0 as the downsample...
        downsample = 1.0;
    }
    LOG_INFO("the downsample factor is {}", downsample);
    Status::ErrorCode status;
    if (!have_pixel_size_microns) {
        status = internal::cell_detection_impl(original_image,
                                               detect_channel,
                                               Hematoxylin_channel,
                                               DAB_channel,
                                               background_radius,
                                               max_background,
                                               median_radius,
                                               sigma,
                                               threshold,
                                               min_area,
                                               max_area,
                                               merge_all,
                                               watershed_postprocess,
                                               exclude_DAB,
                                               cell_expansion,
                                               smooth_boundaries,
                                               make_measurements,
                                               background_by_reconstruction,
                                               refine_boundary,
                                               downsample,
                                               nuclei_rois,
                                               cell_rois);
    } else {
        // transform some datas!
        double pixel_size_microns = internal::compute_averaged_pixel_size_microns(
            pixel_size_microns_h, pixel_size_microns_w);
        LOG_INFO("transform the image proc parmas by divide pixe size microns {}",
                 pixel_size_microns);
        // the radius param...
        double new_background_radius = background_radius / pixel_size_microns;
        double new_median_radius     = median_radius / pixel_size_microns;
        double new_sigma             = sigma / pixel_size_microns;
        double new_min_area          = min_area / (pixel_size_microns * pixel_size_microns);
        double new_max_area          = max_area / (pixel_size_microns * pixel_size_microns);
        double new_cell_expansion    = cell_expansion / pixel_size_microns;

        status = internal::cell_detection_impl(original_image,
                                               detect_channel,
                                               Hematoxylin_channel,
                                               DAB_channel,
                                               new_background_radius,
                                               max_background,
                                               new_median_radius,
                                               new_sigma,
                                               threshold,
                                               new_min_area,
                                               new_max_area,
                                               merge_all,
                                               watershed_postprocess,
                                               exclude_DAB,
                                               new_cell_expansion,
                                               smooth_boundaries,
                                               make_measurements,
                                               background_by_reconstruction,
                                               refine_boundary,
                                               downsample,
                                               nuclei_rois,
                                               cell_rois);
    }
    return (status == Status::ErrorCode::Ok);
}

bool WatershedCellDetector::cell_detection(const ImageMat<float>& original_image,
                                           int                    detect_channel) {
    if (exclude_DAB) {
        LOG_ERROR("can not exclude DAB because you did not specify Hematoxylin channel and DAB "
                  "channel...");
        return false;
    }
    constexpr int Hematoxylin_channel = -1;
    constexpr int DAB_channel         = -1;
    bool ret = cell_detection(original_image, detect_channel, Hematoxylin_channel, DAB_channel);
    return ret;
}

bool WatershedCellDetector::cell_detection(const ImageMat<uint8_t>& original_image,
                                           int detect_channel, int Hematoxylin_channel,
                                           int DAB_channel) {
    ImageMat<float> original_image_f32 =
        internal::convert_mat_dtype<uint8_t, float>(original_image);
    bool ret = cell_detection(original_image_f32, detect_channel, Hematoxylin_channel, DAB_channel);
    return ret;
}

bool WatershedCellDetector::cell_detection(const ImageMat<uint8_t>& original_image,
                                           int                      detect_channel) {
    if (exclude_DAB) {
        LOG_ERROR(
            "can not exclude DAB because you did not specify Hematoxylin channel and DAB channel");
        return false;
    }
    constexpr int Hematoxylin_channel = -1;
    constexpr int DAB_channel         = -1;
    bool ret = cell_detection(original_image, detect_channel, Hematoxylin_channel, DAB_channel);
    return ret;
}

}   // namespace watershed_cell_detection
}   // namespace segmentation
}   // namespace fish