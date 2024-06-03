#include "core/base.h"
#include "core/mat.h"
#include "image_proc/rank_filter.h"
#include "segmentation/estimate_backgroud.h"
#include "segmentation/morphological_transform.h"
#include "utils/logging.h"
#include <cmath>
#include <limits>

namespace fish {
namespace segmentation {
namespace estimate {
using namespace fish::image_proc::rank_filter;
using namespace fish::segmentation::morphological;
Status::ErrorCode estimate_background(const ImageMat<float>& image,
                                      ImageMat<float>&       background_image,
                                      ImageMat<uint8_t>& background_mask, double rank_radius,
                                      double max_background, bool opening_by_reconstruct) {
    // the background image is a output param,should have the same shape with input_image
    int height   = image.get_height();
    int width    = image.get_width();
    int channels = image.get_channels();
    if (channels != 1) {
        LOG_ERROR("the transform image must have channel 1,but got channels {}", channels);
        return Status::ErrorCode::InvalidMatChannle;
    }
    if (!image.compare_shape(background_image)) {
        LOG_ERROR("the image and estimate background image should have same shape,but mismatch!");
        return Status::ErrorCode::MatShapeMismatch;
    }
    // do the rank filter for image
    rank_filter(background_image, background_image, FilterType::MIN, rank_radius);

    uint8_t* mask_ptr                = background_mask.get_data_ptr();
    size_t   background_mask_cnt     = 0;
    bool     is_max_background_valid = (!std::isinf(max_background) && max_background > 0);
    if (is_max_background_valid) {
        background_mask.resize(height, width, 1, true);
        background_mask.set_zero();
        LOG_INFO("the max background is {},we will apply estimate background transform!",
                 max_background);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // here we only handler the first channel
                if (background_image(y, x) > max_background) {
                    ++background_mask_cnt;
                    background_mask(y, x) = 1;
                }
            }
        }
        if (background_mask_cnt > 0) {
            LOG_INFO("apply the rank filter to backgound_mask...");
            constexpr float negative_inf_value = std::numeric_limits<float>::lowest();
            rank_filter(background_mask, background_mask, FilterType::MAX, rank_radius * 2);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    if (background_mask(y, x) != 0) {
                        background_image(y, x) = negative_inf_value;
                    }
                }
            }
        } else {
            // free the memory of mask
            background_mask.release_mat();
        }
    } else {
        LOG_INFO("the max background is {} which is invalid,so we will not apply the background "
                 "estimate...",
                 max_background);
    }
    if (opening_by_reconstruct) {
        morphological_transform(background_image, image);
    } else {
        rank_filter(background_image, background_image, FilterType::MAX, rank_radius);
    }
    return Status::Ok;
}


ImageMat<uint8_t> estimate_background(const ImageMat<float>& image,
                                      ImageMat<float>& background_image, double rank_radius,
                                      double max_background, bool open_by_reconstruct) {
    ImageMat<uint8_t> background_mask;
    Status::ErrorCode status = estimate_background(
        image, background_image, background_mask, rank_radius, max_background, open_by_reconstruct);
    if (status != Status::ErrorCode::Ok) {
        LOG_ERROR("estimate error...");
    }
    return background_mask;
}
}   // namespace estimate
}   // namespace segmentation
}   // namespace fish