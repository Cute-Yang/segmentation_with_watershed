#pragma once
#include "common/fishdef.h"
#include "core/mat.h"
#include <array>
#include <cstdint>

namespace fish {
namespace image_proc {
namespace statistic {
using namespace fish::core::mat;
using histogram_t = std::array<int, 256>;
FISH_ALWAYS_INLINE histogram_t get_image_histogram(const ImageMat<uint8_t>& image) {
    int                  height    = image.get_height();
    int                  width     = image.get_width();
    const unsigned char* image_ptr = image.get_data_ptr();
    histogram_t          histogram;
    std::fill(histogram.begin(), histogram.end(), 0);
    for (int i = 0; i < height * width; ++i) {
        ++histogram[image_ptr[i]];
    }
    return histogram;
}



template<class T>
float compute_roi_mean(const ImageMat<T>& image, ImageMat<uint8_t>& mask, int x1, int y1) {
    int    mask_h = mask.get_height();
    int    mask_w = mask.get_width();
    double sum    = 0.0;
    int    count  = 0;
    for (int y = 0; y < mask_h; ++y) {
        for (int x = 0; x < mask_w; ++x) {
            if (mask(y, x) != 0) {
                sum += image(y + y1, x + x1);
                ++count;
            }
        }
    }
    if (count == 0) {
        return 0.0;
    }
    return sum / count;
}

struct StatResult {
    double sum;
    int    pixel_count;
};
template<class T>
StatResult compute_roi_stat(const ImageMat<T>& image, ImageMat<uint8_t>& mask, int x1, int y1) {
    int    mask_h = mask.get_height();
    int    mask_w = mask.get_width();
    double sum    = 0.0;
    int    count  = 0;
    for (int y = 0; y < mask_h; ++y) {
        for (int x = 0; x < mask_w; ++x) {
            if (mask(y, x) != 0) {
                sum += image(y + y1, x + x1);
                ++count;
            }
        }
    }
    StatResult stat;
    stat.sum         = sum;
    stat.pixel_count = count;
    return stat;
}
}   // namespace statistic
}   // namespace image_proc
}   // namespace fish