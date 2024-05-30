#pragma once
#include "core/mat.h"
namespace fish {
namespace segmentation {
namespace morphological {
using namespace fish::core;
using namespace fish::core::mat;
Status::ErrorCode morphological_transform(ImageMat<float>&       image_marker,
                                          const ImageMat<float>& image_mask);
// you can speicfy a buffer to reuse the memory!
void find_regional_maxima(const ImageMat<float>& image, ImageMat<float>& marked_maximum_image,
                          float threshold);

ImageMat<float> find_regional_maxima(const ImageMat<float>& image, float threshold);


void find_regional_maxima_and_binarize(const ImageMat<float>& image,
                                       ImageMat<float>&       marked_maximum_image,
                                       ImageMat<uint8_t>& mask, float threshold);

void find_regional_maxima_and_binarize(const ImageMat<float>& image, ImageMat<uint8_t>& mask,
                                       float threshold);

ImageMat<uint8_t> find_regional_maxima_and_binarize(const ImageMat<float>& image, float threshold);

}   // namespace morphological
}   // namespace segmentation
}   // namespace fish