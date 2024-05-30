#pragma once
#include "core/base.h"
#include "core/mat.h"

namespace fish {
namespace segmentation {
namespace estimate {
using namespace fish::core::mat;

// if return a empty mat,means that invoke failed...
ImageMat<uint8_t> estimate_background(const ImageMat<float>& image,
                                      ImageMat<float>& background_image, double radius,
                                      double max_background, bool opening_by_reconstruct);

// if success,return Ok!
Status::ErrorCode estimate_background(const ImageMat<float>& image,
                                      ImageMat<float>&       background_image,
                                      ImageMat<uint8_t>& background_mask, double radius,
                                      double max_background, bool opening_by_reconstruct);



}   // namespace estimate
}   // namespace segmentation
}   // namespace fish