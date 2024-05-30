#pragma once
#include "core/base.h"
#include "core/mat.h"

namespace fish {
namespace image_proc {
namespace distance_transform {
using namespace fish::core::mat;
using namespace fish::core::base;
/**
 * @brief
 *
 * @tparam T
 * @param input_mat
 * @param output_mat
 * @param treat_edge_as_background
 * @return Status::ErrorCode
 */
template<class T, typename = image_dtype_limit<T>>
Status::ErrorCode distance_transform(const ImageMat<T>& input_mat, ImageMat<float>& output_mat,
                                     bool treat_edge_as_background, T background_value);


}   // namespace distance_transform
}   // namespace image_proc
}   // namespace fish