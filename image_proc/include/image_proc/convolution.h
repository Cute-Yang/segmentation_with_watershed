#pragma once

#include "core/base.h"
#include "core/mat.h"

namespace fish {
namespace image_proc {
namespace convolution {
using namespace fish::core::base;
using namespace fish::core::mat;
template<class T, typename = image_dtype_limit<T>>
Status::ErrorCode convolution_2d(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                                 const Mat<float>& kernel);

template<class T, typename = image_dtype_limit<T>>
Status::ErrorCode convolution_2d(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                                 float* conv_kernel, int kh, int kw);


}   // namespace convolution
}   // namespace image_proc
}   // namespace fish