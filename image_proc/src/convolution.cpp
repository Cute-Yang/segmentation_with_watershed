
#include "image_proc/convolution.h"
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "utils/logging.h"

using namespace fish::core;
namespace fish {
namespace image_proc {
namespace convolution {

namespace internal {
FISH_ALWAYS_INLINE void normalize_kernel(float* kernel, int kernel_size) {
    LOG_INFO("normalize the kernel value by divied the sum of kernel,exclude the zero value...");
    double kernel_sum = 0.0;
    for (int i = 0; i < kernel_size; ++i) {
        kernel_sum += kernel[i];
    }

    if (kernel_sum == 0) {
        LOG_WARN("the kernel sum got value zero....");
        return;
    }

    for (int i = 0; i < kernel_size; ++i) {
        kernel[i] /= kernel_sum;
    }
}

FISH_ALWAYS_INLINE Mat<float> normalize_kernel(const Mat<float>& kernel) {
    Mat<float> scaled_kernel = kernel;
    normalize_kernel(scaled_kernel.get_data_ptr(), kernel.get_element_num());
    return scaled_kernel;
}
template<class T, typename = dtype_limit<T>>
void convolution_generic_impl(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                              const Mat<float>& kernel, int channel) {
    int height   = input_mat.get_height();
    int width    = input_mat.get_width();
    int channels = input_mat.get_channels();

    int kh = kernel.get_rows();
    int kw = kernel.get_cols();
    LOG_INFO("apply convolution with kh = {},kw = {}", kh, kw);
    int h_pad = kh / 2;
    int w_pad = kw / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float result = 0.0f;
            for (int h_offset = -h_pad; h_offset <= h_pad; ++h_offset) {
                for (int w_offset = -w_pad; w_offset <= w_pad; ++w_offset) {
                    int d0  = (y + h_offset) < 0         ? 0
                              : (y + h_offset) >= height ? height - 1
                                                         : (y + h_offset);
                    int d1  = (x + w_offset) < 0        ? 0
                              : (x + w_offset) >= width ? width - 1
                                                        : (x + w_offset);
                    int kd0 = w_offset + w_pad;
                    int kd1 = h_offset + h_pad;
                    result += static_cast<float>(input_mat(d0, d1, channel)) * kernel(kd0, kd1);
                }
            }
            output_mat.set_value_f(y, x, channel, result);
        }
    }
}

template<class T, typename = dtype_limit<T>>
void convolution_3x3_impl(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                          const Mat<float>& kernel, int channel) {
    int height   = input_mat.get_height();
    int width    = input_mat.get_width();
    int channels = input_mat.get_channels();

    if (height == 1 && width == 1) [[unlikely]] {
        LOG_INFO("the height and width == 1....");
        output_mat.set_value_f(
            0, 0, channel, static_cast<float>(input_mat(0, 0, channel)) * kernel(0, 0));
        return;
    }

    // myabe all in register!
    float w_00 = kernel(0, 0);
    float w_01 = kernel(0, 1);
    float w_02 = kernel(0, 2);
    float w_10 = kernel(1, 0);
    float w_11 = kernel(1, 1);
    float w_12 = kernel(1, 2);
    float w_20 = kernel(2, 0);
    float w_21 = kernel(2, 1);
    float w_22 = kernel(2, 2);

    if (height == 1) [[unlikely]] {
        LOG_INFO("apply conv with an row vector....");
        // specify for first element...
        float v_10, v_11, v_12;
        v_11 = static_cast<float>(input_mat(0, 0, channel));
        v_12 = static_cast<float>(input_mat(0, 1, channel));
        float result =
            v_11 * (w_00 + w_01 + w_10 + w_11 + w_20 + w_21) + v_12 * (w_02 + w_12 + w_22);
        output_mat.set_value_f(0, 0, channel, result);

        for (int i = 1; i < width - 1; ++i) {
            v_10 = static_cast<float>(input_mat(0, i - 1, channel));
            v_11 = static_cast<float>(input_mat(0, i, channel));
            v_12 = static_cast<float>(input_mat(0, i + 1, channel));
            // repeat along row
            result = v_10 * (w_00 + w_10 + w_20) + v_11 * (w_01 + w_11 + w_21) +
                     v_12 * (w_02 + w_12 + w_22);
            output_mat.set_value_f(0, i, channel, result);
        }

        v_10   = static_cast<float>(input_mat(0, width - 2, channel));
        v_11   = static_cast<float>(input_mat(0, width - 1, channel));
        result = v_10 * (w_00 + w_10 + w_20) + v_11 * (w_01 + w_02 + w_11 + w_12 + w_21 + w_22);
        output_mat.set_value_f(0, width - 1, channel, result);
        return;
    }

    if (width == 1) [[unlikely]] {
        float v_01, v_11, v_21;
        v_11 = static_cast<float>(input_mat(0, 0, channel));
        v_21 = static_cast<float>(input_mat(1, 0, channel));
        float result =
            v_11 * (w_00 + w_01 + w_02 + w_10 + w_11 + w_12) + v_21 * (w_20 + w_21 + w_22);
        output_mat.set_value_f(0, 0, channel, result);
        for (int i = 0; i < height - 1; ++i) {
            v_01 = static_cast<float>(input_mat(i - 1, 0, channel));
            v_11 = static_cast<float>(input_mat(i, 0, channel));
            v_21 = static_cast<float>(input_mat(i + 1, 0, channel));
            // repeat along column
            result = v_01 * (w_00 + w_01 + w_02) + v_11 * (w_10 + w_11 + w_12) +
                     v_21 * (w_20 + w_21 + w_22);
            output_mat.set_value_f(i, 0, channel, result);
        }
        v_01   = static_cast<float>(input_mat(height - 2, 0, channel));
        v_11   = static_cast<float>(input_mat(height - 1, 0, channel));
        result = v_01 * (w_00 + w_01 + w_02) + v_11 * (w_10 + w_11 + w_12 + w_20 + w_21 + w_22);
        output_mat.set_value_f(height - 1, 0, channel, result);
        return;
    }

    // first row
    float v_00, v_01, v_02, v_10, v_11, v_12, v_20, v_21, v_22;

    v_11 = static_cast<float>(input_mat(0, 0, channel));
    v_12 = static_cast<float>(input_mat(0, 1, channel));

    v_21 = static_cast<float>(input_mat(1, 0, channel));
    v_22 = static_cast<float>(input_mat(1, 1, channel));

    float result = v_11 * (w_00 + w_01 + w_10 + w_11) + v_12 * (w_02 + w_12) +
                   v_21 * (w_20 + w_21) + v_22 * w_22;
    output_mat.set_value_f(0, 0, channel, result);

    for (int i = 1; i < width - 1; ++i) {
        v_10 = static_cast<float>(input_mat(0, i - 1, channel));
        v_11 = static_cast<float>(input_mat(0, i, channel));
        v_12 = static_cast<float>(input_mat(0, i + 1, channel));

        v_20 = static_cast<float>(input_mat(1, i - 1, channel));
        v_21 = static_cast<float>(input_mat(1, i, channel));
        v_22 = static_cast<float>(input_mat(1, i + 1, channel));

        result = v_10 * (w_00 + w_10) + v_11 * (w_01 + w_11) + v_12 * (w_02 + w_12) + w_20 * v_20 +
                 v_21 * w_21 + v_22 * w_22;
        output_mat.set_value_f(0, i, channel, result);
    }

    v_10 = static_cast<float>(input_mat(0, width - 2, channel));
    v_11 = static_cast<float>(input_mat(0, width - 1, channel));

    v_20 = static_cast<float>(input_mat(1, width - 2, channel));
    v_21 = static_cast<float>(input_mat(1, width - 1, channel));

    result = v_10 * (w_00 + w_10) + v_11 * (w_01 + w_02 + w_11 + w_12) + v_20 * w_20 +
             v_21 * (w_21 + w_22);
    output_mat.set_value_f(0, width - 1, channel, result);

    for (int y = 1; y < height - 1; ++y) {
        // for special first
        v_01 = static_cast<float>(input_mat(y - 1, 0, channel));
        v_02 = static_cast<float>(input_mat(y - 1, 1, channel));
        v_11 = static_cast<float>(input_mat(y, 0, channel));
        v_12 = static_cast<float>(input_mat(y, 1, channel));
        v_21 = static_cast<float>(input_mat(y + 1, 0, channel));
        v_22 = static_cast<float>(input_mat(y + 1, 1, channel));

        result = v_01 * (w_00 + w_01) + v_02 * w_02 + v_11 * (w_10 + w_11) + v_12 * w_12 +
                 v_21 * (w_20 + w_21) + v_22 * w_22;
        output_mat.set_value_f(y, 0, channel, result);

        for (int x = 1; x < width - 1; ++x) {
            v_00 = static_cast<float>(input_mat(y - 1, x - 1, channel));
            v_01 = static_cast<float>(input_mat(y - 1, x, channel));
            v_02 = static_cast<float>(input_mat(y - 1, x + 1, channel));
            v_10 = static_cast<float>(input_mat(y, x - 1, channel));
            v_11 = static_cast<float>(input_mat(y, x, channel));
            v_12 = static_cast<float>(input_mat(y, x + 1, channel));
            v_20 = static_cast<float>(input_mat(y + 1, x - 1, channel));
            v_21 = static_cast<float>(input_mat(y + 1, x, channel));
            v_22 = static_cast<float>(input_mat(y + 1, x + 1, channel));

            result = w_00 * v_00 + w_01 * v_01 + w_02 * v_02 + w_10 * v_10 + w_11 * v_11 +
                     w_12 * v_12 + w_20 * v_20 + w_21 * v_21 + w_22 * v_22;
            output_mat.set_value_f(y, x, channel, result);
        }

        v_00 = static_cast<float>(input_mat(y - 1, width - 2, channel));
        v_01 = static_cast<float>(input_mat(y - 1, width - 1, channel));

        v_10 = static_cast<float>(input_mat(y, width - 2, channel));
        v_11 = static_cast<float>(input_mat(y, width - 1, channel));

        v_20   = static_cast<float>(input_mat(y + 1, width - 2, channel));
        v_21   = static_cast<float>(input_mat(y + 1, width - 1, channel));
        result = v_00 * w_00 + v_01 * (w_01 + w_02) + v_10 * w_10 + v_11 * (w_11 + w_12) +
                 v_20 * w_20 + v_21 * (w_21 + w_22);
        output_mat.set_value_f(y, width - 1, channel, result);
    }

    v_01 = static_cast<float>(input_mat(height - 2, 0, channel));
    v_02 = static_cast<float>(input_mat(height - 2, 1, channel));

    v_11 = static_cast<float>(input_mat(height - 1, 0, channel));
    v_12 = static_cast<float>(input_mat(height - 1, 1, channel));

    result = v_01 * (w_00 + w_01) + v_02 * w_02 + v_11 * (w_10 + w_11 + w_20 + w_21) +
             v_12 * (w_12 + w_22);
    output_mat.set_value_f(height - 1, 0, channel, result);

    for (int i = 0; i < width - 1; ++i) {
        v_00 = static_cast<float>(input_mat(height - 2, i - 1, channel));
        v_01 = static_cast<float>(input_mat(height - 2, i, channel));
        v_02 = static_cast<float>(input_mat(height - 2, i + 1, channel));

        v_10 = static_cast<float>(input_mat(height - 1, i - 1, channel));
        v_11 = static_cast<float>(input_mat(height - 1, i, channel));
        v_12 = static_cast<float>(input_mat(height - 1, i + 1, channel));

        result = v_00 * w_00 + v_01 * w_01 + v_02 * w_02 + v_10 * (w_10 + w_20) +
                 v_11 * (w_11 + w_21) + v_12 * (w_12 + w_22);
        output_mat.set_value_f(height - 1, i, channel, result);
    }

    v_00 = static_cast<float>(input_mat(height - 2, width - 2, channel));
    v_01 = static_cast<float>(input_mat(height - 2, width - 1, channel));
    v_10 = static_cast<float>(input_mat(height - 1, width - 2, channel));
    v_11 = static_cast<float>(input_mat(height - 1, width - 1, channel));

    result = v_00 * w_00 + v_01 * (w_01 + w_02) + v_10 * (w_10 + w_20) +
             v_11 * (w_11 + w_12 + w_21 + w_22);
    output_mat.set_value_f(height - 1, width - 1, channel, result);
}

template<class T, typename = dtype_limit<T>>
void convolution_1x1_impl(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                          const Mat<float>& kernel) {
    float    w         = kernel(0, 0);
    const T* input_ptr = input_mat.get_data_ptr();
    int      data_size = input_mat.get_element_num();
    for (int i = 0; i < data_size; ++i) {
        output_mat.set_value_f(i, static_cast<float>(input_ptr[i]) * w);
    }
}
}   // namespace internal

template<class T, typename>
Status::ErrorCode convolution_2d(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                                 const Mat<float>& kernel) {
    int kh = kernel.get_rows();
    int kw = kernel.get_cols();
    if (kh <= 0 || kw <= 0) {
        LOG_ERROR("got invaid kernel with shape ({},{})", kh, kw);
        return Status::ErrorCode::InvalidConvKernel;
    }

    if ((kh & 1) == 0 || (kw & 1) == 0) {
        LOG_ERROR("the kernel size must be odd number,but got kh:{} kw:{}", kh, kw);
        return Status::ErrorCode::InvalidConvKernel;
    }
    int height   = input_mat.get_height();
    int width    = input_mat.get_width();
    int channels = input_mat.get_channels();
    if (input_mat.empty()) {
        LOG_ERROR(
            "the input_mat is an invalid image with shape ({},{},{})", height, width, channels);
        return Status::ErrorCode::InvalidMatShape;
    }

    if (!input_mat.compare_shape(output_mat)) {
        output_mat.resize(height, width, channels, true);
    }
    Mat<float> scaled_kernel = internal::normalize_kernel(kernel);

    if (kh == 1 && kw == 1) {
        internal::convolution_1x1_impl(input_mat, output_mat, scaled_kernel);
    } else if (kh == 3 && kw == 3) {
        LOG_INFO("apply 3x3 special conv...");
        for (int i = 0; i < channels; ++i) {
            internal::convolution_3x3_impl(input_mat, output_mat, scaled_kernel, i);
        }
    } else {
        for (int i = 0; i < channels; ++i) {
            internal::convolution_generic_impl(input_mat, output_mat, scaled_kernel, i);
        }
    }
    return Status::ErrorCode::Ok;
}

template Status::ErrorCode convolution_2d<uint8_t>(const ImageMat<uint8_t>& input_mat,
                                                   ImageMat<uint8_t>&       output_mat,
                                                   const Mat<float>&        kernel);

template Status::ErrorCode convolution_2d<uint16_t>(const ImageMat<uint16_t>& input_mat,
                                                    ImageMat<uint16_t>&       output_mat,
                                                    const Mat<float>&         kernel);

template Status::ErrorCode convolution_2d<float>(const ImageMat<float>& input_mat,
                                                 ImageMat<float>&       output_mat,
                                                 const Mat<float>&      kernel);


template<class T, typename>
Status::ErrorCode convolution_2d(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                                 float* conv_kernel, int kh, int kw) {
    if (kh <= 0 || kw <= 0) {
        LOG_ERROR("got invaid kernel with shape ({},{})", kh, kw);
        return Status::ErrorCode::InvalidConvKernel;
    }

    if ((kh & 1) == 0 || (kw & 1) == 0) {
        LOG_ERROR("the kernel size must be odd number,but got kh:{} kw:{}", kh, kw);
        return Status::ErrorCode::InvalidConvKernel;
    }

    Mat<float> kernel(kh, kw, conv_kernel, MatMemLayout::LayoutRight, false);
    auto       status = convolution_2d<T>(input_mat, output_mat, kernel);
    return status;
}


template Status::ErrorCode convolution_2d<uint8_t>(const ImageMat<uint8_t>& input_mat,
                                                   ImageMat<uint8_t>&       output_mat,
                                                   float* conv_kernel, int kh, int kw);

template Status::ErrorCode convolution_2d<uint16_t>(const ImageMat<uint16_t>& input_mat,
                                                    ImageMat<uint16_t>&       output_mat,
                                                    float* conv_kernel, int kh, int kw);

template Status::ErrorCode convolution_2d<float>(const ImageMat<float>& input_mat,
                                                 ImageMat<float>& output_mat, float* conv_kernel,
                                                 int kh, int kw);

}   // namespace convolution
}   // namespace image_proc
}   // namespace fish