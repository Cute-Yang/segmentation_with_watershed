#include "image_proc/guassian_blur.h"
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "utils/logging.h"
#include <immintrin.h>
#include <vector>

namespace fish {
namespace image_proc {
namespace guassian_blur {
using namespace fish::core::base;
namespace internal {
std::vector<float> compute_kernel_impl(double sigma, int k_radius, int max_k_radius) {
    // clip the k_radius
    std::vector<float> kernel(k_radius);
    max_k_radius = FISH_MIN(max_k_radius, GUASSIAN_K_RADIUS_LIMIT);
    k_radius     = FISH_MIN(k_radius, max_k_radius);
    for (int i = 0; i < k_radius; ++i) {
        kernel[i] = static_cast<float>(std::exp(-0.5 * i * i / sigma / sigma));
    }

    // edge correct!
    if (k_radius > 3 && k_radius < max_k_radius) {
        LOG_INFO("do edge correction for kernel....");
        double slope_sqrt = std::numeric_limits<double>::max();
        int    r          = k_radius;
        while (r > k_radius / 2) {
            --r;
            // computhe the slope
            double temp = std::sqrt(kernel[r]) / (k_radius - r);
            if (temp < slope_sqrt) {
                slope_sqrt = temp;
            } else {
                // until slope greater equal than the previous,then break
                break;
            }
        }
        for (int r1 = r + 2; r1 < k_radius; ++r1) {
            kernel[r1] =
                static_cast<float>((k_radius - r1) * (k_radius - r1) * slope_sqrt * slope_sqrt);
        }
    }
    double guassian_sum;
    if (k_radius < max_k_radius) {
        guassian_sum = kernel[0];
        // because here we only compute the half guassian k,so the kernel size is 2 * k + 1
        for (int i = 1; i < k_radius; ++i) {
            guassian_sum += 2 * kernel[i];
        }
    } else {
        guassian_sum = sigma * std::sqrt(2.0 * Constant::CONSTANT_PI);
    }


    for (int i = 0; i < k_radius; ++i) {
        // normalize,make the sum of the guassian k = 1.0
        double scaled_guassian = kernel[i] / guassian_sum;
        kernel[i]              = scaled_guassian;
    }
    return kernel;
}

// this function will get the same result with path....
void compute_kernel_sum(double sigma, int k_radius, int max_k_radius, float* kernel) {
    std::vector<float> _kernel    = compute_kernel_impl(sigma, k_radius, max_k_radius);
    double             resume_sum = 0.5 + 0.5 * _kernel[0];
    // the resume sum should exlude the first element!
    for (int i = 0; i < k_radius; ++i) {
        resume_sum -= _kernel[i];
        kernel[i] = resume_sum;
    }
}

std::vector<float> compute_kernel_sum(float* _kernel, int _kernel_size) {
    std::vector<float> kernel_sum(_kernel_size);
    double             resume_sum = 0.5 + 0.5 * _kernel[0];
    // the resume sum should exlude the first element!
    for (int i = 0; i < _kernel_size; ++i) {
        resume_sum -= _kernel[i];
        kernel_sum[i] = resume_sum;
    }
    return kernel_sum;
}

void compute_kernel(double sigma, int k_radius, int max_k_radius, float* kernel) {
    // the first is center... hah
    std::vector<float> _kernel = compute_kernel_impl(sigma, k_radius, max_k_radius);
    std::copy(_kernel.begin(), _kernel.end(), kernel);
}



void compute_downscale_kernel(int block_size, float* kernel) {
    if (block_size == 0) {
        LOG_ERROR("the block size can not be zero!");
        return;
    }
    kernel[0] = 0.0f;
    if (((block_size * 3) & 2) == 0) {
        LOG_INFO("the downscale kernel have size {},the first kernel value will be set to zero!",
                 block_size);
    }
    int    mid          = block_size * 3 / 2;
    double block_size_f = 1.0 / static_cast<double>(block_size);
    for (int i = 0; i <= block_size / 2; ++i) {
        double x        = i * block_size_f;
        float  value    = static_cast<float>((0.75 - x * x) * block_size_f);
        kernel[mid - i] = value;
        kernel[mid + i] = value;
    }

    for (int i = block_size / 2; i < (block_size * 3 + 1) / 2; ++i) {
        double x        = i * block_size_f;
        float  value    = static_cast<float>((0.125 + 0.5 * (x - 1.0) * (x - 2.0)) * block_size_f);
        kernel[mid - i] = value;
        kernel[mid + i] = value;
    }
}

void compute_upscale_kernel(int block_size, float* kernel) {
    if (block_size == 0) {
        LOG_ERROR("the block size must greater than zero!");
        return;
    }

    int    mid          = 2 * block_size;
    double block_size_f = 1.0 / block_size;
    for (int i = 0; i < block_size; ++i) {
        double x        = i * block_size_f;
        float  v        = static_cast<float>(2.0 / 3.0 - x * x * (1 - 0.5 * x));
        kernel[mid - i] = v;
        kernel[mid + i] = v;
    }
    for (int i = block_size; i < 2 * block_size; ++i) {
        double x        = i * block_size_f;
        float  v        = static_cast<float>((2.0 - x) * (2.0 - x) * (2.0 - x) * (1.0 / 6.0));
        kernel[mid - i] = v;
        kernel[mid + i] = v;
    }
}

template<ImageDirectionKind direction>
void convolve_vector_impl_pad(const float* datas, int transform_size, ImageMat<float>& output_mat,
                              float* kernel, int kernel_size, int fixed_idx, int channel) {
    constexpr bool width_direction = (direction == ImageDirectionKind::Width);
    for (int i = 0; i < transform_size; ++i) {
        int   center_idx = i + kernel_size - 1;
        float result     = kernel[0] * datas[center_idx];
        for (int j = 1; j < kernel_size; ++j) {
            result += kernel[j] * datas[center_idx - j] + kernel[j] * datas[center_idx + j];
        }
        if constexpr (width_direction) {
            output_mat(fixed_idx, i, channel) = result;
        } else {
            output_mat(i, fixed_idx, channel) = result;
        }
    }
}

void convolve_vector_impl_pad(const float* datas, int transform_size, float* output_datas,
                              float* kernel, int kernel_size) {
    for (int i = 0; i < transform_size; ++i) {
        int   center_idx = i + kernel_size - 1;
        float result     = kernel[0] * datas[center_idx];
        for (int j = 1; j < kernel_size; ++j) {
            result += kernel[j] * datas[center_idx + j] + kernel[j] * datas[center_idx - j];
        }
        output_datas[i] = result;
    }
}

template<ImageDirectionKind direction>
void convolve_vector_impl(const float* datas, int data_size, ImageMat<float>& output_mat,
                          float* kernel, float* kernel_pad, int k_radius, int fixed_idx,
                          int channel) {
    constexpr bool is_width        = (direction == ImageDirectionKind::Width);
    float          left_pad_value  = datas[0];
    float          right_pad_value = datas[data_size - 1];
    int            stage_end       = (k_radius - 1) < data_size ? (k_radius - 1) : data_size;
    int            center_idx      = 0;
    for (; center_idx < stage_end; ++center_idx) {
        float result = datas[center_idx] * kernel[0];
        result += kernel_pad[center_idx] * left_pad_value;
        if (center_idx + k_radius > data_size) {
            result += kernel_pad[data_size - 1 - center_idx] * right_pad_value;
        }

        for (int i = 1; i < k_radius; ++i) {
            if (center_idx - i >= 0) {
                result += kernel[i] * datas[center_idx - i];
            }
            if ((center_idx + i) < data_size) FISH_UNLIKELY_STD {
                    result += kernel[i] * datas[center_idx + i];
                }
        }

        if constexpr (is_width) {
            output_mat(fixed_idx, center_idx, channel) = result;
        } else {
            output_mat(center_idx, fixed_idx, channel) = result;
        }
    }

    stage_end = data_size - k_radius + 1;
    for (; center_idx < stage_end; ++center_idx) {
        float result = datas[center_idx] * kernel[0];
        for (int i = 0; i < k_radius; ++i) {
            result += datas[center_idx - i] * kernel[i] + datas[center_idx + i] * kernel[i];
        }
        if constexpr (is_width) {
            output_mat(fixed_idx, center_idx, channel) = result;
        } else {
            output_mat(center_idx, fixed_idx, channel) = result;
        }
    }

    for (; center_idx < data_size; ++center_idx) {
        float result = datas[center_idx] * kernel[0];
        if (center_idx + k_radius >= data_size) {
            result += kernel_pad[data_size - center_idx - 1] * right_pad_value;
        }
        for (int i = 0; i < k_radius; ++i) {
            if (center_idx - i >= 0) {
                result += kernel[i] * datas[center_idx - i];
            }
            if (center_idx + i < data_size) {
                result += kernel[i] * datas[center_idx + i];
            }
        }
        if constexpr (is_width) {
            output_mat(fixed_idx, center_idx, channel) = result;
        } else {
            output_mat(center_idx, fixed_idx, channel) = result;
        }
    }
}

void convolve_vector_impl(const float* datas, int data_size, float* output_datas, float* kernel,
                          float* kernel_pad, int k_radius) {
    float left_pad_value  = datas[0];
    float right_pad_value = datas[data_size - 1];
    int   stage_end       = (k_radius - 1) < data_size ? (k_radius - 1) : data_size;
    int   center_idx      = 0;
    for (; center_idx < stage_end; ++center_idx) {
        float result = datas[center_idx] * kernel[0];
        result += kernel_pad[center_idx] * left_pad_value;
        if (center_idx + k_radius > data_size) {
            result += kernel_pad[data_size - 1 - center_idx] * right_pad_value;
        }

        for (int i = 1; i < k_radius; ++i) {
            if (center_idx - i >= 0) {
                result += kernel[i] * datas[center_idx - i];
            }
            if ((center_idx + i) < data_size) FISH_UNLIKELY_STD {
                    result += kernel[i] * datas[center_idx + i];
                }
        }
        output_datas[center_idx] = result;
    }

    stage_end = data_size - k_radius + 1;
    for (; center_idx < stage_end; ++center_idx) {
        float result = datas[center_idx] * kernel[0];
        for (int i = 0; i < k_radius; ++i) {
            result += datas[center_idx - i] * kernel[i] + datas[center_idx + i] * kernel[i];
        }
        output_datas[center_idx] = result;
    }

    for (; center_idx < data_size; ++center_idx) {
        float result = datas[center_idx] * kernel[0];
        if (center_idx + k_radius >= data_size) {
            result += kernel_pad[data_size - center_idx - 1] * right_pad_value;
        }
        for (int i = 0; i < k_radius; ++i) {
            if (center_idx - i >= 0) {
                result += kernel[i] * datas[center_idx - i];
            }
            if (center_idx + i < data_size) {
                result += kernel[i] * datas[center_idx + i];
            }
        }
        output_datas[center_idx] = result;
    }
}


template<class T, ImageDirectionKind direction, typename = dtype_limit<T>>
void downscale_vector_impl(const ImageMat<T>& input_mat, float* downscale_datas, int transform_size,
                           float* kernel, int block_size, int unscaled_0, int channel,
                           int fixed_idx) {
    int            data_idx        = (unscaled_0 - block_size * 3 / 2);
    constexpr bool width_direction = (direction == ImageDirectionKind::Width);
    int            data_size = width_direction ? input_mat.get_width() : input_mat.get_height();
    for (int center_idx = -1; center_idx <= transform_size; ++center_idx) {
        float sum_0 = 0.0f;
        float sum_1 = 0.0f;
        float sum_2 = 0.0f;
        for (int s = 0; s < block_size; ++s, ++data_idx) {
            // avoid out of range!
            int   idx = data_idx < 0 ? 0 : (data_idx < data_size) ? data_idx : data_size - 1;
            float value;
            if constexpr (width_direction) {
                value = static_cast<float>(input_mat(fixed_idx, idx, channel));
            } else {
                value = static_cast<float>(input_mat(idx, fixed_idx, channel));
            }
            sum_0 += value * kernel[s + 2 * block_size];
            sum_1 += value * kernel[s + block_size];
            sum_2 += value * kernel[s];
        }
        if (center_idx > 0) {
            downscale_datas[center_idx - 1] += sum_0;
        }
        if (center_idx >= 0 && center_idx < transform_size) {
            downscale_datas[center_idx] += sum_1;
        }
        // right pixel
        if (center_idx + 1 < transform_size) {
            downscale_datas[center_idx + 1] = sum_2;
        }
    }
}

template<ImageDirectionKind direction>
void upscale_vector_impl(const float* sample_datas, int data_size, ImageMat<float>& output_mat,
                         float* kernel, int block_size, int unscaled_0, int fixed_idx,
                         int channel) {
    constexpr bool width_direction = (direction == ImageDirectionKind::Width);
    for (int out_idx = 0; out_idx < data_size; ++out_idx) {
        int in_idx = (out_idx - unscaled_0 + block_size - 1) / block_size;
        // the 0 <= k_idx <= reduce_block_size -1
        int   k_idx = block_size - 1 - (out_idx - unscaled_0 + block_size - 1) % block_size;
        float value = sample_datas[in_idx - 2] * kernel[k_idx] +
                      sample_datas[in_idx - 1] * kernel[k_idx + block_size] +
                      sample_datas[in_idx] * kernel[k_idx + 2 * block_size] +
                      sample_datas[in_idx + 1] * kernel[k_idx + 3 * block_size];
        if constexpr (width_direction) {
            output_mat(fixed_idx, out_idx, channel) = value;
        } else {
            output_mat(out_idx, fixed_idx, channel) = value;
        }
    }
}

template<class T, ImageDirectionKind direction, bool pad = true, typename = image_dtype_limit<T>>
void guassian_blur_1d_impl(const ImageMat<T>& input_mat, ImageMat<float>& output_mat, int channel,
                           double sigma) {
    constexpr bool   is_float_type       = FloatTypeRequire<T>::value;
    constexpr double acc                 = is_float_type ? GUASSIAN_HIGH_ACC : GUASSIAN_LOW_ACC;
    constexpr int    UPSCALE_K_RADIUS    = 2;
    constexpr double MIN_DOWNSCALE_SIGMA = 4.0;
    constexpr bool   width_direction     = (direction == ImageDirectionKind::Width);
    if constexpr (width_direction) {
        LOG_INFO("apply guassian for width direction....");
    } else {
        LOG_INFO("apply guassian for height direction...");
    }

    int height = input_mat.get_height();
    int width  = input_mat.get_width();

    int transform_num = width_direction ? height : width;

    int data_size = width_direction ? width : height;


    bool downscale = (sigma > 2.0 * MIN_DOWNSCALE_SIGMA + 0.5);

    int block_size =
        downscale ? std::min(static_cast<int>(std::floor(sigma / MIN_DOWNSCALE_SIGMA)), data_size)
                  : 1;
    double sigma_guassian =
        downscale ? std::sqrt(sigma * sigma / (block_size * block_size) - 1.0 / 3.0 - 1.0 / 4.0)
                  : sigma;
    LOG_INFO("the given sigam is {},the adjust sigma is {}", sigma, sigma_guassian);

    int max_data_size = downscale
                            ? (data_size + block_size - 1) / block_size + 2 * (UPSCALE_K_RADIUS + 1)
                            : data_size;
    LOG_INFO("the max line size for guassian blur is {}", max_data_size);
    int                k_radius        = compute_k_radius(sigma_guassian, acc);
    std::vector<float> guassian_kernel = compute_kernel(sigma_guassian, k_radius, max_data_size);
    std::vector<float> guassian_kernel_sum =
        compute_kernel_sum(guassian_kernel.data(), guassian_kernel.size());

    int unscaled_0    = -(UPSCALE_K_RADIUS + 1) * block_size;
    int pad_size      = 2 * (k_radius - 1);
    int half_pad_size = k_radius - 1;
    if (downscale) {
        LOG_INFO("we will do downscale conv and upscale for the data...");
        int new_data_size =
            downscale ? (data_size + block_size - 1) / block_size + 2 * (UPSCALE_K_RADIUS + 1)
                      : data_size;
        LOG_INFO("the vector size is {},the new vector size is {}", data_size, new_data_size);
        std::vector<float> downscale_kernel = compute_downscale_kernel(block_size);
        std::vector<float> upscale_kernel   = compute_upscale_kernel(block_size);

        int                downscale_data_size = pad ? new_data_size + pad_size : new_data_size;
        std::vector<float> downscale_datas(downscale_data_size);
        std::vector<float> conv_datas(new_data_size);
        float*             downscale_output_ptr =
            pad ? downscale_datas.data() + half_pad_size : downscale_datas.data();
        for (int i = 0; i < transform_num; ++i) {
            downscale_vector_impl<T, direction>(input_mat,
                                                downscale_output_ptr,
                                                new_data_size,
                                                downscale_kernel.data(),
                                                block_size,
                                                unscaled_0,
                                                channel,
                                                i);
            if constexpr (pad) {
                float left_pad_value = downscale_datas[half_pad_size];
                for (int j = 0; j < half_pad_size; ++j) {
                    downscale_datas[j] = left_pad_value;
                }
                float right_pad_value = downscale_datas[half_pad_size + new_data_size - 1];
                for (int j = 0; j < half_pad_size; ++j) {
                    downscale_datas[half_pad_size + new_data_size + j] = right_pad_value;
                }
            }
            if constexpr (pad) {
                convolve_vector_impl_pad(downscale_datas.data(),
                                         new_data_size,
                                         conv_datas.data(),
                                         guassian_kernel.data(),
                                         guassian_kernel.size());
            } else {
                convolve_vector_impl(downscale_datas.data(),
                                     new_data_size,
                                     conv_datas.data(),
                                     guassian_kernel.data(),
                                     guassian_kernel_sum.data(),
                                     k_radius);
            }
            // the first value should be set zero!
            conv_datas[0] = 0.0f;
            upscale_vector_impl<direction>(conv_datas.data(),
                                           data_size,
                                           output_mat,
                                           upscale_kernel.data(),
                                           block_size,
                                           unscaled_0,
                                           i,
                                           channel);
        }


    } else {
        int                temp_data_size = pad ? data_size + pad_size : data_size;
        std::vector<float> transform_datas(temp_data_size);
        for (int i = 0; i < transform_num; ++i) {
            if constexpr (width_direction) {
                if constexpr (pad) {
                    // padding left
                    float left_pad_vaue = static_cast<float>(input_mat(i, 0, channel));
                    for (int j = 0; j < half_pad_size; ++j) {
                        transform_datas[j] = left_pad_vaue;
                    }
                    for (int j = 0; j < data_size; ++j) {
                        transform_datas[half_pad_size + j] =
                            static_cast<float>(input_mat(i, j, channel));
                    }
                    // padding right
                    float right_pad_value = input_mat(i, data_size - 1, channel);
                    for (int j = 0; j < half_pad_size; ++j) {
                        transform_datas[half_pad_size + data_size + j] = right_pad_value;
                    }
                } else {
                    for (int j = 0; j < data_size; ++j) {
                        transform_datas[j] = static_cast<float>(input_mat(i, j, channel));
                    }
                }
            } else {
                if constexpr (pad) {
                    float left_pad_value = static_cast<float>(input_mat(0, i, channel));
                    for (int j = 0; j < half_pad_size; ++j) {
                        transform_datas[j] = left_pad_value;
                    }
                    for (int j = 0; j < data_size; ++j) {
                        transform_datas[half_pad_size + j] =
                            static_cast<float>(input_mat(j, i, channel));
                    }
                    float right_pad_value =
                        static_cast<float>(input_mat(data_size - 1, i, channel));
                    for (int j = 0; j < half_pad_size; ++j) {
                        transform_datas[half_pad_size + data_size + j] = right_pad_value;
                    }
                } else {
                    for (int j = 0; j < data_size; ++j) {
                        transform_datas[j] = static_cast<float>(input_mat(j, i, channel));
                    }
                }
            }
            if constexpr (pad) {
                convolve_vector_impl_pad<direction>(transform_datas.data(),
                                                    data_size,
                                                    output_mat,
                                                    guassian_kernel.data(),
                                                    guassian_kernel.size(),
                                                    i,
                                                    channel);
            } else {
                convolve_vector_impl<direction>(transform_datas.data(),
                                                data_size,
                                                output_mat,
                                                guassian_kernel.data(),
                                                guassian_kernel_sum.data(),
                                                k_radius,
                                                i,
                                                channel);
            }
        }
    }
}


template<class T, typename = image_dtype_limit<T>>
void guassian_blur_2d_impl(const ImageMat<T>& input_mat, ImageMat<T>& output_mat, double sigma_x,
                           double sigma_y) {
    constexpr bool is_expected_mat_type = ImageTypeRequire<T>::value;
    FISH_StaticAssert(is_expected_mat_type, "the type is not expected!");
    if (input_mat.compare_shape(output_mat)) {
        LOG_WARN("the input mat and output mat have differenct shape,we will reshape output mat!");
        output_mat.resize(
            input_mat.get_height(), input_mat.get_width(), input_mat.get_channels(), true);
    }
    output_mat.set_layout(input_mat.get_layout());

    // make the output_mat have the same layout with input!
    constexpr bool is_float_type = FloatTypeRequire<T>::value;
    if constexpr (is_float_type) {
        for (int i = 0; i < input_mat.get_channels(); ++i) {
            guassian_blur_1d_impl<float, ImageDirectionKind::Width>(
                input_mat, output_mat, i, sigma_x);
            guassian_blur_1d_impl<float, ImageDirectionKind::Height>(
                output_mat, output_mat, i, sigma_y);
        }
    } else {
        LOG_INFO("the mat type is interger,we need to allocate a middle mat to store the result!");
        int          height   = input_mat.get_height();
        int          width    = input_mat.get_width();
        int          channles = input_mat.get_channels();
        MatMemLayout laytout  = input_mat.get_layout();

        ImageMat<float> middle_mat(height, width, channles, laytout);
        for (int i = 0; i < input_mat.get_channels(); ++i) {
            guassian_blur_1d_impl<T, ImageDirectionKind::Width>(input_mat, middle_mat, i, sigma_x);
            guassian_blur_1d_impl<float, ImageDirectionKind::Height>(
                middle_mat, middle_mat, i, sigma_y);
        }
        // convert_mat(middle_mat, output_mat);
        convert_image<float, T>(middle_mat, output_mat);
    }
}
}   // namespace internal


template<class T, typename>
Status::ErrorCode guassian_blur_2d(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                                   double sigma) {
    if (sigma <= 0) {
        LOG_ERROR("Invalid sigma value for guassian blur...");
        return Status::ErrorCode::InvalidGuassianParam;
    }
    if (input_mat.get_height() == 0 || input_mat.get_width() == 0 ||
        input_mat.get_channels() == 0) {
        LOG_ERROR("the mat can not have zero dimension,but have shape ({},{},{})",
                  input_mat.get_height(),
                  input_mat.get_width(),
                  input_mat.get_channels());
        return Status::ErrorCode::InvalidMatShape;
    }
    LOG_INFO("the sigma_x and sigma_y will be same {}", sigma);
    internal::guassian_blur_2d_impl<T>(input_mat, output_mat, sigma, sigma);
    return Status::ErrorCode::Ok;
}

template Status::ErrorCode guassian_blur_2d<uint8_t>(const ImageMat<uint8_t>& input_mat,
                                                     ImageMat<uint8_t>& output_mat, double sigma);

template Status::ErrorCode guassian_blur_2d<uint16_t>(const ImageMat<uint16_t>& input_mat,
                                                      ImageMat<uint16_t>& output_mat, double sigma);

template Status::ErrorCode guassian_blur_2d<float>(const ImageMat<float>& input_mat,
                                                   ImageMat<float>& output_mat, double sigma);

template<class T>
Status::ErrorCode guassian_blur_2d(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                                   double sigma_x, double sigma_y) {
    if (sigma_x <= 0 || sigma_y <= 0) {
        LOG_ERROR("Invalid sigma value for guassian blur...");
        return Status::ErrorCode::InvalidGuassianParam;
    }
    if (input_mat.get_height() == 0 || input_mat.get_width() == 0 ||
        input_mat.get_channels() == 0) {
        LOG_ERROR("the mat can not have zero dimension,but have shape ({},{},{})",
                  input_mat.get_height(),
                  input_mat.get_width(),
                  input_mat.get_channels());
        return Status::ErrorCode::InvalidMatShape;
    }
    internal::guassian_blur_2d_impl<T>(input_mat, output_mat, sigma_x, sigma_y);
    return Status::ErrorCode::Ok;
}



}   // namespace guassian_blur
}   // namespace image_proc
}   // namespace fish