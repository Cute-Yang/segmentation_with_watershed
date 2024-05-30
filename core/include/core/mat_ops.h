#pragma once
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <xmmintrin.h>
namespace fish {
namespace core {
namespace mat_ops {
using namespace fish::core::mat;
enum ValueOpKind : uint32_t {
    ONLY_COPY = 0,
    // using 255 - src for char
    COPY_INVERTED    = 1,
    COPY_TRANSPARENT = 2,
    ADD              = 3,
    SUBSTRACT        = 4,
    MULTIPLY         = 5,
    DIVIDE           = 6,
    AVERAGE          = 7,
    // abs(dst -src)
    DIFFERENCE            = 8,
    AND                   = 9,
    OR                    = 10,
    XOR                   = 11,
    MIN                   = 12,
    MAX                   = 13,
    COPY_ZERO_TRANSPARENT = 14,
    OpCount
};
constexpr bool is_valid_channel(int channel_idx, int channels) {
    return channel_idx >= 0 && channel_idx <= channels;
}
template<class T, typename = image_dtype_limit<T>>
Status::ErrorCode copy_image_mat(const ImageMat<T>& src_mat, ImageMat<T>& dst_mat, ValueOpKind op);

enum class MatCompareOpType : uint8_t {
    GREATER       = 0,
    LESS          = 1,
    EQUAL         = 2,
    GREATER_EQUAL = 3,
    LESS_EQULA    = 4
};
// can use avx to optimize!
//  reuse the memory!
template<class T, MatCompareOpType op, typename = image_dtype_limit<T>>
Status::ErrorCode compare_mat(const ImageMat<T>& lhs, const ImageMat<T>& rhs,
                              ImageMat<uint8_t>& mask) {
    if (!lhs.compare_shape(rhs)) {
        // if the shape mismatch,we do not apply compare!
        return Status::ErrorCode::MatShapeMismatch;
    }
    int height   = lhs.get_height();
    int width    = lhs.get_width();
    int channels = lhs.get_channels();

    // only for empty mat is ok!
    if (!mask.compare_shape(lhs)) {
        mask.resize(height, width, channels, true);
    }

    const T* lhs_ptr = lhs.get_data_ptr();
    const T* rhs_ptr = rhs.get_data_ptr();
    // ImageArray<unsigned char>::zeros(height, width, 1, ImageMemoryLayout::NHWC);
    unsigned char* mask_ptr = mask.get_data_ptr();
    for (int i = 0; i < height * width; ++i) {
        if constexpr (op == MatCompareOpType::EQUAL) {
            if (lhs_ptr[i] == rhs_ptr[i]) {
                mask_ptr[i] = 255;
            } else {
                mask_ptr[i] = 0;
            }
        } else if constexpr (op == MatCompareOpType::GREATER) {
            if (lhs_ptr[i] > rhs_ptr[i]) {
                mask_ptr[i] = 255;
            } else {
                mask_ptr[i] = 0;
            }
        } else if constexpr (op == MatCompareOpType::GREATER_EQUAL) {
            if (lhs_ptr[i] >= rhs_ptr[i]) {
                mask_ptr[i] = 255;
            } else {
                mask_ptr[i] = 0;
            }
        } else if constexpr (op == MatCompareOpType::LESS) {
            if (lhs_ptr[i] < rhs_ptr[i]) {
                mask_ptr[i] = 255;
            } else {
                mask_ptr[i] = 0;
            }
        } else if constexpr (op == MatCompareOpType::LESS_EQULA) {
            if (lhs_ptr[i] <= rhs_ptr[i]) {
                mask_ptr[i] = 255;
            } else {
                mask_ptr[i] = 0;
            }
        }
    }
    return Status::Ok;
}


template<class T, MatCompareOpType op, typename = image_dtype_limit<T>>
ImageMat<uint8_t> compare_mat(const ImageMat<T>& lhs, const ImageMat<T>& rhs) {
    ImageMat<uint8_t> mask;
    compare_mat<T, op>(lhs, rhs, mask);
    return mask;
}




template<class T, typename = image_dtype_limit<T>>
Status::ErrorCode threshold_above(const ImageMat<T>& image, ImageMat<uint8_t>& mask, T threshold) {
    const T* image_ptr = image.get_data_ptr();
    int      height    = image.get_height();
    int      width     = image.get_width();
    int      channels  = image.get_channels();
    if (channels != 1) {
        return Status::ErrorCode::InvalidMatChannle;
    }

    if (!mask.shape_equal(height, width, 1)) {
        mask.resize(height, width, 1, true);
    }
    // make sure the mask and image have same layout!
    if (image.get_layout() != mask.get_layout()) {
        return Status::ErrorCode::MatLayoutMismath;
    }

    unsigned char* mask_ptr  = mask.get_data_ptr();
    int            data_size = height * width;
    for (int i = 0; i < data_size; ++i) {
        if (image_ptr[i] > threshold) {
            mask_ptr[i] = 255;
        } else {
            mask_ptr[i] = 0;
        }
    }
    return Status::ErrorCode::Ok;
}

template<class T, typename = image_dtype_limit<T>>
ImageMat<uint8_t> threshold_above(const ImageMat<T>& image, T threshold) {
    ImageMat<uint8_t> mask(image.get_height(), image.get_width(), 1, MatMemLayout::LayoutRight);
    threshold_above(image, mask, threshold);
    return mask;
}


using LutValueType = std::array<uint8_t, 256>;
enum class LutValueOpType : uint8_t {
    INVERT  = 0,
    FILL    = 1,
    SET     = 2,
    ADD     = 3,
    MULT    = 4,
    AND     = 5,
    OR      = 6,
    XOR     = 7,
    GAMMA   = 8,
    LOG     = 9,
    EXP     = 10,
    SQR     = 11,
    SQRT    = 12,
    MINIMUM = 13,
    MAXIMUM = 14,
    OTHER   = 15
};


template<LutValueOpType op> LutValueType compute_lut(double value, int fg_color) {
    static double SCALE   = 255.0 / std::log(255.0);
    int           value_i = static_cast<int>(value);
    int           lut_v;
    LutValueType  lut;
    for (int i = 0; i < 256; ++i) {
        if constexpr (op == LutValueOpType::INVERT) {
            lut_v = 255 - i;
        } else if constexpr (op == LutValueOpType::FILL) {
            lut_v = fg_color;
        } else if constexpr (op == LutValueOpType::SET) {
            lut_v = value_i;
        } else if constexpr (op == LutValueOpType::ADD) {
            lut_v = i + value_i;
        } else if constexpr (op == LutValueOpType::MULT) {
            lut_v = static_cast<int>(i * value + 0.5f);
        } else if constexpr (op == LutValueOpType::AND) {
            lut_v = i & value_i;
        } else if constexpr (op == LutValueOpType::OR) {
            lut_v = i | value_i;
        } else if constexpr (op == LutValueOpType::XOR) {
            lut_v = i ^ value_i;
        } else if constexpr (op == LutValueOpType::GAMMA) {
            lut_v = static_cast<int>(std::exp(std::log(i / 255.0) * value) * 255.0);
        } else if constexpr (op == LutValueOpType::LOG) {
            lut_v = i == 0 ? 0 : static_cast<int>(std::log(i) * SCALE);
        } else if constexpr (op == LutValueOpType::EXP) {
            lut_v = static_cast<int>(std::exp(i / SCALE));
        } else if constexpr (op == LutValueOpType::SQR) {
            lut_v = i * i;
        } else if constexpr (op == LutValueOpType::SQRT) {
            lut_v = static_cast<int>(std::sqrt(i));
        } else if constexpr (op == LutValueOpType::MINIMUM) {
            lut_v = FISH_MIN(i, value);
        } else if constexpr (op == LutValueOpType::MAXIMUM) {
            lut_v = FISH_MAX(i, value);
        } else {
            lut_v = i;
        }
        if (lut_v < 0) {
            lut_v = 0;
        } else if (lut_v > 255) {
            lut_v = 255;
        }
        lut[i] = lut_v;
    }
    return lut;
}

FISH_INLINE void fill_continous_memory(void* dst, int data_size, uint8_t filled_value) {
    uint8_t*  buf  = reinterpret_cast<uint8_t*>(dst);
    uintptr_t addr = reinterpret_cast<uintptr_t>(dst);
    // if the addr % word != 0,we firstly copy these unaligned data one by one,then copy the remains
    // by word!
    constexpr int aligned_byte     = sizeof(size_t);
    uintptr_t     word_addr        = addr / aligned_byte * aligned_byte;
    int           not_aligned_size = 0;
    if (word_addr < addr) {
        not_aligned_size = word_addr + aligned_byte - addr;
        not_aligned_size = FISH_MIN(not_aligned_size, data_size);
        // copy one by one!
        for (int i = 0; i < not_aligned_size; ++i) {
            buf[i] = filled_value;
        }
        data_size -= not_aligned_size;
    }
    int   filled_value_word;
    char* filled_value_word_ptr = reinterpret_cast<char*>(&filled_value_word);

    // fill by word!
    if constexpr (aligned_byte == 4) {
        filled_value_word_ptr[0] = filled_value;
        filled_value_word_ptr[1] = filled_value;
        filled_value_word_ptr[2] = filled_value;
        filled_value_word_ptr[3] = filled_value;
    } else if constexpr (aligned_byte == 8) {
        filled_value_word_ptr[0] = filled_value;
        filled_value_word_ptr[1] = filled_value;
        filled_value_word_ptr[2] = filled_value;
        filled_value_word_ptr[3] = filled_value;
        filled_value_word_ptr[4] = filled_value;
        filled_value_word_ptr[5] = filled_value;
        filled_value_word_ptr[6] = filled_value;
        filled_value_word_ptr[7] = filled_value;
    }
    // copy by word!
    size_t* buf_word = reinterpret_cast<size_t*>(buf + not_aligned_size);
    for (int i = 0; i < data_size / aligned_byte; ++i) {
        buf_word[i] = filled_value_word;
    }

    // copy one by one!
    for (int i = data_size * aligned_byte / aligned_byte; i < data_size; ++i) {
        buf[i + not_aligned_size] = filled_value;
    }
}

namespace simple_threshold {
// this func support multi channels...
template<class T, typename = dtype_limit<T>>
Status::ErrorCode greater_equal_than(const ImageMat<T>& lhs, const ImageMat<T>& rhs,
                                     ImageMat<uint8_t>& mask, uint8_t fill_value) {
    // if fill value is 255,we can use the fast fill with avx ...
    if (!lhs.compare_shape(rhs)) {
        return Status::ErrorCode::MatShapeMismatch;
    }
    if (lhs.get_layout() != rhs.get_layout()) {
        return Status::ErrorCode::MatLayoutMismath;
    }
    int height   = lhs.get_height();
    int width    = lhs.get_width();
    int channels = lhs.get_channels();

    // if the buffer is enough,do not allocate new buffer!
    mask.resize(height, width, channels, false);
    mask.set_zero();

    const T* lhs_ptr  = lhs.get_data_ptr();
    const T* rhs_ptr  = rhs.get_data_ptr();
    uint8_t* mask_ptr = mask.get_data_ptr();
    // use sse to optimzie it
    // to do,if fill value = 255,_mm_gteq__
    int data_size = height * width * channels;
    for (int i = 0; i < data_size; ++i) {
        // add avx
        if (lhs_ptr[i] >= rhs_ptr[i]) {
            mask_ptr[i] = fill_value;
        }
    }
}

template<class T, typename = dtype_limit<T>>
Status::ErrorCode greater_equal_than(const ImageMat<T>& lhs, const ImageMat<T>& rhs,
                                     ImageMat<uint8_t>& mask, int channel, uint8_t fill_value) {
    if (!lhs.compare_shape(rhs)) {
        return Status::ErrorCode::MatShapeMismatch;
    }
    if (lhs.get_layout() != rhs.get_layout()) {
        return Status::ErrorCode::MatLayoutMismath;
    }
    int height   = lhs.get_height();
    int width    = lhs.get_width();
    int channels = lhs.get_channels();

    if (channel < 0 || channel >= channels) {
        return Status::ErrorCode::InvalidMatChannle;
    }

    // if the buffer is enough,do not allocate new buffer!
    mask.resize(height, width, 1, false);
    mask.set_zero();

    const T* lhs_ptr  = lhs.get_data_ptr();
    const T* rhs_ptr  = rhs.get_data_ptr();
    uint8_t* mask_ptr = mask.get_data_ptr();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (lhs(y, x, channel) > rhs(y, x, channel)) {
                mask(y, x) = fill_value;
            }
        }
    }
}

template<class T, typename = dtype_limit<T>>
Status::ErrorCode greater_equal_than(const ImageMat<T>& image, ImageMat<uint8_t>& mask,
                                     int lhs_channel, int rhs_channel, uint8_t fill_value) {
    if (image.empty()) {
        return Status::ErrorCode::InvalidMatShape;
    }
    int height   = image.get_height();
    int width    = image.get_width();
    int channels = image.get_channels();
    if (!is_valid_channel(lhs_channel, channels) || !is_valid_channel(rhs_channel, channels)) {
        return Status::InvalidMatChannle;
    }

    if (!mask.shape_equal(height, width, 1)) {
        mask.resize(height, width, 1, true);
    }

    mask.set_zero();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (image(y, x, lhs_channel) > image(y, x, rhs_channel)) {
                mask(y, x) = fill_value;
            }
        }
    }
    return Status::ErrorCode::Ok;
}

template<class T, typename = dtype_limit<T>>
ImageMat<uint8_t> greater_equal_than(const ImageMat<T>& lhs, const ImageMat<T>& rhs,
                                     uint8_t fill_value) {
    ImageMat<uint8_t> mask;
    // here we return an invalid mask!
    if (!lhs.compare_shape(rhs)) {
        return mask;
    }
    greater_equal_than(lhs, rhs, mask, fill_value);
    return mask;
}

template<class T, typename = dtype_limit<T>>
ImageMat<uint8_t> greater_equal_than(const ImageMat<T>& lhs, const ImageMat<T>& rhs, int channel,
                                     uint8_t fill_value) {
    ImageMat<uint8_t> mask;
    // here we return an invalid mask!
    if (!lhs.compare_shape(rhs)) {
        return mask;
    }
    greater_equal_than(lhs, rhs, mask, channel, fill_value);
    return mask;
}

// if the ret is empty,means that fail to invoke...
template<class T, typename = dtype_limit<T>>
ImageMat<uint8_t> greater_equal_than(const ImageMat<T>& image, int lhs_channel, int rhs_channel,
                                     uint8_t fill_value) {
    ImageMat<uint8_t> mask;
    if (image.empty()) {
        return mask;
    }
    greater_equal_than(image, mask, lhs_channel, rhs_channel, fill_value);
    return mask;
}
}   // namespace simple_threshold

}   // namespace mat_ops
}   // namespace core
}   // namespace fish