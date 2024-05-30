#include "core/mat_ops.h"
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "utils/logging.h"
#include <cmath>
#include <limits>

namespace fish {
namespace core {
namespace mat_ops {
namespace internal {
template<class T, typename = dtype_limit<T>> T FISH_ALWAYS_INLINE clip_value(float value) {
    constexpr T     type_min_value   = std::numeric_limits<T>::min();
    constexpr T     type_max_value   = std::numeric_limits<T>::max();
    constexpr float type_min_value_f = static_cast<float>(type_min_value);
    constexpr float type_max_value_f = static_cast<float>(type_max_value);
    if (value < type_min_value_f) [[unlikely]] {
        return type_min_value;
    } else if (value > type_max_value_f) [[unlikely]] {
        return type_max_value;
    } else {
        return static_cast<T>(value + 0.5f);
    }
}


template<class T, ValueOpKind op, typename = dtype_limit<T>>
FISH_ALWAYS_INLINE T value_op(T src, T dst) {
    constexpr T    zero_value{0};
    T              value          = zero_value;
    constexpr T    type_min_value = std::numeric_limits<T>::min();
    constexpr bool type_is_signed = type_min_value < 0;
    if constexpr (op == ValueOpKind::ONLY_COPY) {
        value = src;
    } else if constexpr (op == ValueOpKind::COPY_ZERO_TRANSPARENT) {
        value = (src == 0 ? dst : src);
    } else if constexpr (op == ValueOpKind::ADD) {
        value = src + dst;
    } else if constexpr (op == ValueOpKind::AVERAGE) {
        if constexpr (FloatTypeRequire<T>::value) {
            T _value = (src + dst) * 0.5;
            value    = _value;
        } else {
            float _value = (static_cast<float>(src) + static_cast<float>(dst)) * 0.5f;
            value        = _value;
        }
    } else if constexpr (op == ValueOpKind::DIFFERENCE) {
        if constexpr (type_is_signed) {
            value = std::abs(dst - src);
        } else {
            // to handle the unsigned int type!
            if (dst > src) {
                value = dst - src;
            } else {
                value = src - dst;
            }
        }
    } else if constexpr (op == ValueOpKind::SUBSTRACT) {
        value = dst - src;
    } else if constexpr (op == ValueOpKind::MULTIPLY) {
        if constexpr (FloatTypeRequire<T>::value) {
            value = dst * src;
        } else {
            float _value = static_cast<float>(src) * static_cast<float>(dst);
            value        = clip_value<T>(_value);
        }
    } else if constexpr (op == ValueOpKind::DIVIDE) {
        if constexpr (FloatTypeRequire<T>::value) {
            if (src == 0.0) [[unlikely]] {
                value = 0.0;
            } else {
                value = dst / src;
            }
        } else {
            if (src == 0) [[unlikely]] {
                value = 0;
            } else {
                float _value = static_cast<float>(dst) / static_cast<float>(src);
                value        = clip_value<T>(_value);
            }
        }
    } else if constexpr (op == ValueOpKind::AND) {
        if constexpr (FloatTypeRequire<T>::value) {
            value = static_cast<int>(src) & static_cast<int>(dst);
        } else {
            value = src & dst;
        }
    } else if constexpr (op == ValueOpKind::OR) {
        if constexpr (FloatTypeRequire<T>::value) {
            value = static_cast<int>(src) | static_cast<int>(dst);
        } else {
            value = src | dst;
        }
    } else if constexpr (op == ValueOpKind::XOR) {
        if constexpr (FloatTypeRequire<T>::value) {
            value = static_cast<int>(src) ^ static_cast<int>(dst);
        } else {
            value = src ^ dst;
        }
    } else if constexpr (op == ValueOpKind::MIN) {
        value = FISH_MIN(src, dst);
    } else if constexpr (op == ValueOpKind::MAX) {
        value = FISH_MAX(src, dst);
    }
    return value;
}

template<class T, ValueOpKind op, typename = image_dtype_limit<T>>
void copy_image_mat_impl(const ImageMat<T>& src_mat, ImageMat<T>& dst_mat) {
    const T* src_ptr = src_mat.get_data_ptr();
    T*       dst_ptr = dst_mat.get_data_ptr();

    int data_size = src_mat.get_element_num();
    for (int i = 0; i < data_size; ++i) {
        dst_ptr[i] = value_op<T, op>(src_ptr[i], dst_ptr[i]);
    }
}
}   // namespace internal

template<class T, typename>
Status::ErrorCode copy_image_mat(const ImageMat<T>& src_mat, ImageMat<T>& dst_mat, ValueOpKind op) {
    if (src_mat.empty()) {
        LOG_ERROR("input mat is empty which is invalid....");
        return Status::ErrorCode::InvalidMatShape;
    }

    if (!src_mat.compare_shape(dst_mat)) {
        return Status::ErrorCode::MatShapeMismatch;
    }
    if (src_mat.get_layout() != dst_mat.get_layout()) {
        LOG_ERROR("sorry...we expecte two mat have same layout!");
        return Status::ErrorCode::MatLayoutMismath;
    }

    if (op == ValueOpKind::ONLY_COPY) {
        internal::copy_image_mat_impl<T, ValueOpKind::ONLY_COPY>(src_mat, dst_mat);
    } else if (op == ValueOpKind::COPY_ZERO_TRANSPARENT) {
        internal::copy_image_mat_impl<T, ValueOpKind::COPY_ZERO_TRANSPARENT>(src_mat, dst_mat);
    } else if (op == ValueOpKind::ADD) {
        internal::copy_image_mat_impl<T, ValueOpKind::ADD>(src_mat, dst_mat);
    } else if (op == ValueOpKind::AVERAGE) {
        internal::copy_image_mat_impl<T, ValueOpKind::AVERAGE>(src_mat, dst_mat);
    } else if (op == ValueOpKind::DIFFERENCE) {
        internal::copy_image_mat_impl<T, ValueOpKind::DIFFERENCE>(src_mat, dst_mat);
    } else if (op == ValueOpKind::SUBSTRACT) {
        internal::copy_image_mat_impl<T, ValueOpKind::SUBSTRACT>(src_mat, dst_mat);
    } else if (op == ValueOpKind::MULTIPLY) {
        internal::copy_image_mat_impl<T, ValueOpKind::MULTIPLY>(src_mat, dst_mat);
    } else if (op == ValueOpKind::DIVIDE) {
        internal::copy_image_mat_impl<T, ValueOpKind::DIVIDE>(src_mat, dst_mat);
    } else if (op == ValueOpKind::AND) {
        internal::copy_image_mat_impl<T, ValueOpKind::AND>(src_mat, dst_mat);
    } else if (op == ValueOpKind::OR) {
        internal::copy_image_mat_impl<T, ValueOpKind::OR>(src_mat, dst_mat);
    } else if (op == ValueOpKind::XOR) {
        internal::copy_image_mat_impl<T, ValueOpKind::XOR>(src_mat, dst_mat);
    } else if (op == ValueOpKind::MIN) {
        internal::copy_image_mat_impl<T, ValueOpKind::MIN>(src_mat, dst_mat);
    } else if (op == ValueOpKind::MAX) {
        internal::copy_image_mat_impl<T, ValueOpKind::MAX>(src_mat, dst_mat);
    } else {
        LOG_ERROR("unsupported op type...");
        return Status::ErrorCode::UnsupportedValueOp;
    }
    return Status::Ok;
}
template Status::ErrorCode copy_image_mat<uint8_t>(const ImageMat<uint8_t>& src_mat,
                                                   ImageMat<uint8_t>& dst_mat, ValueOpKind op);

template Status::ErrorCode copy_image_mat<uint16_t>(const ImageMat<uint16_t>& src_mat,
                                                    ImageMat<uint16_t>& dst_mat, ValueOpKind op);

template Status::ErrorCode copy_image_mat<uint32_t>(const ImageMat<uint32_t>& src_mat,
                                                    ImageMat<uint32_t>& dst_mat, ValueOpKind op);

template Status::ErrorCode copy_image_mat<float>(const ImageMat<float>& src_mat,
                                                 ImageMat<float>& dst_mat, ValueOpKind op);

}   // namespace mat_ops
}   // namespace core
}   // namespace fish