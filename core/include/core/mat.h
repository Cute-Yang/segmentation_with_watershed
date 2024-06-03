
#pragma once
#include "common/fishdef.h"
#include "core/base.h"
#include "utils/logging.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <ostream>
#include <type_traits>

namespace fish {
namespace core {
namespace mat {
using namespace fish::core::base;
template<class T> struct TypeNameParser { static constexpr char name[] = ""; };
template<> struct TypeNameParser<uint8_t> { static constexpr char name[] = "uin8_t"; };
template<> struct TypeNameParser<int8_t> { static constexpr char name[] = "int8_t"; };

template<> struct TypeNameParser<uint16_t> { static constexpr char name[] = "uint16_t"; };
template<> struct TypeNameParser<int16_t> { static constexpr char name[] = "int16_t"; };

template<> struct TypeNameParser<uint32_t> { static constexpr char name[] = "uint32_t"; };
template<> struct TypeNameParser<int32_t> { static constexpr char name[] = "int32_t"; };

template<> struct TypeNameParser<uint64_t> { static constexpr char name[] = "uin64_t"; };
template<> struct TypeNameParser<int64_t> { static constexpr char name[] = "int64_t"; };

template<> struct TypeNameParser<float> { static constexpr char name[] = "float32"; };
template<> struct TypeNameParser<double> { static constexpr char name[] = "float64"; };



template<class T> struct IntegerTypeRequire {
    static constexpr bool value =
        std::is_same_v<T, std::uint8_t> || std::is_same_v<T, std::int8_t> ||
        std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::int16_t> ||
        std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, std::uint64_t> || std::is_same_v<T, std::int64_t>;
};

template<class T> struct CharTypeRequire {
    static constexpr bool value = std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>;
};

template<class T> struct ImageTypeRequire {
    // byte/short/float image!
    static constexpr bool value = std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                                  std::is_same_v<T, uint32_t> || std::is_same_v<T, float>;
};

template<class T> struct Image16BitRequire {
    static constexpr bool value = std::is_same_v<T, uint16_t>;
};

template<class T> struct FloatTypeRequire {
    static constexpr bool value =
        std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, fish::float16_t>;
};

template<class T> struct NumericTypeRequire {
    static constexpr bool value = IntegerTypeRequire<T>::value || FloatTypeRequire<T>::value;
};

// define the layout!
enum class MatMemLayout : uint8_t { LayoutLeft = 0, LayoutRight = 1 };
template<class T> using dtype_limit = std::enable_if_t<NumericTypeRequire<T>::value, T>;

template<class T> using image_dtype_limit = std::enable_if_t<ImageTypeRequire<T>::value, T>;

// the default memory order is layout right!
template<class T, typename = dtype_limit<T>> class ImageMat {
private:
    // dimension
    int height;     // do
    int width;      // d1
    int channels;   // d2

    // stride
    int stride_h;
    int stride_w;
    int stride_c;

    MatMemLayout layout;
    bool         own_data;

    T* data_ptr;

    void init_stride() noexcept {
        if (layout == MatMemLayout::LayoutRight) {
            stride_c = 1;
            stride_w = channels;
            stride_h = channels * width;
        } else {
            stride_h = 1;
            stride_w = height;
            stride_c = height * width;
        }
    }

    void check_access_index(int d0, int d1, int d2) const {
        if (d0 >= height || d1 >= width || d2 >= channels) {
            LOG_ERROR("got invalid index ({},{},{}),but shape is ({},{},{})",
                      d0,
                      d1,
                      d2,
                      height,
                      width,
                      channels);
            // throw FishException(ErrorCode::InvallidMatIndex,
            //                     "mat index out of range",
            //                     __FILE__,
            //                     FISH_FUNC,
            //                     __LINE__);
        }
    }

    void set_mat_empty() {
        data_ptr = nullptr;
        height   = 0;
        width    = 0;
        channels = 0;
        stride_h = 0;
        stride_w = 0;
        stride_c = 0;
    }

    bool check_dimension() {
        if (height == 0 || width == 0 || channels == 0) {
            LOG_ERROR("the height:{} widht:{} channels:{} maybe have invalid value",
                      height,
                      width,
                      channels);
            return false;
        }
        return true;
    }

    size_t compute_allocate_bytes() { return height * width * channels * sizeof(T); }

public:
    ImageMat(int height_, int width_, int channels_,
             MatMemLayout layout_ = MatMemLayout::LayoutRight)
        : height(height_)
        , width(width_)
        , channels(channels_)
        , layout(layout_)
        , own_data(true) {
        if (!check_dimension()) {
            set_mat_empty();
            return;
        }
        // avoid c++ apply init...
        size_t allocate_bytes = compute_allocate_bytes();
        data_ptr              = reinterpret_cast<T*>(malloc(allocate_bytes));
        init_stride();
        if (data_ptr == nullptr) {
            LOG_ERROR("allocate byte {} fail,so construct an empty mat...", allocate_bytes);
            set_mat_empty();
        }
    }

    ImageMat()
        : height(0)
        , width(0)
        , channels(0)
        , own_data(true)
        , data_ptr(nullptr) {
        init_stride();
    }

    ImageMat(int height_, int width_, int channels_, T* buf_,
             MatMemLayout layout_ = MatMemLayout::LayoutRight, bool copy = true)
        : height(height_)
        , width(width_)
        , channels(channels_)
        , layout(layout_) {
        if (!check_dimension()) {
            set_mat_empty();
            return;
        }
        if (buf_ == nullptr) {
            LOG_ERROR("the given buf is an invalid pointer,so we just return an empty Mat!");
            set_mat_empty();
            return;
        }
        if (!copy) {
            data_ptr = buf_;
            LOG_INFO("Matrix do not have the owner ship of data,and wlll do noting while destroy!");
            own_data = false;
        } else {
            size_t allocate_bytes = compute_allocate_bytes();
            data_ptr              = reinterpret_cast<T*>(malloc(allocate_bytes));
            if (data_ptr == nullptr) {
                LOG_ERROR("fail to allocate memory with {} bytes,set mat to be empty!",
                          allocate_bytes);
                set_mat_empty();
            } else {
                LOG_INFO("copy data from {} to {}",
                         reinterpret_cast<uintptr_t>(buf_),
                         reinterpret_cast<uintptr_t>(data_ptr));
                std::copy(buf_, buf_ + height * width * channels, data_ptr);
            }
            own_data = true;
        }
        check_dimension();
        init_stride();
    }

    ImageMat(const ImageMat<T>& rhs)
        : height(rhs.height)
        , width(rhs.width)
        , channels(rhs.channels)
        , layout(rhs.layout)
        , own_data(true) {
        size_t allocate_bytes = compute_allocate_bytes();
        data_ptr              = reinterpret_cast<T*>(malloc(allocate_bytes));
        if (data_ptr == nullptr) {
            LOG_ERROR("fail to allocate,so construct an empty mat!");
            set_mat_empty();
        } else {
            std::copy(rhs.data_ptr, rhs.data_ptr + height * width * channels, data_ptr);
        }
        init_stride();
    }

    // must add noexcept!
    ImageMat(ImageMat<T>&& rhs) noexcept
        : height(rhs.height)
        , width(rhs.width)
        , channels(rhs.channels)
        , layout(rhs.layout)
        , data_ptr(rhs.data_ptr)
        , own_data(rhs.own_data) {
        rhs.data_ptr = nullptr;
        rhs.set_mat_empty();
        if (!check_dimension()) {
            set_mat_empty();
        }
        init_stride();
    }

    ImageMat<T>& operator=(const ImageMat<T>&) = delete;
    ImageMat<T>& operator=(ImageMat<T>&&)      = delete;

    void set_zero() { std::fill(data_ptr, data_ptr + height * width * channels, 0); }

    bool not_empty() const { return height > 0 && width > 0 && channels > 0; }

    bool empty() const { return height == 0 || width == 0 || channels == 0; }

    ~ImageMat() { release_mat(); }

    constexpr T get_dtype_min() { return std::numeric_limits<T>::lowest(); }

    constexpr T get_type_max() { return std::numeric_limits<T>::max(); }


    void swap(ImageMat<T>& rhs) noexcept {
        // firstly,copy current source to a temp space!
        T*           temp_ptr      = data_ptr;
        int          temp_height   = height;
        int          temp_width    = width;
        int          temp_channels = channels;
        MatMemLayout temp_layout   = layout;
        data_ptr                   = rhs.data_ptr;
        height                     = rhs.height;
        width                      = rhs.width;
        channels                   = rhs.channels;
        layout                     = rhs.layout;
        init_stride();
        rhs.data_ptr = temp_ptr;
        rhs.height   = temp_height;
        rhs.width    = temp_width;
        rhs.channels = temp_channels;
        rhs.layout   = temp_layout;
        rhs.init_stride();
    }

    void release_mat() noexcept {
        if (own_data && data_ptr != nullptr) {
            // LOG_INFO("free data buf at 0x{:x}....", reinterpret_cast<size_t>(data_ptr));
            free(data_ptr);
            set_mat_empty();
        }
    }

    size_t get_element_num() const noexcept { return height * width * channels; }

    size_t get_nbytes() { return get_element_num() * sizeof(T); }

    T*       get_data_ptr() { return data_ptr; }
    const T* get_data_ptr() const { return data_ptr; }

    int get_height() const noexcept { return height; }

    int get_width() const noexcept { return width; }

    int get_channels() const noexcept { return channels; }

    MatMemLayout get_layout() const noexcept { return layout; }

    FISH_ALWAYS_INLINE T& at(int i0, int i1, int i2) {
        check_access_index(i0, i1, i2);
        return data_ptr[i0 * stride_h + i1 * stride_w + i2 * stride_c];
    }

    FISH_ALWAYS_INLINE const T& at(int i0, int i1, int i2) const {
        check_access_index(i0, i1, i2);
        return data_ptr[i0 * stride_h + i1 * stride_w + i2 * stride_c];
    }

    FISH_ALWAYS_INLINE T& operator()(int i0, int i1, int i2) {
        return data_ptr[i0 * stride_h + i1 * stride_w + i2 * stride_c];
    }

    FISH_ALWAYS_INLINE T& operator()(int i0, int i1) {
        return data_ptr[i0 * stride_h + i1 * stride_w];
    }

    FISH_ALWAYS_INLINE const T& operator()(int i0, int i1, int i2) const {
        return data_ptr[i0 * stride_h + i1 * stride_w + i2 * stride_c];
    }

    FISH_ALWAYS_INLINE const T& operator()(int i0, int i1) const {
        return data_ptr[i0 * stride_h + i1 * stride_w];
    }

    bool shape_equal(int h, int w, int c) { return (height == h && width == w && channels == c); }

    bool resize(int h, int w, int c, bool always_allocate = false) {
        // if the shape is same,do not do anything!
        if (h == height && w == width && c == channels) {
            return true;
        }
        bool buffer_is_enough = (h * w * c) <= (height * width * channels);
        height                = h;
        width                 = w;
        channels              = c;
        // while buffer is enough and not speicfy aloway allocate,just change the meta data..
        if (buffer_is_enough && !always_allocate) {
            LOG_INFO("the prev buffer is enough,so we will not allocate new memory...");
            init_stride();
            return true;
        }
        if (data_ptr != nullptr) {
            // show with hex...
            LOG_INFO("free buffer at 0x{:x}", reinterpret_cast<uintptr_t>(data_ptr));
            free(data_ptr);
        }
        // replace the newe/delete to malloc/free to avoid compiler do intialize...
        size_t allocate_bytes = compute_allocate_bytes();
        data_ptr              = reinterpret_cast<T*>(malloc(allocate_bytes));
        if (data_ptr == nullptr) {
            LOG_ERROR("fail to re allocate memory with bytes {}", allocate_bytes);
            set_mat_empty();
            return false;
        }
        init_stride();
        return true;
    }

    bool reshape(int h, int w, int c) {
        if (h * w * c != height * width * channels) {
            LOG_ERROR("can not reshape mat from ({},{},{}) to ({},{},{})",
                      height,
                      width,
                      channels,
                      h,
                      w,
                      c);
            return false;
        }
        height   = h;
        width    = w;
        channels = c;
        init_stride();
        return true;
    }

    template<class X, typename = dtype_limit<X>> bool compare_shape(const ImageMat<X>& rhs) const {
        if (height != rhs.get_height() || width != rhs.get_width() ||
            channels != rhs.get_channels()) {
            LOG_WARN("dimenions mismatch,left image has shape ({},{},{}) right image has shape "
                     "({},{},{})",
                     height,
                     width,
                     channels,
                     rhs.get_height(),
                     rhs.get_width(),
                     rhs.get_channels());
            return false;
        }
        return true;
    }

    void set_layout(MatMemLayout layout_) {
        layout = layout_;
        init_stride();
    }

    FISH_ALWAYS_INLINE void set_value_f(int dh, int dw, int dc, float value) {
        int index = dh * stride_h + dw * stride_w + dc * stride_c;
        if constexpr (FloatTypeRequire<T>::value) {
            data_ptr[index] = value;
        } else {
            constexpr T     type_min   = std::numeric_limits<T>::min();
            constexpr T     type_max   = std::numeric_limits<T>::max();
            constexpr float type_min_f = static_cast<float>(type_min);
            constexpr float type_max_f = static_cast<float>(type_max);
            if (value < type_min_f) FISH_UNLIKELY_STD {
                    data_ptr[index] = type_min;
                }
            else if (value >= type_min_f)
                FISH_UNLIKELY_STD {
                    data_ptr[index] = type_max;
                }
            else
                FISH_LIKELY_STD {
                    data_ptr[index] = static_cast<T>(value + 0.5f);
                }
        }
    }

    FISH_ALWAYS_INLINE void set_value_f(int idx, float value) {
        if constexpr (FloatTypeRequire<T>::value) {
            data_ptr[idx] = value;
        } else {
            constexpr T     type_min   = std::numeric_limits<T>::min();
            constexpr T     type_max   = std::numeric_limits<T>::max();
            constexpr float type_min_f = static_cast<float>(type_min);
            constexpr float type_max_f = static_cast<float>(type_max);
            if (value < type_min_f) FISH_UNLIKELY_STD {
                    data_ptr[idx] = type_min;
                }
            else if (value >= type_min_f)
                FISH_UNLIKELY_STD {
                    data_ptr[idx] = type_max;
                }
            else
                FISH_LIKELY_STD {
                    data_ptr[idx] = static_cast<T>(value + 0.5f);
                }
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const ImageMat<T>& mat) {
        constexpr int min_require_height   = 10;
        constexpr int min_require_width    = 10;
        constexpr int min_require_channels = 3;
        os << "*********Mat*************"
           << "\n";
        int display_height   = FISH_MIN(min_require_height, mat.height);
        int display_width    = FISH_MIN(min_require_width, mat.width);
        int display_channels = FISH_MIN(min_require_channels, mat.channels);
        os << "ImageMat with height=" << mat.height << " width=" << mat.width
           << " channels=" << mat.channels << "\n";

        os << "datas:"
           << "\n";

        for (int h = 0; h < display_height; ++h) {
            for (int w = 0; w < display_width; ++w) {
                for (int c = 0; c < display_channels - 1; ++c) {
                    if constexpr (CharTypeRequire<T>::value) {
                        os << std::setw(8) << static_cast<int>(mat.at(h, w, c)) << " ";
                    } else {
                        os << std::setw(8) << mat.at(h, w, c) << " ";
                    }
                }
                if constexpr (CharTypeRequire<T>::value) {
                    os << static_cast<int>(mat.at(h, w, display_channels - 1));
                } else {
                    os << std::setw(8) << mat.at(h, w, display_channels - 1);
                }
                if (w < mat.width - 1) {
                    os << ",";
                }
            }
            if (display_width < mat.width) {
                os << ".....";
            }
            int tail_width_start = FISH_MAX(display_width, mat.width - display_width);
            for (int w = tail_width_start; w < mat.width; ++w) {
                for (int c = 0; c < display_channels - 1; ++c) {
                    if constexpr (CharTypeRequire<T>::value) {
                        os << std::setw(8) << static_cast<int>(mat.at(h, w, c)) << " ";
                    } else {
                        os << std::setw(8) << mat.at(h, w, c) << " ";
                    }
                }
                if constexpr (CharTypeRequire<T>::value) {
                    os << std::setw(8) << static_cast<int>(mat.at(h, w, display_channels - 1));
                } else {
                    os << std::setw(8) << mat.at(h, w, display_channels - 1);
                }
                if (w < mat.width - 1) {
                    os << ",";
                }
            }
            if (h < mat.height - 1) {
                os << "\n";
            }
        }

        int height_tail_start = FISH_MAX(display_height, mat.height - display_height);
        if (display_height < mat.height) {
            os << "....." << '\n';
        }

        for (int h = height_tail_start; h < mat.height; ++h) {
            for (int w = 0; w < display_width; ++w) {
                for (int c = 0; c < display_channels - 1; ++c) {
                    if constexpr (CharTypeRequire<T>::value) {
                        os << std::setw(8) << static_cast<int>(mat.at(h, w, c)) << " ";
                    } else {
                        os << std::setw(8) << mat.at(h, w, c) << " ";
                    }
                }
                if constexpr (CharTypeRequire<T>::value) {
                    os << std::setw(8) << static_cast<int>(mat.at(h, w, display_channels - 1));
                } else {
                    os << std::setw(8) << mat.at(h, w, display_channels - 1);
                }
                if (w < mat.width - 1) {
                    os << ",";
                }
            }
            if (display_width < mat.width) {
                os << ".....";
            }
            int tail_width_start = FISH_MAX(display_width, mat.width - display_width);
            for (int w = tail_width_start; w < mat.width; ++w) {
                for (int c = 0; c < display_channels - 1; ++c) {
                    if constexpr (CharTypeRequire<T>::value) {
                        os << std::setw(8) << static_cast<int>(mat.at(h, w, c)) << " ";
                    } else {
                        os << std::setw(8) << mat.at(h, w, c) << " ";
                    }
                }
                if constexpr (CharTypeRequire<T>::value) {
                    os << std::setw(8) << static_cast<int>(mat.at(h, w, display_channels - 1));
                } else {
                    os << std::setw(8) << mat.at(h, w, display_channels - 1);
                }
                if (w < mat.width - 1) {
                    os << ",";
                }
            }
            if (h < mat.height - 1) {
                os << "\n";
            }
        }
        os << "\n";
        os << "**********************"
           << "\n";
        return os;
    }
};

template<class T1, class T2, typename = dtype_limit<T1>, typename = dtype_limit<T2>>
Status::ErrorCode convert_mat(const ImageMat<T1>& input_mat, ImageMat<T2>& output_mat) {
    if (!input_mat.compare_shape(output_mat)) {
        LOG_ERROR(
            "the input_mat and output_mat have different shape,so we can't convert between them!");
        return Status::ErrorCode::MatShapeMismatch;
    }
    if (input_mat.get_layout() != output_mat.get_layout()) {
        return Status ::ErrorCode::MatLayoutMismath;
    }
    LOG_INFO(
        "convert type from type {} to type {}", TypeNameParser<T1>::name, TypeNameParser<T2>::name);
    const T1*      src_ptr        = input_mat.get_data_ptr();
    T2*            dst_ptr        = output_mat.get_data_ptr();
    size_t         data_size      = input_mat.get_element_num();
    constexpr bool have_same_type = std::is_same_v<T1, T2>;
    if constexpr (have_same_type) {
        LOG_INFO("the input mat and output mat have same shape,just do copy!");
        std::copy(src_ptr, src_ptr + data_size, dst_ptr);
    } else {
        // if output mat is float type,just copy it,do a simple type cast!
        if constexpr (FloatTypeRequire<T2>::value) {
            for (size_t i = 0; i < data_size; ++i) {
                dst_ptr[i] = static_cast<T2>(src_ptr[i]);
            }
        } else {
            // T2 is a integer type
            constexpr T2 dst_type_min = std::numeric_limits<T2>::min();
            constexpr T2 dst_type_max = std::numeric_limits<T2>::max();

            constexpr T1 src_type_min = std::numeric_limits<T1>::min();
            constexpr T1 src_type_max = std::numeric_limits<T1>::max();
            if constexpr (IntegerTypeRequire<T1>::value) {
                constexpr bool dst_type_is_larger =
                    (dst_type_min <= src_type_min && dst_type_max >= src_type_max);
                if (dst_type_is_larger) {
                    for (size_t i = 0; i < data_size; ++i) {
                        dst_ptr[i] = src_ptr[i];
                    }
                } else {
                    for (size_t i = 0; i < data_size; ++i) {
                        if (src_ptr[i] <= dst_type_min) {
                            dst_ptr[i] = dst_type_min;
                        } else if (src_ptr[i] >= dst_type_max) {
                            dst_ptr[i] = dst_type_max;
                        } else {
                            dst_ptr[i] = src_ptr[i];
                        }
                    }
                }
            } else {
                // src is float,dst is integer!
                constexpr T1 dst_type_min_f  = static_cast<T1>(dst_type_min);
                constexpr T1 dst_type_max_f  = static_cast<T1>(dst_type_max);
                constexpr T1 zero_dot_five_f = static_cast<T1>(0.5);
                for (size_t i = 0; i < data_size; ++i) {
                    if (src_ptr[i] <= dst_type_min_f) FISH_UNLIKELY_STD {
                            dst_ptr[i] = dst_type_min;
                        }
                    else if (src_ptr[i] >= dst_type_max_f)
                        FISH_UNLIKELY_STD {
                            dst_ptr[i] = dst_type_max;
                        }
                    else
                        FISH_LIKELY_STD {
                            dst_ptr[i] = static_cast<T2>(src_ptr[i] + zero_dot_five_f);
                        }
                }
            }
        }
    }
    return Status::ErrorCode::Ok;
}

template<class T1, class T2, typename = image_dtype_limit<T1>, typename = image_dtype_limit<T2>>
Status::ErrorCode convert_image(const ImageMat<T1>& input_mat, ImageMat<T2>& output_mat) {
    if (!input_mat.compare_shape(output_mat)) {
        LOG_ERROR("sorry,we only convert the type with same shape....,but got "
                  "mismatch mats...");
        return Status::ErrorCode::MatShapeMismatch;
    }
    if (input_mat.get_layout() != output_mat.get_layout()) {
        LOG_ERROR("sorry,the layout of two mats mismatch....");
        return Status::ErrorCode::MatLayoutMismath;
    }

    LOG_INFO("convert mat from {} to {} and will scale value with limit!",
             TypeNameParser<T1>::name,
             TypeNameParser<T2>::name);
    const T1*      input_ptr       = input_mat.get_data_ptr();
    T2*            output_ptr      = output_mat.get_data_ptr();
    int            data_size       = input_mat.get_element_num();
    constexpr bool input_is_float  = FloatTypeRequire<T1>::value;
    constexpr bool output_is_float = FloatTypeRequire<T2>::value;
    constexpr bool same_type       = std::is_same_v<T1, T2>;

    constexpr T1    input_min   = std::numeric_limits<T1>::min();
    constexpr float input_min_f = static_cast<float>(input_min);
    constexpr T1    input_max   = std::numeric_limits<T1>::max();
    constexpr float input_max_f = static_cast<float>(input_max);

    constexpr T2    output_min   = std::numeric_limits<T2>::min();
    constexpr float output_min_f = static_cast<float>(output_min);
    constexpr T2    output_max   = std::numeric_limits<T2>::max();
    constexpr float output_max_f = static_cast<float>(output_max);

    constexpr float strip_constant = 0.5f;

    if constexpr (same_type) {
        std::copy(input_ptr, input_ptr + data_size, output_ptr);
    } else {
        if constexpr (input_is_float) {
            if (output_is_float) {
                for (int i = 0; i < data_size; ++i) {
                    output_ptr[i] = input_ptr[i];
                }
            } else {
                for (int i = 0; i < data_size; ++i) {
                    if (input_ptr[i] >= output_max_f) [[unlikely]] {
                        output_ptr[i] = output_max;
                    } else if (input_ptr[i] <= output_min_f) [[unlikely]] {
                        output_ptr[i] = output_min;
                    } else [[likely]] {
                        output_ptr[i] = static_cast<T2>(input_ptr[i] + strip_constant);
                    }
                }
            }
        } else {
            if constexpr (output_is_float) {
                // just do cast,do worry overflow!
                for (int i = 0; i < data_size; ++i) {
                    output_ptr[i] = static_cast<float>(input_ptr[i]);
                }
            } else {
                // we should do scale....
                for (int i = 0; i < data_size; ++i) {
                    constexpr float scale        = output_max_f / input_max_f;
                    float           scaled_value = static_cast<float>(input_ptr[i]) * scale;
                    output_ptr[i]                = static_cast<T2>(scaled_value + strip_constant);
                }
            }
        }
    }
    return Status::ErrorCode::Ok;
}


template<class T, typename = dtype_limit<T>> class Mat {
    int rows;
    int cols;

    int stride_d0;
    int stride_d1;

    MatMemLayout layout;
    bool         own_data;
    T*           data_ptr;


    void init_stride() noexcept {
        if (layout == MatMemLayout::LayoutRight) {
            stride_d0 = cols;
            stride_d1 = 1;
        } else {
            stride_d0 = 1;
            stride_d1 = rows;
        }
    }

    void set_mat_empty() {
        data_ptr  = nullptr;
        rows      = 0;
        cols      = 0;
        stride_d0 = 0;
        stride_d1 = 0;
    }

    size_t compute_allocate_bytes() { return rows * cols * sizeof(T); }

public:
    Mat(int rows_, int cols_, MatMemLayout layout_ = MatMemLayout::LayoutRight)
        : rows(rows_)
        , cols(cols_)
        , layout(layout_) {
        if (rows == 0 || cols == 0) {
            set_mat_empty();
            return;
        }
        size_t allocate_bytes = compute_allocate_bytes();
        data_ptr              = reinterpret_cast<T*>(malloc(allocate_bytes));
        if (data_ptr == nullptr) {
            LOG_ERROR("fail to allocate memory with {} bytes", allocate_bytes);
            set_mat_empty();
        }
        init_stride();
    }

    Mat(int rows_, int cols_, T* buf_, MatMemLayout layout_ = MatMemLayout::LayoutRight,
        bool copy = true)
        : rows(rows_)
        , cols(cols_)
        , layout(layout_) {
        if (rows == 0 || cols == 0) {
            LOG_ERROR("invalid shape({},{}),so construct an empty mat...", rows, cols);
            set_mat_empty();
            return;
        }
        if (buf_ == nullptr) {
            LOG_ERROR("buf is an invalid pointer!");
            set_mat_empty();
            return;
        }
        if (!copy) {
            LOG_INFO("Mat is just a view,do not take the ownership!");
            data_ptr = buf_;
            own_data = false;
        } else {
            size_t allocate_bytes = compute_allocate_bytes();
            data_ptr              = reinterpret_cast<T*>(allocate_bytes);
            if (data_ptr == nullptr) {
                set_mat_empty();
            } else {
                std::copy(buf_, buf_ + rows * cols, data_ptr);
            }
            own_data = true;
        }
        init_stride();
    }

    Mat(const Mat<T>& rhs)
        : rows(rhs.rows)
        , cols(rhs.cols)
        , layout(rhs.layout)
        , own_data(true) {
        size_t allocate_bytes = compute_allocate_bytes();
        data_ptr              = reinterpret_cast<T*>(malloc(allocate_bytes));
        if (data_ptr == nullptr) {
            set_mat_empty();
            LOG_ERROR("fail allocate memory,so construct empty mat...");
        } else {
            std::copy(rhs.data_ptr, rhs.data_ptr + rows * cols, data_ptr);
        }

        init_stride();
    }

    Mat(Mat<T>&& rhs)
        : rows(rhs.rows)
        , cols(rhs.cols)
        , data_ptr(rhs.data_ptr)
        , layout(rhs.layout)
        , own_data(rhs.own_data) {
        rhs.set_mat_empty();
        init_stride();
    }
    ~Mat() {
        if (own_data && data_ptr != nullptr) {
            free(data_ptr);
        }
        set_mat_empty();
    }

    const T& operator()(int d0, int d1) const { return data_ptr[d0 * stride_d0 + d1 * stride_d1]; }

    T& operator()(int d0, int d1) { return data_ptr[d0 * stride_d0 + d1 * stride_d1]; }

    int get_rows() const noexcept { return rows; }

    int get_cols() const noexcept { return cols; }

    T*       get_data_ptr() noexcept { return data_ptr; }
    const T* get_data_ptr() const noexcept { return data_ptr; }

    int get_element_num() const noexcept { return rows * cols; }
};

enum ImageDirectionKind : uint8_t { Width = 0, Height = 1 };

template<class T> struct GenericCoordinate2d {
    T x;
    T y;
    GenericCoordinate2d(int x_, int y_)
        : x(x_)
        , y(y_) {}
    GenericCoordinate2d()
        : x(0)
        , y(0) {}
    GenericCoordinate2d(const GenericCoordinate2d& rhs)
        : x(rhs.x)
        , y(rhs.y) {}
    GenericCoordinate2d(GenericCoordinate2d&& rhs) noexcept
        : x(rhs.x)
        , y(rhs.y) {}
    GenericCoordinate2d& operator=(const GenericCoordinate2d& rhs) {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }

    GenericCoordinate2d& operator=(GenericCoordinate2d&& rhs) {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }

    void set_coor(int x_, int y_) noexcept {
        x = x_;
        y = y_;
    }

    bool operator<(const GenericCoordinate2d<T>& rhs) const {
        if (x < rhs.x) {
            return true;
        } else if (x > rhs.x) {
            return false;
        } else {
            return y < rhs.y;
        }
    }

    bool operator>(const GenericCoordinate2d<T>& rhs) const {
        if (x > rhs.x) {
            return true;
        } else if (x < rhs.x) {
            return false;
        } else {
            return y > rhs.y;
        }
    }


    bool operator==(const GenericCoordinate2d<T>& rhs) const noexcept {
        return x == rhs.x && y == rhs.y;
    }

    bool operator!=(const GenericCoordinate2d<T>& rhs) const noexcept {
        return x != rhs.x || y != rhs.y;
    }
};

using Coordinate2d    = GenericCoordinate2d<int>;
using Coordinate2df32 = GenericCoordinate2d<float>;
using Coordinate2df64 = GenericCoordinate2d<double>;

struct Rectangle {
    int x;
    int y;
    int height;
    int width;

    Rectangle(int x_, int y_, int height_, int width_)
        : x(x_)
        , y(y_)
        , height(height_)
        , width(width_) {}
    Rectangle()
        : x(0)
        , y(0)
        , height(0)
        , width(0) {}

    Rectangle(const Rectangle& rhs)
        : x(rhs.x)
        , y(rhs.y)
        , height(rhs.height)
        , width(rhs.width) {}
};

// compute the clip value from t1 - t2
template<class T, typename = dtype_limit<T>> T compute_clip_value(float value) {
    if constexpr (FloatTypeRequire<T>::value) {
        return value;
        // do not do any clip for float value!
    } else {
        constexpr T     type_min_value   = std::numeric_limits<T>::min();
        constexpr T     type_max_value   = std::numeric_limits<T>::max();
        constexpr float type_min_value_f = static_cast<float>(type_min_value);
        constexpr float type_max_value_f = static_cast<float>(type_max_value);
        if (value < type_min_value_f) {
            return type_min_value;
        } else if (value > type_max_value_f) {
            return type_max_value;
        } else {
            return static_cast<T>(value + 0.5f);
        }
    }
}
}   // namespace mat
}   // namespace core
}   // namespace fish