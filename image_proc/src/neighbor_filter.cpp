#include "image_proc/neighbor_filter.h"
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "utils/logging.h"
#include <cstdlib>
#include <limits>

namespace fish {
namespace image_proc {
namespace neighbor_filter {
namespace internal {
template<class T, typename = dtype_limit<T>> struct SumTypeHelper { using type = T; };

template<> struct SumTypeHelper<int8_t> { using type = int32_t; };

template<> struct SumTypeHelper<uint8_t> { using type = uint32_t; };

template<> struct SumTypeHelper<int16_t> { using type = int32_t; };

template<> struct SumTypeHelper<uint16_t> { using type = uint32_t; };

template<> struct SumTypeHelper<int32_t> { using type = int64_t; };

template<> struct SumTypeHelper<uint32_t> { using type = uint64_t; };

template<> struct SumTypeHelper<int64_t> { using type = int64_t; };

template<> struct SumTypeHelper<uint64_t> { using type = uint64_t; };


template<class T, typename = dtype_limit<T>>
FISH_ALWAYS_INLINE T compute_median_value(T* triple_datas) {
    if (triple_datas[0] > triple_datas[1]) {
        T temp          = triple_datas[0];
        triple_datas[0] = triple_datas[1];
        triple_datas[1] = temp;
    }

    if (triple_datas[0] > triple_datas[2]) {
        T temp          = triple_datas[0];
        triple_datas[0] = triple_datas[2];
        triple_datas[2] = temp;
    }

    // now the first value is min
    T value = FISH_MIN(triple_datas[1], triple_datas[2]);
    return value;
}

template<class T, NeighborFilterType filter_type, bool is_edge>
T neighbor_filter_3x3_detail(T* window_datas, int binary_count) {
    constexpr T type_max_value   = std::numeric_limits<T>::max();
    using SumType                = typename SumTypeHelper<T>::type;
    constexpr T backgroup_value  = 0;
    constexpr T foreground_value = 255;
    T           value{0};
    if constexpr (filter_type == NeighborFilterType::BLUR_MORE) {
        value = (window_datas[0] + window_datas[1] + window_datas[2] + window_datas[3] +
                 window_datas[4] + window_datas[5] + window_datas[6] + window_datas[7] +
                 window_datas[8] + 4) /
                9;
    } else if constexpr (filter_type == NeighborFilterType::FIND_EDGES) {
        SumType sum_0 = window_datas[0] + 2 * window_datas[1] + window_datas[2] - window_datas[6] -
                        2 * window_datas[7] - window_datas[8];
        SumType sum_1 = window_datas[0] + 2 * window_datas[3] + window_datas[6] - window_datas[2] -
                        2 * window_datas[5] - window_datas[8];
        SumType _value = static_cast<int>(std::sqrt(sum_0 * sum_0 + sum_1 * sum_1));
        if (_value > type_max_value) [[unlikely]] {
            value = type_max_value;
        } else {
            value = _value;
        }
    } else if constexpr (filter_type == NeighborFilterType::MIN) {
        value = window_datas[0];
        for (int i = 1; i < 9; ++i) {
            value = FISH_MIN(value, window_datas[i]);
        }
    } else if constexpr (filter_type == NeighborFilterType::MAX) {
        value = window_datas[0];
        for (int i = 1; i < 9; ++i) {
            value = FISH_MAX(value, window_datas[i]);
        }
    } else if constexpr (filter_type == NeighborFilterType::ERODE) {
        // if middle value equal to backgroup,just return
        if (window_datas[4] == backgroup_value) {
            value = backgroup_value;
        } else {
            int count = 0;
            for (int i = 0; i < 9; ++i) {
                if constexpr (IntegerTypeRequire<T>::value) {
                    if (window_datas[i] == backgroup_value) {
                        ++count;
                    }
                } else {
                    constexpr float eps = 1e-5;
                    if (std::abs(window_datas[i] - backgroup_value) <= eps) {
                        ++count;
                    }
                }
            }
            if (count > binary_count) {
                value = backgroup_value;
            } else {
                value = foreground_value;
            }
        }
    } else if constexpr (filter_type == NeighborFilterType::DILATE) {
        if (window_datas[4] == foreground_value) {
            value = backgroup_value;
        } else {
            int count = 0;
            if constexpr (IntegerTypeRequire<T>::value) {
                for (int i = 0; i < 9; ++i) {
                    if (window_datas[i] == foreground_value) {
                        ++count;
                    }
                }
            } else {
                constexpr float eps = 1e-5;
                for (int i = 0; i < 9; ++i) {
                    if (std::abs(window_datas[i] - foreground_value) <= eps) {
                        ++count;
                    }
                }
            }
            if (count >= binary_count) {
                value = foreground_value;
            } else {
                value = backgroup_value;
            }
        }
    } else if constexpr (filter_type == NeighborFilterType::MEDIAN_FILTER) {
        if constexpr (is_edge) {
            value = window_datas[4];
        } else {
            T triple_datas[3];
            triple_datas[0] = compute_median_value(window_datas);
            triple_datas[1] = compute_median_value(window_datas + 3);
            triple_datas[2] = compute_median_value(window_datas + 6);
            value           = compute_median_value(triple_datas);
        }
    } else {
        value = window_datas[4];
    }
    return value;
}

template<class T, NeighborFilterType filter_type, ImageDirectionKind direction, bool pad_edges>
void neighbor_filter_1d_detail(const T* input_datas, T* output_datas, int data_size,
                               int binary_count) {
    using SumType                 = typename SumTypeHelper<T>::type;
    constexpr T    type_max_value = std::numeric_limits<T>::max();
    T              window_datas[9];
    constexpr T    background_value = 0;
    constexpr T    foreground_value = 255;
    constexpr bool width_direction  = direction == ImageDirectionKind::Width;
    if constexpr (width_direction) {
        if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
            window_datas[0] = foreground_value;
            window_datas[1] = foreground_value;
            window_datas[2] = foreground_value;
            window_datas[3] = foreground_value;
            window_datas[4] = input_datas[0];
            window_datas[5] = input_datas[1];
            window_datas[6] = foreground_value;
            window_datas[7] = foreground_value;
            window_datas[8] = foreground_value;
        } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                             (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
            window_datas[0] = background_value;
            window_datas[1] = background_value;
            window_datas[2] = background_value;
            window_datas[3] = background_value;
            window_datas[4] = input_datas[0];
            window_datas[5] = input_datas[1];
            window_datas[6] = background_value;
            window_datas[7] = background_value;
            window_datas[8] = background_value;
        } else {
            window_datas[0] = input_datas[0];
            window_datas[1] = input_datas[0];
            window_datas[2] = input_datas[1];
            window_datas[3] = input_datas[0];
            window_datas[4] = input_datas[0];
            window_datas[5] = input_datas[1];
            window_datas[6] = input_datas[0];
            window_datas[7] = input_datas[0];
            window_datas[8] = input_datas[1];
        }
    } else {
        // pad with foreground
        if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
            window_datas[0] = foreground_value;   // x-1,y-1
            window_datas[1] = foreground_value;   // x,y -1
            window_datas[2] = foreground_value;   // x + 1,y -1
            window_datas[3] = foreground_value;   // x-1,y
            window_datas[4] = input_datas[0];     // x,y
            window_datas[5] = foreground_value;   // x + 1,y
            window_datas[6] = foreground_value;   // x-1,y + 1
            window_datas[7] = input_datas[1];     // x,y+1
            window_datas[8] = foreground_value;   // x + 1,y + 1
        } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                             (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
            window_datas[0] = background_value;   // x-1,y-1
            window_datas[1] = background_value;   // x,y -1
            window_datas[2] = background_value;   // x + 1,y -1
            window_datas[3] = background_value;   // x-1,y
            window_datas[4] = input_datas[0];     // x,y
            window_datas[5] = background_value;   // x + 1,y
            window_datas[6] = background_value;   // x-1,y + 1
            window_datas[7] = input_datas[1];     // x,y+1
            window_datas[8] = background_value;   // x + 1,y + 1
        } else {
            // pad with nearest!
            window_datas[0] = input_datas[0];   // x-1,y-1
            window_datas[1] = input_datas[0];   // x,y -1
            window_datas[2] = input_datas[0];   // x + 1,y -1
            window_datas[3] = input_datas[0];   // x-1,y
            window_datas[4] = input_datas[0];   // x,y
            window_datas[5] = input_datas[0];   // x + 1,y
            window_datas[6] = input_datas[1];   // x-1,y + 1
            window_datas[7] = input_datas[1];   // x,y+1
            window_datas[8] = input_datas[1];   // x + 1,y + 1
        }
    }
    output_datas[0] = neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);

    for (int i = 1; i < data_size - 1; ++i) {
        if constexpr (width_direction) {
            if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
                window_datas[0] = foreground_value;
                window_datas[1] = foreground_value;
                window_datas[2] = foreground_value;

                window_datas[3] = input_datas[i - 1];
                window_datas[4] = input_datas[i];
                window_datas[5] = input_datas[i + 1];

                window_datas[6] = foreground_value;
                window_datas[7] = foreground_value;
                window_datas[8] = foreground_value;
            } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                                 (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
                window_datas[0] = background_value;
                window_datas[1] = background_value;
                window_datas[2] = background_value;

                window_datas[3] = input_datas[i - 1];
                window_datas[4] = input_datas[i];
                window_datas[5] = input_datas[i + 1];

                window_datas[6] = background_value;
                window_datas[7] = background_value;
                window_datas[8] = background_value;
            } else {
                window_datas[0] = input_datas[i - 1];
                window_datas[1] = input_datas[i];
                window_datas[2] = input_datas[i + 1];

                window_datas[3] = input_datas[i - 1];
                window_datas[4] = input_datas[i];
                window_datas[5] = input_datas[i + 1];

                window_datas[6] = input_datas[i - 1];
                window_datas[7] = input_datas[i];
                window_datas[8] = input_datas[i + 1];
            }
        } else {
            if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
                //重复同一个元素
                window_datas[0] = foreground_value;
                window_datas[1] = input_datas[i - 1];
                window_datas[2] = foreground_value;

                window_datas[3] = foreground_value;
                window_datas[4] = input_datas[i];
                window_datas[5] = foreground_value;

                window_datas[6] = foreground_value;
                window_datas[7] = input_datas[i + 1];
                window_datas[8] = foreground_value;
            } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                                 (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
                window_datas[0] = background_value;
                window_datas[1] = input_datas[i - 1];
                window_datas[2] = background_value;

                window_datas[3] = background_value;
                window_datas[4] = input_datas[i];
                window_datas[5] = background_value;

                window_datas[6] = background_value;
                window_datas[7] = input_datas[i + 1];
                window_datas[8] = background_value;
            } else {
                window_datas[0] = input_datas[i - 1];
                window_datas[1] = input_datas[i - 1];
                window_datas[2] = input_datas[i - 1];

                window_datas[3] = input_datas[i];
                window_datas[4] = input_datas[i];
                window_datas[5] = input_datas[i];

                window_datas[6] = input_datas[i + 1];
                window_datas[7] = input_datas[i + 1];
                window_datas[8] = input_datas[i + 1];
            }
        }
        output_datas[i] =
            neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);
    }

    // handle the last one!
    if constexpr (width_direction) {
        if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
            window_datas[0] = foreground_value;
            window_datas[1] = foreground_value;
            window_datas[2] = foreground_value;
            window_datas[3] = input_datas[data_size - 2];
            window_datas[4] = input_datas[data_size - 1];
            window_datas[5] = foreground_value;
            window_datas[6] = foreground_value;
            window_datas[7] = foreground_value;
            window_datas[8] = foreground_value;
        } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                             (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
            window_datas[0] = background_value;
            window_datas[1] = background_value;
            window_datas[2] = background_value;
            window_datas[3] = input_datas[data_size - 2];
            window_datas[4] = input_datas[data_size - 1];
            window_datas[5] = background_value;
            window_datas[6] = background_value;
            window_datas[7] = background_value;
            window_datas[8] = background_value;
        } else {
            window_datas[0] = input_datas[data_size - 2];
            window_datas[1] = input_datas[data_size - 1];
            window_datas[2] = input_datas[data_size - 1];
            window_datas[3] = input_datas[data_size - 2];
            window_datas[4] = input_datas[data_size - 1];
            window_datas[5] = input_datas[data_size - 1];
            window_datas[6] = input_datas[data_size - 2];
            window_datas[7] = input_datas[data_size - 1];
            window_datas[8] = input_datas[data_size - 1];
        }
    } else {
        // pad with foreground
        if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
            window_datas[0] = foreground_value;             // x-1,y-1
            window_datas[1] = input_datas[data_size - 2];   // x,y -1
            window_datas[2] = foreground_value;             // x + 1,y -1
            window_datas[3] = foreground_value;             // x-1,y
            window_datas[4] = input_datas[data_size - 1];   // x,y
            window_datas[5] = foreground_value;             // x + 1,y
            window_datas[6] = foreground_value;             // x-1,y + 1
            window_datas[7] = foreground_value;             // x,y+1
            window_datas[8] = foreground_value;             // x + 1,y + 1
        } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                             (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
            window_datas[0] = background_value;             // x-1,y-1
            window_datas[1] = input_datas[data_size - 2];   // x,y -1
            window_datas[2] = background_value;             // x + 1,y -1
            window_datas[3] = background_value;             // x-1,y
            window_datas[4] = input_datas[data_size - 1];   // x,y
            window_datas[5] = background_value;             // x + 1,y
            window_datas[6] = background_value;             // x-1,y + 1
            window_datas[7] = background_value;             // x,y+1
            window_datas[8] = background_value;             // x + 1,y + 1
        } else {
            // pad with nearest!
            window_datas[0] = input_datas[data_size - 2];   // x-1,y-1
            window_datas[1] = input_datas[data_size - 2];   // x,y -1
            window_datas[2] = input_datas[data_size - 2];   // x + 1,y -1
            window_datas[3] = input_datas[data_size - 1];   // x-1,y
            window_datas[4] = input_datas[data_size - 1];   // x,y
            window_datas[5] = input_datas[data_size - 1];   // x + 1,y
            window_datas[6] = input_datas[data_size - 1];   // x-1,y + 1
            window_datas[7] = input_datas[data_size - 1];   // x,y+1
            window_datas[8] = input_datas[data_size - 1];   // x + 1,y + 1
        }
    }
    output_datas[data_size - 1] =
        neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);
}

template<class T, NeighborFilterType filter_type, bool pad_edges, typename = dtype_limit<T>>
void neighbor_filter_3x3_impl(const ImageMat<T>& input_mat, ImageMat<T>& output_mat, int channel,
                              int binary_count) {
    int width  = input_mat.get_width();
    int height = input_mat.get_height();

    if (height == 1) [[unlikely]] {
        const T* input_ptr  = input_mat.get_data_ptr();
        T*       output_ptr = output_mat.get_data_ptr();
        neighbor_filter_1d_detail<T, filter_type, ImageDirectionKind::Width, pad_edges>(
            input_ptr, output_ptr, width, binary_count);
        return;
    }

    if (width == 1) [[unlikely]] {
        const T* input_ptr  = input_mat.get_data_ptr();
        T*       output_ptr = output_mat.get_data_ptr();
        neighbor_filter_1d_detail<T, filter_type, ImageDirectionKind::Height, pad_edges>(
            input_ptr, output_ptr, height, binary_count);
        return;
    }

    T           window_datas[9];
    constexpr T foreground_value = 255;
    constexpr T background_value = 0;
    if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
        window_datas[0] = foreground_value;
        window_datas[1] = foreground_value;
        window_datas[2] = foreground_value;

        window_datas[3] = foreground_value;
        window_datas[4] = input_mat(0, 0, channel);   // x=0,y=0
        window_datas[5] = input_mat(0, 1, channel);   // x=1,y=0

        window_datas[6] = foreground_value;
        window_datas[7] = input_mat(1, 0, channel);   // x=0,y=1
        window_datas[8] = input_mat(1, 1, channel);   // x=1,y=1
    } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                         (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
        window_datas[0] = background_value;
        window_datas[1] = background_value;
        window_datas[2] = background_value;

        window_datas[3] = background_value;
        window_datas[4] = input_mat(0, 0, channel);   // x=0,y=0
        window_datas[5] = input_mat(0, 1, channel);   // x=1,y=0

        window_datas[6] = background_value;
        window_datas[7] = input_mat(1, 0, channel);   // x=0,y=1
        window_datas[8] = input_mat(1, 1, channel);   // x=1,y=1
    } else {
        window_datas[0] = input_mat(0, 0, channel);
        window_datas[1] = input_mat(0, 0, channel);
        window_datas[2] = input_mat(0, 1, channel);

        window_datas[3] = input_mat(0, 0, channel);
        window_datas[4] = input_mat(0, 0, channel);   // x=0,y=0
        window_datas[5] = input_mat(0, 1, channel);   // x=1,y=0

        window_datas[6] = input_mat(1, 0, channel);
        window_datas[7] = input_mat(1, 0, channel);   // x=0,y=1
        window_datas[8] = input_mat(1, 1, channel);   // x=1,y=1
    }
    output_mat(0, 0, channel) =
        neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);

    for (int x = 1; x < width - 1; ++x) {
        if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
            window_datas[0] = foreground_value;
            window_datas[1] = foreground_value;
            window_datas[2] = foreground_value;

            window_datas[3] = input_mat(0, x - 1, channel);
            window_datas[4] = input_mat(0, x, channel);
            window_datas[5] = input_mat(0, x + 1, channel);

            window_datas[6] = input_mat(1, x - 1, channel);
            window_datas[7] = input_mat(1, x, channel);
            window_datas[8] = input_mat(1, x + 1, channel);
        } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                             (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
            window_datas[0] = background_value;
            window_datas[1] = background_value;
            window_datas[2] = background_value;

            window_datas[3] = input_mat(0, x - 1, channel);
            window_datas[4] = input_mat(0, x, channel);
            window_datas[5] = input_mat(0, x + 1, channel);

            window_datas[6] = input_mat(1, x - 1, channel);
            window_datas[7] = input_mat(1, x, channel);
            window_datas[8] = input_mat(1, x + 1, channel);
        } else {
            window_datas[0] = input_mat(0, x - 1, channel);
            window_datas[1] = input_mat(0, x, channel);
            window_datas[2] = input_mat(0, x + 1, channel);

            window_datas[3] = input_mat(0, x - 1, channel);
            window_datas[4] = input_mat(0, x, channel);
            window_datas[5] = input_mat(0, x + 1, channel);

            window_datas[6] = input_mat(1, x - 1, channel);
            window_datas[7] = input_mat(1, x, channel);
            window_datas[8] = input_mat(1, x + 1, channel);
        }
        output_mat(0, x, channel) =
            neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);
    }

    // (width-1,0)
    if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
        window_datas[0] = foreground_value;
        window_datas[1] = foreground_value;
        window_datas[2] = foreground_value;

        window_datas[3] = input_mat(0, width - 2, channel);
        window_datas[4] = input_mat(0, width - 1, channel);
        window_datas[5] = foreground_value;

        window_datas[6] = input_mat(1, width - 2, channel);
        window_datas[7] = input_mat(1, width - 1, channel);
        window_datas[8] = foreground_value;
    } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                         (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
        window_datas[0] = background_value;
        window_datas[1] = background_value;
        window_datas[2] = background_value;

        window_datas[3] = input_mat(0, width - 2, channel);
        window_datas[4] = input_mat(0, width - 1, channel);
        window_datas[5] = background_value;

        window_datas[6] = input_mat(1, width - 2, channel);
        window_datas[7] = input_mat(1, width - 1, channel);
        window_datas[8] = background_value;
    } else {
        window_datas[0] = input_mat(0, width - 2, channel);
        window_datas[1] = input_mat(0, width - 1, channel);
        window_datas[2] = input_mat(0, width - 1, channel);

        window_datas[3] = input_mat(0, width - 2, channel);
        window_datas[4] = input_mat(0, width - 1, channel);
        window_datas[5] = input_mat(0, width - 1, channel);

        window_datas[6] = input_mat(1, width - 2, channel);
        window_datas[7] = input_mat(1, width - 1, channel);   // x=width -1,y=1
        window_datas[8] = input_mat(1, width - 1, channel);   // x=width -1,y=1
    }
    output_mat(0, width - 1, 0) =
        neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);

    for (int y = 1; y < height - 1; ++y) {
        // 防止写入内存不连续,合并第一列
        if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
            window_datas[0] = foreground_value;
            window_datas[1] = input_mat(y - 1, 0, channel);
            window_datas[2] = input_mat(y - 1, 1, channel);

            window_datas[3] = foreground_value;
            window_datas[4] = input_mat(y, 0, channel);
            window_datas[5] = input_mat(y, 1, channel);

            window_datas[6] = foreground_value;
            window_datas[7] = input_mat(y + 1, 0, channel);
            window_datas[8] = input_mat(y + 1, 1, channel);
        } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                             (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
            window_datas[0] = foreground_value;
            window_datas[1] = input_mat(y - 1, 0, channel);
            window_datas[2] = input_mat(y - 1, 1, channel);

            window_datas[3] = foreground_value;
            window_datas[4] = input_mat(y, 0, channel);
            window_datas[5] = input_mat(y, 1, channel);

            window_datas[6] = foreground_value;
            window_datas[7] = input_mat(y + 1, 0, channel);
            window_datas[8] = input_mat(y + 1, 1, channel);
        } else {
            window_datas[0] = input_mat(y - 1, 0, channel);
            window_datas[1] = input_mat(y - 1, 0, channel);
            window_datas[2] = input_mat(y - 1, 1, channel);

            window_datas[3] = input_mat(y, 0, channel);
            window_datas[4] = input_mat(y, 0, channel);
            window_datas[5] = input_mat(y, 1, channel);

            window_datas[6] = input_mat(y + 1, 0, channel);
            window_datas[7] = input_mat(y + 1, 0, channel);
            window_datas[8] = input_mat(y + 1, 1, channel);
        }
        output_mat(y, 0, channel) =
            neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);
        for (int x = 1; x < width - 1; ++x) {
            window_datas[0] = input_mat(y - 1, x - 1, channel);
            window_datas[1] = input_mat(y - 1, x, channel);
            window_datas[2] = input_mat(y - 1, x + 1, channel);

            window_datas[3] = input_mat(y, x - 1, channel);
            window_datas[4] = input_mat(y, x, channel);
            window_datas[5] = input_mat(y, x + 1, channel);

            window_datas[6] = input_mat(y + 1, x - 1, channel);
            window_datas[7] = input_mat(y + 1, x, channel);
            window_datas[8] = input_mat(y + 1, x + 1, channel);

            //非边缘数据不需要
            output_mat(y, x, channel) =
                neighbor_filter_3x3_detail<T, filter_type, false>(window_datas, binary_count);
        }

        // handle the last column
        if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
            window_datas[0] = input_mat(y - 1, width - 2, channel);
            window_datas[1] = input_mat(y - 1, width - 1, channel);
            window_datas[2] = foreground_value;

            window_datas[3] = input_mat(y, width - 2, channel);
            window_datas[4] = input_mat(y, width - 1, channel);
            window_datas[5] = foreground_value;

            window_datas[6] = input_mat(y + 1, width - 2, channel);
            window_datas[7] = input_mat(y + 1, width - 1, channel);
            window_datas[8] = foreground_value;
        } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                             (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
            window_datas[0] = input_mat(y - 1, width - 2, channel);
            window_datas[1] = input_mat(y - 1, width - 1, channel);
            window_datas[2] = background_value;

            window_datas[3] = input_mat(y, width - 2, channel);
            window_datas[4] = input_mat(y, width - 1, channel);
            window_datas[5] = background_value;

            window_datas[6] = input_mat(y + 1, width - 2, channel);
            window_datas[7] = input_mat(y + 1, width - 1, channel);
            window_datas[8] = background_value;
        } else {
            // pad with nearest!
            window_datas[0] = input_mat(y - 1, width - 2, channel);
            window_datas[1] = input_mat(y - 1, width - 1, channel);
            window_datas[2] = input_mat(y - 1, width - 1, channel);

            window_datas[3] = input_mat(y, width - 2, channel);
            window_datas[4] = input_mat(y, width - 1, channel);
            window_datas[5] = input_mat(y, width - 1, channel);

            window_datas[6] = input_mat(y + 1, width - 2, channel);
            window_datas[7] = input_mat(y + 1, width - 1, channel);
            window_datas[8] = input_mat(y + 1, width - 1, channel);
        }
        output_mat(y, width - 1, channel) =
            neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);
    }

    //(0,height-1)
    if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
        window_datas[0] = foreground_value;
        window_datas[1] = input_mat(height - 2, 0, channel);
        window_datas[2] = input_mat(height - 2, 1, channel);

        window_datas[3] = foreground_value;
        window_datas[4] = input_mat(height - 1, 0, channel);
        window_datas[5] = input_mat(height - 1, 1, channel);

        window_datas[6] = foreground_value;
        window_datas[7] = foreground_value;
        window_datas[8] = foreground_value;
    } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                         (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
        window_datas[0] = background_value;
        window_datas[1] = input_mat(height - 2, 0, channel);
        window_datas[3] = input_mat(height - 2, 1, channel);

        window_datas[3] = foreground_value;
        window_datas[4] = input_mat(height - 1, 0, channel);
        window_datas[5] = input_mat(height - 1, 1, channel);

        window_datas[6] = foreground_value;
        window_datas[7] = foreground_value;
        window_datas[8] = foreground_value;
    } else {
        window_datas[0] = input_mat(height - 2, 0, channel);
        window_datas[1] = input_mat(height - 2, 0, channel);
        window_datas[2] = input_mat(height - 2, 1, channel);

        window_datas[3] = input_mat(height - 1, 0, channel);
        window_datas[4] = input_mat(height - 1, 0, channel);
        window_datas[5] = input_mat(height - 1, 1, channel);

        window_datas[6] = input_mat(height - 1, 0, channel);
        window_datas[7] = input_mat(height - 1, 0, channel);
        window_datas[8] = input_mat(height - 1, 1, channel);
    }
    output_mat(height - 1, 0, channel) =
        neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);


    for (int x = 1; x < width - 1; ++x) {
        if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
            window_datas[0] = input_mat(height - 2, x - 1, channel);
            window_datas[1] = input_mat(height - 2, x, channel);
            window_datas[2] = input_mat(height - 2, x + 1, channel);

            window_datas[3] = input_mat(height - 1, x - 1, channel);
            window_datas[4] = input_mat(height - 1, x, channel);
            window_datas[5] = input_mat(height - 1, x + 1, channel);

            window_datas[6] = foreground_value;
            window_datas[7] = foreground_value;
            window_datas[8] = foreground_value;
        } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                             (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
            window_datas[0] = input_mat(height - 2, x - 1, channel);
            window_datas[1] = input_mat(height - 2, x, channel);
            window_datas[2] = input_mat(height - 2, x + 1, channel);

            window_datas[3] = input_mat(height - 1, x - 1, channel);
            window_datas[4] = input_mat(height - 1, x, channel);
            window_datas[5] = input_mat(height - 1, x + 1, channel);

            window_datas[6] = background_value;
            window_datas[7] = background_value;
            window_datas[8] = background_value;
        } else {
            window_datas[0] = input_mat(height - 2, x - 1, channel);
            window_datas[1] = input_mat(height - 2, x, channel);
            window_datas[2] = input_mat(height - 2, x + 1, channel);

            window_datas[3] = input_mat(height - 1, x - 1, channel);
            window_datas[4] = input_mat(height - 1, x, channel);
            window_datas[5] = input_mat(height - 1, x + 1, channel);

            window_datas[6] = input_mat(height - 1, x - 1, channel);
            window_datas[7] = input_mat(height - 1, x, channel);
            window_datas[8] = input_mat(height - 1, x + 1, channel);
        }
        output_mat(height - 1, x, channel) =
            neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);
    }

    // width-1,height -1
    if constexpr (filter_type == NeighborFilterType::ERODE && pad_edges) {
        window_datas[0] = input_mat(height - 2, width - 2, channel);
        window_datas[1] = input_mat(height - 2, width - 1, channel);
        window_datas[2] = foreground_value;

        window_datas[3] = input_mat(height - 1, width - 2, channel);
        window_datas[4] = input_mat(height - 1, width - 1, channel);
        window_datas[5] = foreground_value;

        window_datas[6] = foreground_value;
        window_datas[7] = foreground_value;
        window_datas[8] = foreground_value;
    } else if constexpr (filter_type == NeighborFilterType::DILATE ||
                         (!pad_edges && filter_type == NeighborFilterType::ERODE)) {
        window_datas[0] = input_mat(height - 2, width - 2, channel);
        window_datas[1] = input_mat(height - 2, width - 1, channel);
        window_datas[2] = background_value;

        window_datas[3] = input_mat(height - 1, width - 2, channel);
        window_datas[4] = input_mat(height - 1, width - 1, channel);
        window_datas[5] = background_value;

        window_datas[6] = background_value;
        window_datas[7] = background_value;
        window_datas[8] = background_value;
    } else {
        window_datas[0] = input_mat(height - 2, width - 2, channel);
        window_datas[1] = input_mat(height - 2, width - 1, channel);
        window_datas[2] = input_mat(height - 2, width - 1, channel);

        window_datas[3] = input_mat(height - 1, width - 2, channel);
        window_datas[4] = input_mat(height - 1, width - 1, channel);
        window_datas[5] = input_mat(height - 1, width - 1, channel);

        window_datas[6] = input_mat(height - 1, width - 2, channel);
        window_datas[7] = input_mat(height - 1, width - 1, channel);
        window_datas[8] = input_mat(height - 1, width - 1, channel);
    }
    output_mat(height - 1, width - 1, channel) =
        neighbor_filter_3x3_detail<T, filter_type, true>(window_datas, binary_count);
}

}   // namespace internal

template<class T, typename>
Status::ErrorCode neighbor_filter_with_3x3_window(const ImageMat<T>& input_mat,
                                                  ImageMat<T>&       output_mat,
                                                  NeighborFilterType filter_type, bool pad_edges,
                                                  int binary_count) {
    if (input_mat.empty()) {
        LOG_ERROR("the input mat is invalid...");
        return Status::ErrorCode::InvalidMatShape;
    }

    if (input_mat.get_layout() != output_mat.get_layout()) {
        return Status::ErrorCode::MatLayoutMismath;
    }
    int height   = input_mat.get_height();
    int width    = input_mat.get_width();
    int channels = input_mat.get_channels();

    if (!input_mat.compare_shape(output_mat)) {
        output_mat.resize(height, width, channels, true);
    }

    const char* filter_type_str = get_neighbor_filter_str(filter_type);
    LOG_INFO("apply neighbor filte with type {}", filter_type_str);
    if (pad_edges) {
        for (int channel = 0; channel < channels; ++channel) {
            if (filter_type == NeighborFilterType::BLUR_MORE) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::BLUR_MORE, true>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::FIND_EDGES) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::FIND_EDGES, true>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::MEDIAN_FILTER) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::MEDIAN_FILTER, true>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::MIN) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::MIN, true>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::MAX) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::MAX, true>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::ERODE) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::ERODE, true>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::DILATE) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::DILATE, true>(
                    input_mat, output_mat, channel, binary_count);
            } else {
                LOG_ERROR("neight 3x3 filter for {} is not implemented!", filter_type_str);
                return Status::ErrorCode::UnsupportedNeighborFilterType;
            }
        }
    } else {
        for (int channel = 0; channel < channels; ++channel) {
            if (filter_type == NeighborFilterType::BLUR_MORE) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::BLUR_MORE, false>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::FIND_EDGES) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::FIND_EDGES, false>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::MEDIAN_FILTER) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::MEDIAN_FILTER, false>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::MIN) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::MIN, false>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::MAX) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::MAX, false>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::ERODE) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::ERODE, false>(
                    input_mat, output_mat, channel, binary_count);
            } else if (filter_type == NeighborFilterType::DILATE) {
                internal::neighbor_filter_3x3_impl<T, NeighborFilterType::DILATE, false>(
                    input_mat, output_mat, channel, binary_count);
            } else {
                LOG_ERROR("neight 3x3 filter for {} is not implemented!", filter_type_str);
                return Status::ErrorCode::UnsupportedNeighborFilterType;
            }
        }
    }
    return Status::ErrorCode::Ok;
}

template Status::ErrorCode neighbor_filter_with_3x3_window<uint8_t>(
    const ImageMat<uint8_t>& input_mat, ImageMat<uint8_t>& output_mat,
    NeighborFilterType filter_type, bool pad_edges, int binary_count);

template Status::ErrorCode neighbor_filter_with_3x3_window<uint16_t>(
    const ImageMat<uint16_t>& input_mat, ImageMat<uint16_t>& output_mat,
    NeighborFilterType filter_type, bool pad_edges, int binary_count);


template Status::ErrorCode neighbor_filter_with_3x3_window<float>(const ImageMat<float>& input_mat,
                                                                  ImageMat<float>&       output_mat,
                                                                  NeighborFilterType filter_type,
                                                                  bool pad_edges, int binary_count);
}   // namespace neighbor_filter
}   // namespace image_proc
}   // namespace fish