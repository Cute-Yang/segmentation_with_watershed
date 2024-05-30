#pragma once
#include "common/fishdef.h"
#include "core/mat.h"
#include "image_proc/rank_filter.h"
namespace fish {
namespace image_proc {
namespace rank_filter {
namespace internal {

template<class T> struct SumTypeHelper { using type = uint32_t; };
template<> struct SumTypeHelper<int64> { using type = int64_t; };
template<> struct SumTypeHelper<uint64_t> { using type = uint64_t; };
template<> struct SumTypeHelper<float> { using type = float; };
template<> struct SumTypeHelper<double> { using type = double; };

template<class T, typename = image_dtype_limit<T>>
void copy_to_cache(const ImageMat<T>& input_mat, T* cache_ptr, int y_in_cache, int cache_width,
                   int y, int channel, int k_radius) {
    // the offset of read original value!
    int width  = input_mat.get_width();
    int offset = y_in_cache * cache_width + k_radius;
    for (int x = 0; x < width; ++x) {
        cache_ptr[offset + x] = input_mat(y, x, channel);
    }

    // do pad
    const T pad_left_value = cache_ptr[offset];
    // the last value
    const T pad_right_value = cache_ptr[offset + width - 1];

    int pad_left_offset = y_in_cache * cache_width;
    for (int x = 0; x < k_radius; ++x) {
        cache_ptr[pad_left_offset + x] = pad_left_value;
    }

    int pad_right_offset = offset + width;
    for (int x = 0; x < k_radius; ++x) {
        cache_ptr[pad_right_offset + x] = pad_right_value;
    }
}

/**
 * @brief
 *
 * @tparam T
 * @tparam typename
 * @param cache
 * @param cache_points this is the coor of data!
 * @param cache_points_size
 * @param x0
 * @param ignore_right
 * @param max_value
 * @return T
 */
template<class T, typename = dtype_limit<T>>
T compute_area_max_value(const T* cache, const int* cache_points, int cache_points_size, int x0,
                         int ignore_right, T max_value) {
    for (int kk = 0; kk < cache_points_size; kk += 2) {
        for (size_t offset = cache_points[kk]; offset <= cache_points[kk + 1] - ignore_right;
             ++offset) {
            if (max_value < cache[x0 + offset]) {
                max_value = cache[x0 + offset];
            }
        }
    }
    return max_value;
}

template<class T, bool is_right, typename = dtype_limit<T>>
T compute_side_max_value(const T* cache, const int* cache_points, int cache_points_size, int x0) {
    T max_value = std::numeric_limits<T>::lowest();
    if constexpr (is_right) {
        for (int kk = 1; kk < cache_points_size; kk += 2) {
            if (max_value < cache[x0 + cache_points[kk]]) {
                max_value = cache[x0 + cache_points[kk]];
            }
        }
    } else {
        --x0;
        for (int kk = 0; kk < cache_points_size; ++kk) {
            if (max_value < cache[x0 + cache_points[kk]]) {
                max_value = cache[x0 + cache_points[kk]];
            }
        }
    }
    return max_value;
}


template<class T, typename = dtype_limit<T>>
T compute_area_min_value(const T* cache, const int* cache_points, int cache_points_size, int x0,
                         int ignore_right, T min_value) {
    for (int kk = 0; kk < cache_points_size; kk += 2) {
        for (int offset = cache_points[kk]; offset <= cache_points[kk + 1] - ignore_right;
             ++offset) {
            if (min_value > cache[x0 + offset]) {
                min_value = cache[x0 + offset];
            }
        }
    }
    return min_value;
}

template<class T, bool is_right, typename = dtype_limit<T>>
T compute_side_min_value(const T* cache, const int* cache_points, int cache_points_size, int x0) {
    T min_value = std::numeric_limits<T>::max();
    if constexpr (is_right) {
        for (size_t kk = 1; kk < cache_points_size; kk += 2) {
            if (min_value > cache[x0 + cache_points[kk]]) {
                min_value = cache[x0 + cache_points[kk]];
            }
        }
    } else {
        --x0;
        for (size_t kk = 0; kk < cache_points_size; kk += 2) {
            if (min_value > cache[x0 + cache_points[kk]]) {
                min_value = cache[x0 + cache_points[kk]];
            }
        }
    }
    return min_value;
}


template<class T, typename = dtype_limit<T>>
double compute_area_sum_value(const T* cache, const int* cache_points, int cache_points_size,
                              int x0) {
    double sum_value = 0.0;
    for (int kk = 0; kk < cache_points_size; kk += 2) {
        for (int p = cache_points[kk] + x0; p <= cache_points[kk + 1] + x0; ++p) {
            sum_value += static_cast<double>(cache[p]);
        }
    }
    return sum_value;
}

template<class T, typename = dtype_limit<T>>
void compute_area_sum_value(const T* cache, const int* cache_points, int cache_points_size, int x0,
                            double* sum_ptr) {
    double sum_value_0 = 0.0;
    double sum_value_1 = 0.0;
    for (int kk = 0; kk < cache_points_size; kk += 2) {
        for (int p = cache_points[kk] + x0; p <= cache_points[kk + 1] + x0; ++p) {
            sum_value_0 += cache[p];
            sum_value_1 += (cache[p] * cache[p]);
        }
    }
    sum_ptr[0] = sum_value_0;
    sum_ptr[1] = sum_value_1;
}


template<class T, typename = dtype_limit<T>>
double compute_side_sum_value(const T* cache, const int* cache_points, int cache_points_size,
                              int x0) {
    double sum_value = 0.0;
    for (size_t kk = 0; kk < cache_points_size; kk += 2) {
        sum_value -= static_cast<double>(cache[cache_points[kk] + x0 - 1]);
        sum_value += static_cast<double>(cache[cache_points[kk + 1] + x0]);
    }
    return sum_value;
}


template<class T, typename = dtype_limit<T>>
void compute_side_sum_value(const T* cache, const int* cache_points, int cache_points_size, int x0,
                            double* sum_ptr) {
    double sum_value_0 = 0.0;
    double sum_value_1 = 0.0;
    for (int kk = 0; kk < cache_points_size; kk += 2) {
        sum_value_0 -= cache[cache_points[kk] + x0 - 1];
        sum_value_1 -= cache[cache_points[kk] + x0 - 1] * cache[cache_points[kk] + x0 - 1];

        sum_value_0 += cache[cache_points[kk + 1] + x0];
        sum_value_1 += cache[cache_points[kk + 1] + x0] * cache[cache_points[kk + 1] + x0];
    }
    sum_ptr[0] = sum_value_0;
    sum_ptr[1] = sum_value_1;
}

template<class T, typename = dtype_limit<T>>
T compute_area_median_value(const T* cache, const int* cache_points, int cache_points_size, int x0,
                            T* temp_sort_buffer, int sample_num) {
    int data_idx = 0;
    for (size_t kk = 0; kk < cache_points_size; kk += 2) {
        for (int p = cache_points[kk] + x0; p <= cache_points[kk + 1] + x0; ++p) {
            temp_sort_buffer[data_idx] = cache[p];
            ++data_idx;
        }
    }
    // maybe slow...
    std::sort(temp_sort_buffer, temp_sort_buffer + data_idx);
    return temp_sort_buffer[sample_num / 2];
}

constexpr bool is_implemented(FilterType rank_filter_type) {
    if (rank_filter_type == FilterType::MAX || rank_filter_type == FilterType::MIN ||
        rank_filter_type == FilterType::MEAN || rank_filter_type == FilterType::VARIANCE ||
        rank_filter_type == FilterType::MEDIAN || rank_filter_type == FilterType::OUTLIER) {
        return true;
    }
    return false;
}

}   // namespace internal
}   // namespace rank_filter
}   // namespace image_proc
}   // namespace fish