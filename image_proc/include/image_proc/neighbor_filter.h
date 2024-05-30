#pragma once
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include <array>
#include <cstdint>
enum NeighborFilterType : uint8_t {
    BLUR_MORE     = 0,
    FIND_EDGES    = 1,
    MEDIAN_FILTER = 2,
    MIN           = 3,
    MAX           = 4,
    CONVOLVE      = 5,
    ERODE         = 6,
    DILATE        = 7,
    FilterTypeCount
};

constexpr std::array<const char*, NeighborFilterType::FilterTypeCount> NeighborFilterTypeStr = {
    "BLUR_MORE", "FIND_EDGES", "MEDIAN_FILTER", "MIN", "MAX", "CONVOLVE", "ERODE", "DILATE"};

FISH_ALWAYS_INLINE const char* get_neighbor_filter_str(NeighborFilterType filter_type) {
    return NeighborFilterTypeStr[filter_type];
}

namespace fish {
namespace image_proc {
namespace neighbor_filter {
using namespace fish::core::mat;
template<class T, typename = dtype_limit<T>>
Status::ErrorCode neighbor_filter_with_3x3_window(const ImageMat<T>& input_mat,
                                                  ImageMat<T>&       output_mat,
                                                  NeighborFilterType filter_type, bool pad_edges,
                                                  int binary_count);

FISH_INLINE Status::ErrorCode neighbor_filter_with_3x3_window_u8(const ImageMat<uint8_t>& input_mat,
                                                                 ImageMat<uint8_t>& output_mat,
                                                                 NeighborFilterType filter_type,
                                                                 bool pad_edges, int binary_count) {
    return neighbor_filter_with_3x3_window<uint8_t>(
        input_mat, output_mat, filter_type, pad_edges, binary_count);
}


FISH_INLINE Status::ErrorCode neighbor_filter_with_3x3_window_u16(
    const ImageMat<uint16_t>& input_mat, ImageMat<uint16_t>& output_mat,
    NeighborFilterType filter_type, bool pad_edges, int binary_count) {
    return neighbor_filter_with_3x3_window<uint16_t>(
        input_mat, output_mat, filter_type, pad_edges, binary_count);
}

FISH_INLINE Status::ErrorCode neighbor_filter_with_3x3_window_f32(const ImageMat<float>& input_mat,
                                                                  ImageMat<float>&       output_mat,
                                                                  NeighborFilterType filter_type,
                                                                  bool               pad_edges,
                                                                  int                binary_count) {
    return neighbor_filter_with_3x3_window<float>(
        input_mat, output_mat, filter_type, pad_edges, binary_count);
}


}   // namespace neighbor_filter
}   // namespace image_proc
}   // namespace fish