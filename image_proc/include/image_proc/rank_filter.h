#pragma once
#include "core/base.h"
#include "core/mat.h"
#include <array>

namespace fish {
namespace image_proc {
namespace rank_filter {
using namespace fish::core::base;
using namespace fish::core::mat;
enum OutlierValueKind : uint8_t { WhiteOutlier = 0, DarkOutlier = 1 };
enum FilterType : uint32_t {
    MEAN       = 0,
    MIN        = 1,
    MAX        = 2,
    VARIANCE   = 3,
    MEDIAN     = 4,
    OUTLIER    = 5,
    DESPECKLE  = 6,
    REMOVE_NAN = 7,
    OPEN       = 8,
    CLOSE      = 9,
    TOP_HAT    = 10,
    FilterTypeCount
};

constexpr std::array<const char*, FilterType::FilterTypeCount> FilterTypeStr = {
    "MEAN", "MIN", "MAX", "VARIANCE", "MEDIAN", "OUTLIER", "DESPECKLE", "OPEN", "CLOSE", "TOP_HAT"};

template<class T, typename = image_dtype_limit<T>>
Status::ErrorCode rank_filter(const ImageMat<T>& input_mat, ImageMat<T>& out_mat,
                              FilterType rank_filter_type, double radius);
}   // namespace rank_filter

}   // namespace image_proc
}   // namespace fish