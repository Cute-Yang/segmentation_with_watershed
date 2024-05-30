#include "core/base.h"
#include "core/mat.h"
namespace fish {
namespace image_proc {
namespace find_maximum {
using namespace fish::core::mat;
enum class EDMOutputType : uint8_t {
    SINGLE_POINTS   = 0,
    IN_TOLERANCE    = 1,
    SEGMENTED       = 2,
    POINT_SELECTION = 3,
    LIST            = 4,
    COUNT           = 5
};
constexpr float   NO_THRESHOLD = -808080.0;
ImageMat<uint8_t> find_maxima(ImageMat<float>& distance_mat, ImageMat<uint8_t>& distance_mask,
                              bool strict, float to_lerance, float threshold,
                              EDMOutputType output_type, bool exclude_on_edges, bool is_EDM);
Status::ErrorCode find_maxima(ImageMat<float>& distance_mat, ImageMat<uint8_t>& distance_mask,
                              ImageMat<uint8_t>& maximum_mask, bool strict, float to_lerance,
                              float threshold, EDMOutputType output_type, bool exclude_on_edges,
                              bool is_EDM);
// template<class T>
// ImageMat<uint8_t> find_maximum()
}   // namespace find_maximum
}   // namespace image_proc
}   // namespace fish