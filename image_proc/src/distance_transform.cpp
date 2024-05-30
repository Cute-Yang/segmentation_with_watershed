#include "image_proc/distance_transform.h"
#include "core/base.h"
#include "core/mat.h"
#include <cmath>
#include <limits>
#include <vector>

namespace fish {
namespace image_proc {
namespace distance_transform {
namespace internal {
enum DistanceKind : uint8_t { EuclideanDistance = 0, ManhattanDistance = 1 };

template<DistanceKind kind> struct DistanceTypeHelper { using type = float; };

template<> struct DistanceTypeHelper<DistanceKind::ManhattanDistance> { using type = int; };

enum ImageEdgeHandleType : uint8_t { TreatAsBackgroud = 0, TreatAsNormal = 1 };

constexpr int NO_POINT = -1;

float compute_min_distance(std::vector<Coordinate2d>& coors, Coordinate2d prev_coor,
                           Coordinate2d prev_diag_coor, int x, int y, int distance) {
    Coordinate2d coor         = coors[x];
    Coordinate2d nearest_coor = coor;
    int          new_distance;
    if (coor.x != NO_POINT) {
        new_distance = (x - coor.x) * (x - coor.x) + (y - coor.y) * (y - coor.y);
        if (new_distance < distance) {
            distance = new_distance;
        }
    }

    if (prev_diag_coor.x != NO_POINT && prev_diag_coor != coor) {
        new_distance = (x - prev_diag_coor.x) * (x - prev_diag_coor.x) +
                       (y - prev_diag_coor.y) * (y - prev_diag_coor.y);
        if (new_distance < distance) {
            distance     = new_distance;
            nearest_coor = prev_diag_coor;
        }
    }

    if (prev_coor.x != NO_POINT && prev_coor != coor) {
        new_distance =
            (x - prev_coor.x) * (x - prev_coor.x) + (y - prev_coor.y) * (y - prev_coor.y);
        if (new_distance < distance) {
            distance     = new_distance;
            nearest_coor = prev_coor;
        }
    }
    coors[x] = nearest_coor;
    return static_cast<float>(distance);
}

template<class T, ImageEdgeHandleType handle_type, typename = image_dtype_limit<T>>
void distance_transform_detail(const ImageMat<T>& input_mat, ImageMat<float>& distance_mat,
                               std::vector<Coordinate2d>& l2r_coors,
                               std::vector<Coordinate2d>& r2l_coors, int y, T background_value,
                               int y_dist) {
    Coordinate2d prev_coor(NO_POINT, NO_POINT);
    Coordinate2d prev_diag_coor(NO_POINT, NO_POINT);
    Coordinate2d next_diag_coor(NO_POINT, NO_POINT);

    int distance = std::numeric_limits<int>::max();
    int width    = input_mat.get_width();

    for (int x = 0; x < width; ++x) {
        // if you move to next row,the next_diag_coor maybe have a valid coor,it means the last
        // row's minist value,so treat it as previsou_diag
        next_diag_coor = l2r_coors[x];
        if (input_mat(y, x) == background_value) {
            l2r_coors[x].set_coor(x, y);
        } else {
            if constexpr (handle_type == ImageEdgeHandleType::TreatAsBackgroud) {
                distance = (x + 1) < y_dist ? (x + 1) * (x + 1) : y_dist * y_dist;
            }
            float min_distance = static_cast<float>(
                compute_min_distance(l2r_coors, prev_coor, prev_diag_coor, x, y, distance));
            if (min_distance < distance_mat(y, x)) {
                distance_mat(y, x) = min_distance;
            }
        }
        prev_coor      = l2r_coors[x];
        prev_diag_coor = next_diag_coor;
    }

    prev_coor.set_coor(NO_POINT, NO_POINT);
    prev_diag_coor.set_coor(NO_POINT, NO_POINT);
    for (int x = width - 1; x >= 0; --x) {
        next_diag_coor = r2l_coors[x];
        if (input_mat(y, x) == background_value) {
            r2l_coors[x].set_coor(x, y);
        } else {
            if constexpr (handle_type == ImageEdgeHandleType::TreatAsBackgroud) {
                distance = (width - x) < y_dist ? (width - x) * (width - x) : y_dist * y_dist;
            }
            float min_distance =
                compute_min_distance(r2l_coors, prev_coor, prev_diag_coor, x, y, distance);
            if (min_distance < distance_mat(y, x)) {
                distance_mat(y, x) = min_distance;
            }
        }
        prev_coor      = r2l_coors[x];
        prev_diag_coor = next_diag_coor;
    }
}


template<class T, ImageEdgeHandleType handle_type, typename = image_dtype_limit<T>>
void distance_transform_impl(const ImageMat<T>& input_mat, ImageMat<float>& distance_mat,
                             T background_value) {
    int height = input_mat.get_height();
    int width  = input_mat.get_width();

    constexpr float max_distance_value  = std::numeric_limits<float>::max();
    constexpr float zero_distance_value = 0.0f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (input_mat(y, x) != background_value) {
                distance_mat(y, x) = max_distance_value;
            } else {
                distance_mat(y, x) = zero_distance_value;
            }
        }
    }

    int                       y_dist = std::numeric_limits<int>::max();
    std::vector<Coordinate2d> l2r_coors(width, {NO_POINT, NO_POINT});
    std::vector<Coordinate2d> r2l_coors(width, {NO_POINT, NO_POINT});

    for (int y = 0; y < height; ++y) {
        if constexpr (handle_type == ImageEdgeHandleType::TreatAsBackgroud) {
            y_dist = y + 1;
        }
        distance_transform_detail<T, handle_type>(
            input_mat, distance_mat, l2r_coors, r2l_coors, y, background_value, y_dist);
    }

    l2r_coors.assign(l2r_coors.size(), {NO_POINT, NO_POINT});
    r2l_coors.assign(r2l_coors.size(), {NO_POINT, NO_POINT});

    for (int y = height - 1; y >= 0; --y) {
        if constexpr (handle_type == ImageEdgeHandleType::TreatAsBackgroud) {
            y_dist = height - y;
        }
        distance_transform_detail<T, handle_type>(
            input_mat, distance_mat, l2r_coors, r2l_coors, y, background_value, y_dist);
    }

    float* distance_ptr = distance_mat.get_data_ptr();
    int    data_size    = distance_mat.get_element_num();
    // apply sqrt!
    for (int i = 0; i < data_size; ++i) {
        distance_ptr[i] = std::sqrt(distance_ptr[i]);
    }
}
}   // namespace internal


template<class T, typename>
Status::ErrorCode distance_transform(const ImageMat<T>& input_mat, ImageMat<float>& distance_mat,
                                     bool treat_edge_as_background, T background_value) {
    if (input_mat.empty()) {
        return Status::ErrorCode::InvalidMatShape;
    }
    if (distance_mat.get_layout() != input_mat.get_layout()) {
        return Status::ErrorCode::MatLayoutMismath;
    }

    int height   = input_mat.get_height();
    int width    = input_mat.get_width();
    int channels = input_mat.get_channels();
    if (channels != 1) {
        return Status::ErrorCode::InvalidMatChannle;
    }

    // very clear!
    if (!distance_mat.shape_equal(height, width, 1)) {
        distance_mat.resize(height, width, 1, true);
    }
    if (treat_edge_as_background) {
        internal::distance_transform_impl<T, internal::ImageEdgeHandleType::TreatAsBackgroud>(
            input_mat, distance_mat, background_value);
    } else {
        internal::distance_transform_impl<T, internal::ImageEdgeHandleType::TreatAsNormal>(
            input_mat, distance_mat, background_value);
    }
    return Status::Ok;
}

template Status::ErrorCode distance_transform<uint8_t>(const ImageMat<uint8_t>& input_mat,
                                                       ImageMat<float>&         output_mat,
                                                       bool    treat_edge_as_background,
                                                       uint8_t background_value);

template Status::ErrorCode distance_transform<uint16_t>(const ImageMat<uint16_t>& input_mat,
                                                        ImageMat<float>&          output_mat,
                                                        bool     treat_edge_as_background,
                                                        uint16_t background_value);


template Status::ErrorCode distance_transform<float>(const ImageMat<float>& input_mat,
                                                     ImageMat<float>&       output_mat,
                                                     bool  treat_edge_as_background,
                                                     float background_value);

}   // namespace distance_transform
}   // namespace image_proc
}   // namespace fish