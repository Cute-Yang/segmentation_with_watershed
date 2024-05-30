#pragma once
#include "core/mat.h"
#include <vector>

namespace fish {
namespace image_proc {
namespace polygon {
using namespace fish::core::mat;
enum class RoiType {
    RECTANGLE  = 0,
    OVAL       = 1,
    POLYGON    = 2,
    FREEROI    = 3,
    TRADED_ROI = 4,
    LINE       = 5,
    POLYLINE   = 6,
    FREELINE   = 7,
    ANGLE      = 8,
    COMPOSITE  = 9,
    POINT      = 10
};

constexpr bool roi_is_line(RoiType roi_type) {
    return roi_type >= RoiType::LINE && roi_type <= RoiType::ANGLE;
}

constexpr bool roi_is_line_or_point(RoiType roi_type) {
    return roi_type == RoiType::POINT || roi_is_line(roi_type);
}


using PolygonType    = std::vector<Coordinate2d>;
using PolygonTypef32 = std::vector<Coordinate2df32>;
using PolygonTypef64 = std::vector<Coordinate2df64>;

Rectangle get_bounding_box(const std::vector<Coordinate2d>& points);
Rectangle get_bounding_box(const Coordinate2d* points, int point_size);
bool      point_in_polygon(const PolygonType& points, int x, int y);

Coordinate2d translate_to_origin(const PolygonType& polygon, PolygonType& out_polygon);
Coordinate2d translate_to_origin(PolygonType& polygon);

void convert_polygon_to_float(const PolygonType& polygon, PolygonTypef32& converted_polygon);

PolygonTypef32 convert_polygon_to_float(const PolygonType& polygon);

// void get_interpolated_polygon(const PolygonTypef32& polygon, double internal, bool smooth);
PolygonTypef32 get_interpolated_polygon(const PolygonTypef32& polygon, double internal, bool smooth,
                                        RoiType roi_type);

void get_interpolated_polygon(const PolygonTypef32& polygon, PolygonTypef32& middle_polygon,
                              PolygonTypef32& out_polygon, double internal, bool smooth,
                              RoiType roi_type);

void           smooth_polygon_roi(const PolygonTypef32& polygon, PolygonTypef32& smoothed_polygon);
PolygonTypef32 smooth_polygon_roi(const PolygonTypef32& polygon);

void scale_polygon_roi_inplace(PolygonTypef32& polygon, float x_offset, float y_offset,
                               float downsample_factor);

void           scale_polygon_roi(const PolygonTypef32& polygon, PolygonTypef32& scaled_polygon,
                                 float x_offset, float y_offet, float downsample_factor);
PolygonTypef32 scale_polygon_roi(const PolygonTypef32& polygon, float x_offset, float y_offet,
                                 float downsample_factor);
// impl the calibrator recovery polygon in qupath!hah!
}   // namespace polygon
}   // namespace image_proc
}   // namespace fish