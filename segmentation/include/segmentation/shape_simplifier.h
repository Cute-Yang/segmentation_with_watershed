#pragma once
#include "image_proc/polygon.h"
namespace fish {
namespace segmentation {
namespace shape_simplier {
using namespace fish::image_proc::polygon;
PolygonTypef32 simplify_polygon_points(const PolygonTypef32& polygon, float altitude_threshold);

PolygonTypef32 simplify_polygon_points_better(const PolygonTypef32& polygon,
                                              float                 altitude_threshold);
}   // namespace shape_simplier
}   // namespace segmentation
}   // namespace fish