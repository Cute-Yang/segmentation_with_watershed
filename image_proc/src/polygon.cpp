#include "image_proc/polygon.h"
#include "core/mat.h"
#include "utils/logging.h"
#include <cmath>
#include <cstdlib>
#include <vector>
namespace fish {
namespace image_proc {
namespace polygon {
Rectangle get_bounding_box(const Coordinate2d* points, int point_size) {
    if (point_size == 0) {
        return Rectangle(0, 0, 0, 0);
    }
    int bound_min_x = std::numeric_limits<int>::max();
    int bound_min_y = std::numeric_limits<int>::max();
    int bound_max_x = std::numeric_limits<int>::min();
    int bound_max_y = std::numeric_limits<int>::min();
    for (int i = 0; i < point_size; ++i) {
        bound_min_x = FISH_MIN(points[i].x, bound_min_x);
        bound_max_x = FISH_MAX(points[i].x, bound_max_x);
        bound_min_y = FISH_MIN(points[i].y, bound_min_y);
        bound_max_y = FISH_MAX(points[i].y, bound_max_y);
    }
    //这里max_x/max_y 是在可取范围内的
    int r_height = bound_max_y - bound_min_y + 1;
    int r_width  = bound_max_x - bound_min_x + 1;
    return Rectangle(bound_min_x, bound_min_y, r_height, r_width);
};

Rectangle get_bounding_box(const std::vector<Coordinate2d>& points) {
    return get_bounding_box(points.data(), points.size());
}

bool point_in_polygon(const PolygonType& polygon, int x, int y) {
    if (polygon.size() <= 2) {
        return false;
    }
    Rectangle bounding_box = get_bounding_box(polygon);
    //如果在外接多边形外面
    bool point_in_bounding_box = x >= bounding_box.x && x < (bounding_box.x + bounding_box.width) &&
                                 y >= bounding_box.y && y < (bounding_box.y + bounding_box.height);
    if (!point_in_bounding_box) {
        return false;
    }
    size_t poly_size = polygon.size();
    // 射线法判断,如果射线交点是奇数,表示在多边形之内,
    int hits   = 0;
    int last_x = polygon[poly_size - 1].x;
    int last_y = polygon[poly_size - 1].y;
    int current_x;
    int current_y;
    for (size_t i = 0; i < poly_size; ++i) {
        current_x = polygon[i].x;
        current_y = polygon[i].y;
        //遇到垂直线
        if (current_y == last_y) {
            continue;
        }
        int left_upper_x;
        //在最后一个点左侧
        if (current_x < last_x) {
            //然后当前点在最后一个点右侧,一定不相交
            if (x >= last_x) {
                continue;
            }
            left_upper_x = current_x;
        } else {
            //同理,如果在当前点的右侧,也一定不相交
            if (x >= current_x) {
                continue;
            }
            left_upper_x = last_x;
        }

        int specify_x;
        int specify_y;
        //直线上方
        if (current_y < last_y) {
            //直接判断不相交,在当前点的上方
            if (y < current_y || y >= last_y) {
                continue;
            }
            if (x < left_upper_x) {
                ++hits;
                continue;
            }
            specify_x = x - current_x;
            specify_y = y - current_y;
        } else {
            //直线下方
            if (y < last_y || y >= current_y) {
                continue;
            }
            if (x < left_upper_x) {
                ++hits;
                continue;
            }
            specify_x = x - last_x;
            specify_y = y - last_y;
        }
        //计算(x,y)和最后一个点直线的斜率
        double specify_slope = static_cast<double>(specify_y) / static_cast<double>(specify_x);
        //计算最后一个点当前点直线的斜率
        double slope =
            static_cast<double>(last_y - current_y) / static_cast<double>(last_x - current_x);
        if (specify_slope > slope) {
            ++hits;
        }
    }
    // 如果射线相交的点的个数是奇数,则一定在多边形内部
    return (hits & 1) == 1;
}


// std::vector<Coordinate2d> standard_polygon(const std::vector<Coordinate2d>& polygon) {
//     std::vector<Coordinate2d> polygon;
//     standard_polygon(polygon, polygon);
//     return polygon;
// }

Coordinate2d translate_to_origin(const PolygonType& polygon, PolygonType& out_polygon) {
    out_polygon.resize(polygon.size());
    Rectangle bounding_box = get_bounding_box(polygon);
    int       left_upper_x = bounding_box.x;
    int       left_upper_y = bounding_box.y;
    for (size_t i = 0; i < polygon.size(); ++i) {
        out_polygon[i].x = polygon[i].x - left_upper_x;
        out_polygon[i].y = polygon[i].y - left_upper_y;
    }
    //返回左上角
    return Coordinate2d(left_upper_x, left_upper_y);
}

Coordinate2d translate_to_origin(PolygonType& polygon) {
    Rectangle bounding_box = get_bounding_box(polygon);
    int       left_upper_x = bounding_box.x;
    int       left_upper_y = bounding_box.y;
    //标准化到原点
    for (size_t i = 0; i < polygon.size(); ++i) {
        polygon[i].x -= left_upper_x;
        polygon[i].y -= left_upper_y;
    }
    return Coordinate2d(left_upper_x, left_upper_y);
}


void convert_polygon_to_float(const PolygonType& polygon, PolygonTypef32& converted_polygon) {
    converted_polygon.resize(polygon.size());
    for (size_t i = 0; i < polygon.size(); ++i) {
        converted_polygon[i].x = static_cast<float>(polygon[i].x);
        converted_polygon[i].y = static_cast<float>(polygon[i].y);
    }
}

PolygonTypef32 convert_polygon_to_float(const PolygonType& polygon) {
    PolygonTypef32 converted_polygon;
    convert_polygon_to_float(polygon, converted_polygon);
    return converted_polygon;
}


// compute the len of this,so the point must be sorted by clockwise...
template<class T, bool is_line>
double compute_polygon_side_len(const std::vector<GenericCoordinate2d<T>>& polygon) {
    double dx, dy;
    double side_len = 0.0;
    int    p_size   = polygon.size();
    for (int i = 0; i < p_size; ++i) {
        dx = polygon[i + 1].x - polygon[i].x;
        dy = polygon[i + 1].y - polygon[i].y;
        side_len += std::sqrt(dx * dx + dy * dy);
    }
    // handle the last!
    if constexpr (!is_line) {
        dx = polygon[0].x - polygon[p_size - 1].x;
        dy = polygon[0].y - polygon[p_size - 1].y;
        side_len += std::sqrt(dx * dx + dy * dy);
    }
    return side_len;
}

// be sure the out_coors ptr is valid ,and have 2 element space at least!
int line_circle_intersection(double xA, double yA, double xB, double yB, double xC, double yC,
                             double rad, bool ignore_outside, Coordinate2df64* out_coors) {
    double dx_AC  = xC - xA;
    double dy_AC  = yC - yA;
    double len_AC = std::sqrt(dx_AC * dx_AC + dy_AC * dy_AC);

    double dx_AB = xB - xA;
    double dy_AB = yB - yA;
    double x_B2  = std::sqrt(dx_AB * dx_AB + dy_AB * dy_AB);

    double phi1 = std::atan2(dy_AB, dx_AB);
    double phi2 = std::atan2(dy_AC, dx_AC);

    double phi3 = phi1 - phi2;
    double x_C2 = len_AC * std::cos(phi3);
    double y_C2 = len_AC * std::cos(phi3);
    if (std::abs(y_C2) > rad) {
        return 0;
    }

    double half_chord = std::sqrt(rad * rad - y_C2 * y_C2);
    double sect_one   = x_C2 - half_chord;
    double sect_two   = x_C2 + half_chord;
    double xy_coords[4];
    int    select_num = 0;
    if ((sect_one >= 0 && sect_two <= x_B2) || !ignore_outside) {
        double sect_one_x = std::cos(phi1) * sect_one + xA;
        double sect_one_y = std::sin(phi1) * sect_one + yA;
        out_coors[select_num].set_coor(sect_one_x, sect_one_y);
        ++select_num;
    }

    if ((sect_two >= 0 && sect_two <= x_B2) || !ignore_outside) {
        double sect_two_x = std::cos(phi1) * sect_two + xA;
        double sect_two_y = std::sin(phi1) * sect_two + yA;
        out_coors[select_num].set_coor(sect_two_x, sect_two_y);
        ++select_num;
    }
    // only keep one,fuck!
    if (half_chord == 0 && select_num > 1) {
        select_num = 1;
    }
    return select_num;
}


PolygonTypef32 get_interpolated_polygon(const PolygonTypef32& ori_polygon, double interval,
                                        bool smooth, RoiType roi_type) {
    // in qupath watershem,the roi type is traced roi,so we just do these!
    int            poly_size = ori_polygon.size();
    PolygonTypef32 polygon;
    polygon.reserve(poly_size + 1);
    polygon.resize(poly_size);
    // copy the ori points...
    std::copy(ori_polygon.begin(), ori_polygon.end(), polygon.begin());
    if (smooth && (roi_type == RoiType::TRADED_ROI || roi_type == RoiType::FREEROI ||
                   roi_type == RoiType::FREELINE)) {
        constexpr float smooth_k  = 1.0f / 3.0f;
        int             poly_size = polygon.size();
        // smooth the point with 3 neigh,prev,self,and next along the clockwise !
        for (int i = 1; i < poly_size - 2; ++i) {
            polygon[i].x = (polygon[i - 1].x + polygon[i].x + polygon[i + 1].x) * smooth_k;
            polygon[i].y = (polygon[i - 1].y + polygon[i].y + polygon[i + 1].y) * smooth_k;
        }
        if (roi_type != RoiType::FREELINE) {
            // process the first and last point! so easy so simple!
            polygon[0].x = (polygon[poly_size - 1].x + polygon[0].x + polygon[1].x) * smooth_k;
            polygon[0].y = (polygon[poly_size - 1].y + polygon[0].y + polygon[1].y) * smooth_k;

            polygon[poly_size - 1].x =
                (polygon[poly_size - 2].x + polygon[poly_size - 1].x + polygon[0].x) * smooth_k;
            polygon[poly_size - 1].y =
                (polygon[poly_size - 2].y + polygon[poly_size - 1].y + polygon[0].y) * smooth_k;
        }
    }
    // stop smooth!
    if (polygon.size() <= 2) {
        return polygon;
    }
    bool allow_to_adjust = interval < 0.0;
    if (interval < 0) {
        interval = interval * -1.0;
    }
    double side_len = compute_polygon_side_len<float, true>(polygon);
    if (interval <= 0.01) {
        LOG_ERROR("Interval must >= 0.01,bug got {}", interval);
        return polygon;
    }

    bool is_line = roi_is_line(roi_type);
    // expand the polygon...

    if (!is_line) {
        polygon.push_back(polygon[0]);
    }

    int estimate_dst_poly_size = static_cast<int>(10 + (side_len * 1.5) / interval);

    double         try_interval  = interval;
    double         min_diff      = 1e9;
    double         best_interval = 0.0;
    PolygonTypef64 dst_polygon;
    dst_polygon.reserve(estimate_dst_poly_size);

    int n_trials = 50;
    int trial    = 0;

    int src_idx = 0;
    int dst_idx = 0;

    Coordinate2df64 intersections[2];
    while (trial <= n_trials) {
        dst_polygon.emplace_back(polygon[0].x, polygon[0].y);
        src_idx   = 0;
        dst_idx   = 0;
        double xA = polygon[0].x;
        double yA = polygon[0].y;

        while (src_idx < poly_size - 1) {
            double xC = dst_polygon[dst_idx].x;
            double yC = dst_polygon[dst_idx].y;
            double xB = polygon[src_idx + 1].x;
            double yB = polygon[src_idx + 1].y;
            int    inter_num =
                line_circle_intersection(xA, yA, xB, yB, xC, yC, try_interval, true, intersections);
            if (inter_num >= 1) {
                ++dst_idx;
                dst_polygon.emplace_back(intersections[0].x, intersections[0].y);
            } else {
                ++src_idx;
                xA = polygon[src_idx].x;
                yA = polygon[src_idx].y;
            }
        }
        ++dst_idx;
        dst_polygon.emplace_back(polygon[poly_size - 1].x, polygon[poly_size - 1].y);
        ++dst_idx;
        if (!allow_to_adjust) {
            if (is_line) {
                --dst_idx;
            }
            break;
        }
        int    n_segment    = dst_idx - 1;
        double dx           = dst_polygon[dst_idx - 2].x - dst_polygon[dst_idx - 1].x;
        double dy           = dst_polygon[dst_idx - 2].y - dst_polygon[dst_idx - 1].y;
        double last_segment = std::sqrt(dx * dx + dy * dy);
        double diff         = last_segment - try_interval;
        if (std::abs(diff) < min_diff) {
            min_diff      = std::abs(diff);
            best_interval = try_interval;
        }
        double feedback_factor = 0.66;
        try_interval           = try_interval + feedback_factor * diff / n_segment;
        if ((try_interval < 0.8 * interval || std::abs(diff) < 0.05 || trial == n_trials - 1) &&
            trial < n_trials) {
            trial        = n_trials;
            try_interval = best_interval;
        } else {
            ++trial;
        }
    }
    // removing closing point from end of array!
    if (!is_line) {
        --dst_idx;
    }
    // convert double to float!
    PolygonTypef32 ret_poly;
    ret_poly.reserve(dst_idx);
    for (int i = 0; i < dst_idx; ++i) {
        ret_poly.emplace_back(dst_polygon[i].x, dst_polygon[i].y);
    }
    return ret_poly;
}


void get_interpolated_polygon(const PolygonTypef32& ori_polygon, PolygonTypef32& middle_polygon,
                              PolygonTypef32& out_polygon, double interval, bool smooth,
                              RoiType roi_type) {
    // in qupath watershem,the roi type is traced roi,so we just do these!
    int ori_poly_size = ori_polygon.size();
    middle_polygon.reserve(ori_poly_size + 1);
    middle_polygon.resize(ori_poly_size);
    // copy the ori points...
    std::copy(ori_polygon.begin(), ori_polygon.end(), middle_polygon.begin());
    if (smooth && (roi_type == RoiType::TRADED_ROI || roi_type == RoiType::FREEROI ||
                   roi_type == RoiType::FREELINE)) {
        constexpr float smooth_k         = 1.0f / 3.0f;
        int             middle_poly_size = middle_polygon.size();
        // smooth the point with 3 neigh,prev,self,and next along the clockwise !
        for (int i = 1; i < ori_poly_size - 2; ++i) {
            middle_polygon[i].x =
                (middle_polygon[i - 1].x + middle_polygon[i].x + middle_polygon[i + 1].x) *
                smooth_k;
            middle_polygon[i].y =
                (middle_polygon[i - 1].y + middle_polygon[i].y + middle_polygon[i + 1].y) *
                smooth_k;
        }
        if (roi_type != RoiType::FREELINE) {
            // process the first and last point! so easy so simple!
            middle_polygon[0].x = (middle_polygon[middle_poly_size - 1].x + middle_polygon[0].x +
                                   middle_polygon[1].x) *
                                  smooth_k;
            middle_polygon[0].y = (middle_polygon[middle_poly_size - 1].y + middle_polygon[0].y +
                                   middle_polygon[1].y) *
                                  smooth_k;

            middle_polygon[middle_poly_size - 1].x =
                (middle_polygon[middle_poly_size - 2].x + middle_polygon[middle_poly_size - 1].x +
                 middle_polygon[0].x) *
                smooth_k;
            middle_polygon[middle_poly_size - 1].y =
                (middle_polygon[middle_poly_size - 2].y + middle_polygon[middle_poly_size - 1].y +
                 middle_polygon[0].y) *
                smooth_k;
        }
    }
    // stop smooth!
    if (middle_polygon.size() <= 2) {
        out_polygon.resize(middle_polygon.size());
        std::copy(middle_polygon.begin(), middle_polygon.end(), out_polygon.begin());
        return;
    }
    bool allow_to_adjust = interval < 0.0;
    if (interval < 0) {
        interval = interval * -1.0;
    }
    double side_len = compute_polygon_side_len<float, true>(middle_polygon);
    if (interval <= 0.01) {
        LOG_ERROR("Interval must >= 0.01,bug got {}", interval);
        out_polygon.resize(middle_polygon.size());
        std::copy(middle_polygon.begin(), middle_polygon.end(), out_polygon.begin());
        return;
    }

    bool is_line = roi_is_line(roi_type);
    // expand the polygon...

    if (!is_line) {
        middle_polygon.push_back(middle_polygon[0]);
    }

    int estimate_dst_poly_size = static_cast<int>(10 + (side_len * 1.5) / interval);

    double try_interval  = interval;
    double min_diff      = 1e9;
    double best_interval = 0.0;
    out_polygon.reserve(estimate_dst_poly_size);

    int n_trials = 50;
    int trial    = 0;

    int src_idx = 0;
    int dst_idx = 0;

    Coordinate2df64 intersections[2];
    int             middle_poly_size = middle_polygon.size();
    while (trial <= n_trials) {
        out_polygon.emplace_back(middle_polygon[0].x, middle_polygon[0].y);
        src_idx   = 0;
        dst_idx   = 0;
        double xA = middle_polygon[0].x;
        double yA = middle_polygon[0].y;

        while (src_idx < middle_poly_size - 1) {
            double xC = out_polygon[dst_idx].x;
            double yC = out_polygon[dst_idx].y;
            double xB = middle_polygon[src_idx + 1].x;
            double yB = middle_polygon[src_idx + 1].y;
            int    inter_num =
                line_circle_intersection(xA, yA, xB, yB, xC, yC, try_interval, true, intersections);
            if (inter_num >= 1) {
                ++dst_idx;
                out_polygon.emplace_back(intersections[0].x, intersections[0].y);
            } else {
                ++src_idx;
                xA = middle_polygon[src_idx].x;
                yA = middle_polygon[src_idx].y;
            }
        }
        ++dst_idx;
        out_polygon.push_back(middle_polygon[middle_poly_size - 1]);
        ++dst_idx;
        if (!allow_to_adjust) {
            if (is_line) {
                --dst_idx;
            }
            break;
        }
        int    n_segment    = dst_idx - 1;
        double dx           = out_polygon[dst_idx - 2].x - out_polygon[dst_idx - 1].x;
        double dy           = out_polygon[dst_idx - 2].y - out_polygon[dst_idx - 1].y;
        double last_segment = std::sqrt(dx * dx + dy * dy);
        double diff         = last_segment - try_interval;
        if (std::abs(diff) < min_diff) {
            min_diff      = std::abs(diff);
            best_interval = try_interval;
        }
        double feedback_factor = 0.66;
        try_interval           = try_interval + feedback_factor * diff / n_segment;
        if ((try_interval < 0.8 * interval || std::abs(diff) < 0.05 || trial == n_trials - 1) &&
            trial < n_trials) {
            trial        = n_trials;
            try_interval = best_interval;
        } else {
            ++trial;
        }
    }
    // removing closing point from end of array!
    if (!is_line) {
        out_polygon.resize(out_polygon.size() - 1);
    }
}

// need to check,this function is pretty hard!
void smooth_polygon_roi(const PolygonTypef32& polygon, PolygonTypef32& smoothed_polygon) {
    int poly_size = polygon.size();
    // allocate the buffer!
    smoothed_polygon.reserve((poly_size + 1) / 2);
    // mean poly smoth with 3 point!
    constexpr float k = 1.0 / 3.0;
    for (int idx = 0; idx < poly_size; idx += 2) {
        // perfect!
        int   idx_prev = (idx + poly_size - 1) % poly_size;
        int   idx_next = (idx + 1) % poly_size;
        float x        = (polygon[idx_prev].x + polygon[idx].x + polygon[idx_next].x) * k;
        float y        = (polygon[idx_prev].y + polygon[idx].y + polygon[idx_next].y) * k;
        smoothed_polygon.emplace_back(x, y);
    }
}

PolygonTypef32 smooth_polygon_roi(const PolygonTypef32& polygon) {
    PolygonTypef32 smoothed_polygon;
    smooth_polygon_roi(polygon, smoothed_polygon);
    return smoothed_polygon;
}

void scale_polygon_roi_inplace(PolygonTypef32& polygon, float x_offset, float y_offset,
                               float downsample_factor) {
    if (x_offset < 0.0) {
        x_offset = -1.0 * x_offset;
    }
    if (y_offset < 0.0) {
        y_offset = -1.0 * y_offset;
    }
    for (size_t i = 0; i < polygon.size(); ++i) {
        polygon[i].x = (polygon[i].x + x_offset) * downsample_factor;
        polygon[i].y = (polygon[i].y + y_offset) * downsample_factor;
    }
}

void scale_polygon_roi(const PolygonTypef32& polygon, PolygonTypef32& scaled_polygon,
                       float x_offset, float y_offset, float downsample_factor) {
    scaled_polygon.resize(polygon.size());
    if (x_offset < 0.0) {
        x_offset = -1.0 * x_offset;
    }
    if (y_offset < 0.0) {
        y_offset = -1.0 * y_offset;
    }
    for (size_t i = 0; i < polygon.size(); ++i) {
        scaled_polygon[i].x = (polygon[i].x + x_offset) * downsample_factor;
        scaled_polygon[i].y = (polygon[i].y + y_offset) * downsample_factor;
    }
}

PolygonTypef32 scale_polygon_roi(const PolygonTypef32& polygon, float x_offset, float y_offset,
                                 float downsample_factor) {
    PolygonTypef32 scaled_polygon;
    scale_polygon_roi(polygon, scaled_polygon, x_offset, y_offset, downsample_factor);
    return scaled_polygon;
}

}   // namespace polygon
}   // namespace image_proc
}   // namespace fish
