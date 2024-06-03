#pragma once
#include "common/fishdef.h"
#include "core/mat.h"
#include "image_proc/polygon.h"
#include "utils/logging.h"
#include <cstdint>
#include <vector>

namespace fish {
namespace image_proc {
namespace fill_mask {
using namespace fish::image_proc::polygon;
using namespace fish::core;
// we always fill the int polygon,not consider the float polygon,just for smooth,the float!
struct PolygonWithMask {
    PolygonType       polygon;
    ImageMat<uint8_t> mask;
    // the coordinate of left upper
    int x0;
    int y0;
};


class PolygonFiller {
public:
    static constexpr int BLACK = 0xff000000;
    static constexpr int WHITE = 0xffffffff;

private:
    int                 edge_num;
    int                 activate_edge_num;
    std::vector<double> edge_x_coors;
    std::vector<int>    edge_upper_y_coors;
    std::vector<int>    edge_lower_y_coors;
    std::vector<double> edge_slopes;
    //记录point中最小点坐标
    int y_min;
    //记录point中最大点坐标
    int              y_max;
    std::vector<int> activate_edges;
    //排序后的边
    std::vector<int> sorted_edges;
    // std::vector<Coordinate2d>* points;
    // avoid copy memory... you can pass me a pointer!
public:
    PolygonFiller(){};

private:
    void build_edge_table(const Coordinate2d* points, int point_size);

    void allocate_sources(int point_size) {
        // reset all value to default
        y_min             = std::numeric_limits<int>::max();
        y_max             = std::numeric_limits<int>::min();
        edge_num          = 0;
        activate_edge_num = 0;
        // reuse the memory!
        uintptr_t p = reinterpret_cast<uintptr_t>(edge_lower_y_coors.data());
        if (point_size == 52) {
            LOG_INFO("error...");
        }
        edge_x_coors.resize(point_size);
        LOG_INFO("prev_size:{} current_size:{} cap:{} lower_y_ptr:{} upper_y_ptr:{} edge_x_ptr:{}",
                 edge_upper_y_coors.size(),
                 point_size,
                 edge_upper_y_coors.capacity(),
                 reinterpret_cast<uintptr_t>(edge_lower_y_coors.data()),
                 reinterpret_cast<uintptr_t>(edge_upper_y_coors.data()),
                 reinterpret_cast<uintptr_t>(edge_x_coors.data()));
        // resize upper broken...
        edge_upper_y_coors.resize(point_size);
        edge_lower_y_coors.resize(point_size);
        edge_slopes.resize(point_size);
        activate_edges.resize(point_size);
        sorted_edges.resize(point_size);
    }

    void shift_x_values_and_activate(int y_start);
    //进行排序
    void sort_activate_edges();

    void sort_activate_edges_with_std();

    void remove_inactivate_edges(int y);

    void apply_activate_edges(int y);

    void update_x_coors();


    void fill_polygon_impl(const Coordinate2d* points, int point_size, ImageMat<uint8_t>& mask,
                           uint8_t fill_value);

public:
    // this function need we provide a mask have same shape with original image!
    void fill_polygon(const Coordinate2d* points, int point_size, ImageMat<uint8_t>& mask,
                      uint8_t fill_value = 255) {
        fill_polygon_impl(points, point_size, mask, fill_value);
    }

    // in this case,we will transfomr the left upper to (0,0) because we do not know the info of ori
    // image! so we maybe generate a point copy,so this is not very good!
    ImageMat<uint8_t> fill_polygon(const Coordinate2d* points, int point_size,
                                   uint8_t fill_value = 255) {
        Rectangle bound = get_bounding_box(points, point_size);
        int       h     = bound.height;
        int       w     = bound.width;
        int       x0    = bound.x;
        int       y0    = bound.y;

        ImageMat<uint8_t>         mask(h, w, 1, MatMemLayout::LayoutRight);
        std::vector<Coordinate2d> translation_points(point_size);
        for (int i = 0; i < point_size; ++i) {
            translation_points[i].x = points[i].x - x0;
            translation_points[i].y = points[i].y - y0;
        }
        fill_polygon_impl(translation_points.data(), point_size, mask, fill_value);
        return mask;
    }

    ImageMat<uint8_t> fill_polygon(const Coordinate2d* points, int point_size, Rectangle& bound,
                                   uint8_t fill_value = 255) {
        int h  = bound.height;
        int w  = bound.width;
        int x0 = bound.x;
        int y0 = bound.y;

        ImageMat<uint8_t>         mask(h, w, 1, MatMemLayout::LayoutRight);
        std::vector<Coordinate2d> translation_points(point_size);
        for (int i = 0; i < point_size; ++i) {
            translation_points[i].x = points[i].x - x0;
            translation_points[i].y = points[i].y - y0;
        }
        fill_polygon_impl(translation_points.data(), point_size, mask, fill_value);
        return mask;
    }


    // we need this to compute the static of image data!
    //  this func will change the points value,be sure you will not use it anymore...
    ImageMat<uint8_t> fill_polygon_inplace(Coordinate2d* points, int point_size,
                                           uint8_t fill_value = 255) {
        Rectangle bound = get_bounding_box(points, point_size);
        int       h     = bound.height;
        int       w     = bound.width;
        int       x0    = bound.x;
        int       y0    = bound.y;

        ImageMat<uint8_t> mask(h, w, 1, MatMemLayout::LayoutRight);
        for (int i = 0; i < point_size; ++i) {
            points[i].x -= x0;
            points[i].y -= y0;
        }
        fill_polygon_impl(points, point_size, mask, fill_value);
        return mask;
    }

    // this function do not need to transilation!
    ImageMat<uint8_t> fill_polygon(const Coordinate2d* points, int point_size, int height,
                                   int width, uint8_t fill_value);
};
}   // namespace fill_mask
}   // namespace image_proc
}   // namespace fish