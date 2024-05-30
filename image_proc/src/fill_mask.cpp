#include "image_proc/fill_mask.h"
#include "common/fishdef.h"
#include "core/mat.h"
#include "image_proc/polygon.h"
#include "utils/logging.h"
#include <cstdint>
#include <vector>

namespace fish {
namespace image_proc {
namespace fill_mask {
void PolygonFiller::build_edge_table(const Coordinate2d* points, int point_size) {
    edge_num       = 0;
    int poly_start = 0;
    LOG_INFO("points size:{}", point_size);
    // 这里就是计算顺时针排列后,两两连线之间的斜率,只不过是相交于y轴(height方向)的斜率
    for (int i = 0; i < point_size; ++i) {
        // 这里要循环遍历,所以最后一个点连接的是第一个点
        // if i == last_points,the next point should be the first point!
        int next_idx = (i + 1) % point_size;
        int x1       = points[i].x;
        int y1       = points[i].y;
        int x2       = points[next_idx].x;
        int y2       = points[next_idx].y;
        //忽略水平线
        if (y1 == y2) {
            continue;
        }
        // keep the maximum is y2,minimum is y1
        if (y1 > y2) {
            // swap them
            int temp = x1;
            x1       = x2;
            x2       = temp;

            temp = y1;
            y1   = y2;
            y2   = temp;
        }

        // 计算斜率(分母是y)
        double slope                 = static_cast<double>(x2 - x1) / static_cast<double>(y2 - y1);
        edge_x_coors[edge_num]       = static_cast<double>(x1) + 0.5 * slope + 1e-8;
        edge_lower_y_coors[edge_num] = y1;
        edge_upper_y_coors[edge_num] = y2;
        edge_slopes[edge_num]        = slope;
        y_min                        = FISH_MIN(y_min, y1);
        y_max                        = FISH_MAX(y_max, y2);
        ++edge_num;
    }
    for (int i = 0; i < edge_num; ++i) {
        sorted_edges[i] = i;
    }
    //实际参与计算的边
    activate_edge_num = 0;
}

void PolygonFiller::shift_x_values_and_activate(int y_start) {
    for (int i = 0; i < edge_num; ++i) {
        int index = sorted_edges[i];
        //如果在区间内
        if (edge_lower_y_coors[i] < y_start && y_start <= edge_upper_y_coors[i]) {
            //计算交点
            edge_x_coors[i] += edge_slopes[index] * (y_start - edge_lower_y_coors[index]);
            activate_edges[activate_edge_num] = index;
            ++activate_edge_num;
        }
    }
    sort_activate_edges();
}


void PolygonFiller::sort_activate_edges() {
    int min_index;
    for (int i = 0; i < activate_edge_num; ++i) {
        min_index = i;
        //交换排序 O(N^2)
        for (int j = i; j < activate_edge_num; ++j) {
            //记录最小的x坐标的位置
            if (edge_x_coors[j] < edge_x_coors[activate_edges[min_index]]) {
                min_index = j;
            }
            //交换,按照升序排列
            std::swap(activate_edges[min_index], i);
        }
    }
}

//数据量较大时,std::sort要快一点
void PolygonFiller::sort_activate_edges_with_std() {
    //升序排列
    auto sort_op = [this](int i, int j) { return edge_x_coors[i] < edge_x_coors[j]; };
    std::sort(activate_edges.begin(), activate_edges.end(), sort_op);
}

void PolygonFiller::apply_activate_edges(int y) {
    for (int i = 0; i < edge_num; ++i) {
        int edge = sorted_edges[i];
        if (y == edge_lower_y_coors[edge]) {
            int index = 0;
            //找到不大于当前边的x
            while (index < activate_edge_num &&
                   edge_x_coors[edge] > edge_x_coors[activate_edges[index]]) {
                ++index;
            }
            //逆序访问,不用担心越界
            for (int j = activate_edge_num - 1; j >= index; --j) {
                activate_edges[j + 1] = activate_edges[j];
            }
            activate_edges[index] = edge;
            ++activate_edge_num;
        }
    }
}

void PolygonFiller::remove_inactivate_edges(int y) {
    int i = 0;
    while (i < activate_edge_num) {
        int index = activate_edges[i];
        //如果越界了
        if (y < edge_lower_y_coors[index] || y >= edge_upper_y_coors[index]) {
            for (int j = i; j < activate_edge_num - 1; ++j) {
                activate_edges[j] = activate_edges[j + 1];
            }
            --activate_edge_num;
        } else {
            ++i;
        }
    }
}

void PolygonFiller::update_x_coors() {
    int    idx;
    double x1 = std::numeric_limits<double>::lowest();
    double x2;
    bool   sorted = true;
    for (int i = 0; i < activate_edge_num; ++i) {
        idx               = activate_edges[i];
        x2                = edge_x_coors[idx] + edge_slopes[idx];
        edge_x_coors[idx] = x2;
        // if sorted,the x2 should greater than x1
        sorted = (x2 >= x1);
        x1     = x2;
    }
    if (!sorted) {
        sort_activate_edges();
    }
}

void fill_continous_memory(void* dst, int data_size, uint8_t fill_value) {
    uint8_t*  buf  = reinterpret_cast<uint8_t*>(dst);
    uintptr_t addr = reinterpret_cast<uintptr_t>(dst);
    // if the addr % word != 0,we firstly copy these unaligned data one by one,then copy the remains
    // by word!
    constexpr int aligned_byte     = sizeof(size_t);
    uintptr_t     word_addr        = addr / aligned_byte * aligned_byte;
    int           not_aligned_size = 0;
    if (word_addr < addr) {
        not_aligned_size = word_addr + aligned_byte - addr;
        not_aligned_size = FISH_MIN(not_aligned_size, data_size);
        // copy one by one!
        for (int i = 0; i < not_aligned_size; ++i) {
            buf[i] = fill_value;
        }
        data_size -= not_aligned_size;
    }
    int   fill_value_word;
    char* fill_value_word_ptr = reinterpret_cast<char*>(&fill_value_word);

    // fill by word!
    if constexpr (aligned_byte == 4) {
        fill_value_word_ptr[0] = fill_value;
        fill_value_word_ptr[1] = fill_value;
        fill_value_word_ptr[2] = fill_value;
        fill_value_word_ptr[3] = fill_value;
    } else if constexpr (aligned_byte == 8) {
        fill_value_word_ptr[0] = fill_value;
        fill_value_word_ptr[1] = fill_value;
        fill_value_word_ptr[2] = fill_value;
        fill_value_word_ptr[3] = fill_value;
        fill_value_word_ptr[4] = fill_value;
        fill_value_word_ptr[5] = fill_value;
        fill_value_word_ptr[6] = fill_value;
        fill_value_word_ptr[7] = fill_value;
    }
    // copy by word!
    size_t* buf_word = reinterpret_cast<size_t*>(buf + not_aligned_size);
    for (int i = 0; i < data_size / aligned_byte; ++i) {
        buf_word[i] = fill_value_word;
    }

    // copy one by one!
    for (int i = data_size * aligned_byte / aligned_byte; i < data_size; ++i) {
        buf[i + not_aligned_size] = fill_value;
    }
}

// here we assume that the coor of point and coor of mask is same!
void PolygonFiller::fill_polygon_impl(const Coordinate2d* points, int point_size,
                                      ImageMat<uint8_t>& mask, uint8_t fill_value) {
    // make sure the point value less than height and width!
    int height = mask.get_height();
    int width  = mask.get_width();
    // if you not do this,will got error!
    allocate_sources(point_size);
    build_edge_table(points, point_size);
    int x1, x2, offset, index;
    int y_start = FISH_MAX(y_min, 0);
    if (y_min != 0) {
        shift_x_values_and_activate(y_start);
    }
    uint8_t* mask_ptr = mask.get_data_ptr();
    int      y_end    = FISH_MIN(height, y_max + 1);
    for (int y = y_start; y < y_end; ++y) {
        remove_inactivate_edges(y);
        apply_activate_edges(y);
        for (int i = 0; i < activate_edge_num / 2 * 2; i += 2) {
            // compute the edge with slope!
            x1                       = static_cast<int>(edge_x_coors[activate_edges[i]] + 0.5);
            x1                       = FISH_CLIP(x1, 0, width);
            x2                       = static_cast<int>(edge_x_coors[activate_edges[i + 1]] + 0.5);
            x2                       = FISH_CLIP(x2, 0, width);
            int      ptr_offset      = width * y + x1;
            uint8_t* filled_mask_ptr = mask_ptr + ptr_offset;
            fill_continous_memory(filled_mask_ptr, x2 - x1, fill_value);
            update_x_coors();
        }
    }
}

ImageMat<uint8_t> PolygonFiller::fill_polygon(const Coordinate2d* points, int point_size,
                                              int height, int width, uint8_t fill_value) {
    Rectangle bound = get_bounding_box(points, point_size);
    int       h     = bound.height;
    int       w     = bound.width;
    int       x0    = bound.x;
    int       y0    = bound.y;

    ImageMat<uint8_t> mask(h, w, 1, MatMemLayout::LayoutRight);
    // make sure the point value less than height and width!
    // if you not do this,will got error!
    allocate_sources(point_size);
    build_edge_table(points, point_size);
    int x1, x2, offset, index;
    int y_start = FISH_MAX(y_min, 0);
    if (y_min != 0) {
        shift_x_values_and_activate(y_start);
    }
    uint8_t* mask_ptr = mask.get_data_ptr();
    int      y_end    = FISH_MIN(height, y_max + 1);
    for (int y = y_start; y < y_end; ++y) {
        remove_inactivate_edges(y);
        apply_activate_edges(y);
        for (int i = 0; i < activate_edge_num / 2 * 2; i += 2) {
            // compute the edge with slope!
            x1 = static_cast<int>(edge_x_coors[activate_edges[i]] + 0.5);
            x1 = FISH_CLIP(x1, 0, width);
            x2 = static_cast<int>(edge_x_coors[activate_edges[i + 1]] + 0.5);
            x2 = FISH_CLIP(x2, 0, width);
            // shift to (0,0)
            int      ptr_offset      = w * (y - y0) + (x1 - x0);
            uint8_t* filled_mask_ptr = mask_ptr + ptr_offset;
            fill_continous_memory(filled_mask_ptr, x2 - x1, fill_value);
            update_x_coors();
        }
    }
    return mask;
}
}   // namespace fill_mask
}   // namespace image_proc
}   // namespace fish