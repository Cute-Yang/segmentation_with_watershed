#pragma once
#include "common/fishdef.h"
#include "core/mat.h"
#include "image_proc/polygon.h"
#include <cmath>
#include <vector>


namespace fish {
namespace image_proc {
namespace contour {
using namespace fish::core;
using namespace fish::image_proc::polygon;
namespace WandMode {
constexpr size_t LEGACY_MODE      = 1UL;
constexpr size_t FOUR_CONNECTED   = 1UL << 2;   // 0x100
constexpr size_t EIGHT_CONNECTAED = 1UL << 3;
constexpr size_t THRESHHOLD_MODE  = 1UL << 8;
}   // namespace WandMode

enum class TraceConnectiveType : uint32_t { FourConnective = 0, EightConnective = 1 };

template<class T, typename = image_dtype_limit<T>> class Wand {
    constexpr static size_t max_initialize_point_size = 1000;

private:
    std::vector<Coordinate2d> points;
    int                       xmin;
    T                         lower_threshold;
    T                         upper_threshold;

private:
    FISH_ALWAYS_INLINE bool inside(int x, int y, const ImageMat<T>& image) {
        int height = image.get_height();
        int width  = image.get_width();
        // the coor shoud be invalid and the pixel value in given interval!
        bool point_is_valid =
            (x >= 0 && x < width && y >= 0 && y < height && image(y, x, 0) >= lower_threshold &&
             image(y, x, 0) <= upper_threshold);
        return point_is_valid;
    }
    bool is_line(int xs, int ys, const ImageMat<T>& image) {
        int           height = image.get_height();
        int           width  = image.get_width();
        constexpr int r      = 5;
        int           xmin   = xs;
        int           xmax   = xs + 2 * r;
        if (xmax >= width) [[unlikely]] {
            xmax = width - 1;
        }
        int ymin = ys;
        if (ymin < 0) [[unlikely]] {
            ymin = 0;
        }
        int ymax = ys + r;
        if (ymax >= height) [[unlikely]] {
            ymax = height - 1;
        }
        int area         = 0;
        int inside_count = 0;
        for (int y = ymin; y <= ymax; ++y) {
            for (int x = xmin; x <= xmax; ++x) {
                ++area;
                if (inside(x, y, image)) {
                    ++inside_count;
                }
            }
        }
        constexpr double rate_upper = 0.25;
        // 这里area 一定 > 0
        double rate = static_cast<double>(inside_count) / area;
        return rate < rate_upper;
    }


    FISH_ALWAYS_INLINE bool inside(int x, int y, int direction, const ImageMat<T>& image) {
        direction &= 3;
        if (direction == 0) {
            return inside(x, y, image);
        } else if (direction == 1) {
            return inside(x, y - 1, image);
        } else if (direction == 2) {
            return inside(x - 1, y - 1, image);
        }
        return inside(x - 1, y, image);
    }
    FISH_ALWAYS_INLINE void add_point(int x, int y) { points.emplace_back(x, y); }
    template<TraceConnectiveType conn_type>
    bool trace_edge(int start_x, int start_y, const ImageMat<T>& image) {
        int width = image.get_width();
        xmin      = width;
        points.resize(0);
        int start_direction;
        if (inside(start_x, start_y, image)) {
            start_direction = 1;
        } else {
            start_direction = 3;
            ++start_y;
        }
        int x         = start_x;
        int y         = start_y;
        int direction = start_direction;
        while (true) {
            int new_direction;
            if constexpr (conn_type == TraceConnectiveType::FourConnective) {
                //循环次数展开为2
                new_direction = direction;
                // for the first time!
                if (inside(x, y, new_direction, image)) {
                    ++new_direction;
                    // for the second time
                    if (inside(x, y, new_direction, image)) {
                        ++new_direction;
                    }
                }
                --new_direction;
            } else {
                new_direction = direction + 1;
                if (!inside(x, y, new_direction, image)) {
                    --new_direction;
                }
            }
            if (new_direction != direction) {
                add_point(x, y);
            }
            switch (new_direction & 3) {
            case 0: ++x; break;
            case 1: --y; break;
            case 2: --x; break;
            case 3: ++y; break;
            default: break;
            }
            direction = new_direction;
            //当重新回到起点的时候,停止循环
            if (x == start_x && y == start_y && (direction & 3) == start_direction) {
                break;
            }
        }

        //不闭合的多边形
        if (points[0].x != x) {
            add_point(x, y);
        }
        return (direction <= 0);
    }


    template<TraceConnectiveType conn_type>
    bool auto_outline_impl(int x, int y, int start_y, int seed_x, const ImageMat<T>& image) {
        int  width = image.get_width();
        bool first = true;
        while (true) {
            bool inside_selected = trace_edge<conn_type>(x, y, image);
            if (inside_selected) {
                if (first) {
                    return true;
                }
                if (xmin <= seed_x) {
                    if (point_in_polygon(points, seed_x, start_y)) {
                        return true;
                    }
                }
            }
            first = false;
            if (!inside(x, y, image)) {
                while (true) {
                    ++x;
                    // 越界
                    // should never happen!
                    if (x > width) [[unlikely]] {
                        return false;
                    }
                    if (inside(x, y, image)) {
                        break;
                    }
                }
            }
            while (true) {
                ++x;
                if (!inside(x, y, image)) {
                    break;
                }
            }
        }
    }

public:
    Wand(size_t point_capacity)
        : points() {
        points.reserve(point_capacity);
    }
    Wand()
        : points() {
        points.reserve(max_initialize_point_size);
    }
    Wand(const Wand& rhs)            = delete;
    Wand(Wand&& rhs)                 = delete;
    Wand& operator=(const Wand& rhs) = delete;
    Wand& operator=(Wand&& rhs)      = delete;

    bool auto_outline(const ImageMat<T>& image, int start_x, int start_y, float tolerance,
                      int mode) {
        // avoid trash...
        this->reset_source();
        bool thresh_mode = (mode & WandMode::THRESHHOLD_MODE) != 0;
        if (!thresh_mode) {
            float value = static_cast<float>(image(start_y, start_x));
            // 这里需要做cast
            if constexpr (FloatTypeRequire<T>::value) {
                lower_threshold = value - tolerance;
                upper_threshold = value + tolerance;
            } else {
                // need to do a cast with trick
                // for example,if the lower thresh is 2.3,and we got a value 2.0,should give up this
                // value! so the value should apply a ceil op
                lower_threshold = static_cast<T>(std::ceil(value - tolerance));
                // and the upper_threhsold,for example,if 4.8 and we should keep 4 and give 5,so
                // apply floor op
                lower_threshold = static_cast<T>(std::floor(value + tolerance));
            }
        }
        int  width       = image.get_width();
        bool legacy_mode = (mode & WandMode::LEGACY_MODE) != 0 && tolerance != 0;
        int  x           = start_x;
        int  y           = start_y;
        int  seed_x;
        // the start of the edge
        if (inside(x, y, image)) {
            seed_x = x;
            // do {
            //     ++x;
            // } while (inside(x, y));
            while (true) {
                ++x;
                //这里自带越界判断
                if (!inside(x, y, image)) {
                    break;
                }
            }
        } else {
            // the end of the start
            while (true) {
                ++x;
                //越界返回
                if (x >= width) {
                    return false;
                }
                //寻找边界外面
                if (inside(x, y, image)) {
                    break;
                }
            }
            seed_x = x;
        }
        bool conn_4;
        if (legacy_mode) [[unlikely]] {
            conn_4 = !thresh_mode && !is_line(x, y, image);
        } else {
            conn_4 = (mode & WandMode::FOUR_CONNECTED) != 0;
        }
        bool first = true;
        if (conn_4) {
            return auto_outline_impl<TraceConnectiveType::FourConnective>(
                x, y, start_y, seed_x, image);
        } else {
            return auto_outline_impl<TraceConnectiveType::EightConnective>(
                x, y, start_y, seed_x, image);
        }
    }

    bool auto_outline(const ImageMat<T>& image, int start_x, int start_y, int mode) {
        return auto_outline(image, start_x, start_y, 0, mode | WandMode::THRESHHOLD_MODE);
    }
    bool auto_outline(const ImageMat<T>& image, int start_x, int start_y) {
        return auto_outline(
            image, start_x, start_y, 0.0, WandMode::LEGACY_MODE | WandMode::THRESHHOLD_MODE);
    }

    // perfect!
    void set_lower_threshold(float threshold) {
        //
        if constexpr (FloatTypeRequire<T>::value) {
            lower_threshold = threshold;
        } else {
            // apply ceil clip
            lower_threshold = static_cast<T>(std::ceil(threshold));
        }
    }
    void set_upper_threshold(T threshold) {
        if constexpr (FloatTypeRequire<T>::value) {
            upper_threshold = threshold;
        } else {
            // need to apply floor !
            upper_threshold = static_cast<T>(std::floor(threshold));
        }
    }
    std::vector<Coordinate2d>&       get_points_ref() { return points; }
    const std::vector<Coordinate2d>& get_points_cref() const { return points; }
    std::vector<Coordinate2d>        get_points() { return points; }

    std::vector<Coordinate2d> get_points() const { return points; }
    int                       get_npoint() const { return points.size(); }

    FISH_ALWAYS_INLINE void reset_source() {
        xmin = 0;
        points.resize(0);
    }
};

}   // namespace contour
}   // namespace image_proc
}   // namespace fish