#include "segmentation/shape_simplifier.h"
#include "common/fishdef.h"
#include "core/mat.h"
#include "image_proc/polygon.h"
#include <cmath>
#include <queue>
#include <set>
#include <vector>
namespace fish {
namespace segmentation {
namespace shape_simplier {
using namespace fish::core;
using namespace fish::image_proc::polygon;
template<class T>
double calculate_triple_area(const GenericCoordinate2d<T>& p1, const GenericCoordinate2d<T>& p2,
                             const GenericCoordinate2d<T>& p3) {
    T      value = (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y));
    double area  = 0.5 * static_cast<double>(value);
    return area;
}

template<class T> struct PointWithArea {
    PointWithArea*         prev;
    PointWithArea*         next;
    GenericCoordinate2d<T> p;
    double                 area;

    PointWithArea(GenericCoordinate2d<T>& p_, double area_)
        : p(p_)
        , area(area_) {}

    PointWithArea(T x, T y, double area_)
        : p(x, y)
        , area(area_) {}

    void set_area(double area_) { area = area_; }

    const GenericCoordinate2d<T>& get_point_ref() { return p; }

    void update_area() {
        area = calculate_triple_area(prev->get_point_ref(), p, next->get_point_ref());
    }

    T get_x() const { return p.x; }

    T get_y() const { return p.y; }

    double get_area() const { return area; }

    void set_prev(PointWithArea<T>* prev_) { prev = prev_; }

    void set_next(PointWithArea<T>* next_) { next = next_; }

    PointWithArea<T>* get_prev() { return prev; }
    PointWithArea<T>* get_next() { return next; }

    bool operator<(const PointWithArea<T>& rhs) { return area < rhs.area; }

    bool operator==(const PointWithArea<T>& rhs) { return area == rhs.area; }

    bool operator>(const PointWithArea<T>& rhs) { return area > rhs.area; }
};

using PointWithAreaF32 = PointWithArea<float>;
using PointWithAreaF64 = PointWithArea<double>;


// min heap!
template<class T> struct PwaCompare {
    bool operator()(const PointWithArea<T>* lhs, const PointWithArea<T>* rhs) {
        return lhs->area > rhs->area;
    }
};

PolygonTypef32 simplify_polygon_points(const PolygonTypef32& polygon, float altitude_threshold) {
    if (polygon.size() <= 1) {
        return polygon;
    }
    // make the uinque...
    PolygonTypef32 removed_adjacent_polygon;
    removed_adjacent_polygon.reserve(polygon.size());
    Coordinate2df32 last_point = polygon[0];
    removed_adjacent_polygon.push_back(polygon[0]);
    for (size_t i = 1; i < polygon.size(); ++i) {
        if (polygon[i] != last_point) {
            removed_adjacent_polygon.push_back(polygon[i]);
            last_point = polygon[i];
        }
    }
    if (last_point == polygon[0]) {
        // delete the last!
        removed_adjacent_polygon.resize(removed_adjacent_polygon.size() - 1);
    }
    if (removed_adjacent_polygon.size() <= 3) {
        return removed_adjacent_polygon;
    }

    int n = removed_adjacent_polygon.size();

    using QueueType =
        std::priority_queue<PointWithAreaF32*, std::vector<PointWithAreaF32*>, PwaCompare<float>>;

    QueueType point_queue;

    Coordinate2df32               prev_point    = removed_adjacent_polygon[n - 1];
    Coordinate2df32               current_point = removed_adjacent_polygon[0];
    PointWithAreaF32*             prev_pwa      = nullptr;
    PointWithAreaF32*             first_pwa     = nullptr;
    std::vector<PointWithAreaF32> pwa_pool;
    pwa_pool.reserve(n);

    // building the queue
    for (int i = 0; i < n; ++i) {
        Coordinate2df32 next_point = removed_adjacent_polygon[(i + 1) % n];
        double          area       = calculate_triple_area(prev_point, current_point, next_point);
        pwa_pool.emplace_back(current_point, area);
        PointWithAreaF32* pwa = pwa_pool.data() + pwa_pool.size();
        pwa->set_prev(prev_pwa);
        if (prev_pwa != nullptr) {
            prev_pwa->set_next(pwa);
        }
        point_queue.push(pwa);
        prev_pwa   = pwa;
        prev_point = current_point;
        if (i == n - 1) {
            pwa->set_next(first_pwa);
            first_pwa->set_prev(pwa);
        } else if (i == 0) {
            first_pwa = pwa;
        }
    }

    double max_area = 0;
    int    min_size = FISH_MIN(n / 100, 3);

    std::set<Coordinate2df32> remove_points;

    while (point_queue.size() > min_size) {
        PointWithAreaF32* pwa = point_queue.top();
        point_queue.pop();
        auto&  _next_p = pwa->get_next()->p;
        auto&  _prev_p = pwa->get_prev()->p;
        float  dx      = _next_p.x - _prev_p.x;
        float  dy      = _next_p.y - _prev_p.y;
        double dist    = std::sqrt(dx * dx + dy * dy) + 1e-9;
        double altitue = pwa->get_area() * 2 / dist;
        if (altitue > altitude_threshold) {
            break;
        }
        if (pwa->get_area() < max_area) {
            pwa->set_area(max_area);
        } else {
            max_area = pwa->get_area();
        }
        remove_points.insert(pwa->get_point_ref());
        prev_pwa                   = pwa->get_prev();
        PointWithAreaF32* next_pwa = pwa->get_next();
        prev_pwa->set_next(next_pwa);
        next_pwa->set_prev(prev_pwa);

        // this is unsafe!
        prev_pwa->update_area();
        next_pwa->update_area();
        // remvoe current and rebuild
        // maybe slow,but I don't have a better method...
        for (size_t i = 0; i < point_queue.size(); ++i) {
            auto* element_ptr = point_queue.top();
            point_queue.pop();
            point_queue.push(element_ptr);
        }
        // adjust the priority!
    }
    if (remove_points.size() == 0) {
        return removed_adjacent_polygon;
    }
    PolygonTypef32 ret_polygon;
    ret_polygon.reserve(removed_adjacent_polygon.size() - remove_points.size());
    for (size_t i = 0; i < removed_adjacent_polygon.size(); ++i) {
        if (remove_points.find(removed_adjacent_polygon[i]) != remove_points.end()) {
            ret_polygon.push_back(removed_adjacent_polygon[i]);
        }
    }
    return ret_polygon;
}


PolygonTypef32 simplify_polygon_points_better(const PolygonTypef32& polygon,
                                              float                 altitude_threshold) {
    if (polygon.size() <= 1) {
        return polygon;
    }
    // make the uinque...
    PolygonTypef32 removed_adjacent_polygon;
    removed_adjacent_polygon.reserve(polygon.size());
    Coordinate2df32 last_point = polygon[0];
    removed_adjacent_polygon.push_back(polygon[0]);
    for (size_t i = 1; i < polygon.size(); ++i) {
        if (polygon[i] != last_point) {
            removed_adjacent_polygon.push_back(polygon[i]);
            last_point = polygon[i];
        }
    }
    if (last_point == polygon[0]) {
        // delete the last!
        removed_adjacent_polygon.resize(removed_adjacent_polygon.size() - 1);
    }
    if (removed_adjacent_polygon.size() <= 3) {
        return removed_adjacent_polygon;
    }

    int                           n             = removed_adjacent_polygon.size();
    Coordinate2df32               prev_point    = removed_adjacent_polygon[n - 1];
    Coordinate2df32               current_point = removed_adjacent_polygon[0];
    PointWithAreaF32*             prev_pwa      = nullptr;
    PointWithAreaF32*             first_pwa     = nullptr;
    std::vector<PointWithAreaF32> pwa_pool;
    pwa_pool.reserve(n);

    // building the queue
    for (int i = 0; i < n; ++i) {
        Coordinate2df32 next_point = removed_adjacent_polygon[(i + 1) % n];
        double          area       = calculate_triple_area(prev_point, current_point, next_point);
        pwa_pool.emplace_back(current_point, area);
        PointWithAreaF32* pwa = pwa_pool.data() + pwa_pool.size();
        pwa->set_prev(prev_pwa);
        if (prev_pwa != nullptr) {
            prev_pwa->set_next(pwa);
        }
        prev_pwa   = pwa;
        prev_point = current_point;
        if (i == n - 1) {
            pwa->set_next(first_pwa);
            first_pwa->set_prev(pwa);
        } else if (i == 0) {
            first_pwa = pwa;
        }
    }

    double max_area = 0;
    int    min_size = FISH_MAX(n / 100, 3);

    std::set<Coordinate2df32>      remove_points;
    int                            remain_size = n;
    int                            search_idx  = 0;
    std::vector<PointWithAreaF32*> pwa_ptrs;
    pwa_ptrs.reserve(pwa_pool.size());
    for (size_t i = 0; i < pwa_pool.size(); ++i) {
        pwa_ptrs.push_back(&pwa_pool[i]);
    }
    // this code is error!
    while (remain_size > min_size) {
        int min_area_idx = search_idx;
        for (int i = search_idx + 1; i < pwa_ptrs.size(); ++i) {
            if (pwa_ptrs[i]->get_area() > pwa_ptrs[min_area_idx]->get_area()) {
                min_area_idx = i;
            }
        }
        // means that the search idx only access once,we will access search idx + 1 at last time!
        //  swap the value of min_area_idx and search idx!
        PointWithAreaF32* pwa  = pwa_ptrs[min_area_idx];
        pwa_ptrs[min_area_idx] = pwa_ptrs[search_idx];
        pwa_ptrs[search_idx]   = pwa;

        auto&  _next_p = pwa->get_next()->p;
        auto&  _prev_p = pwa->get_prev()->p;
        float  dx      = _next_p.x - _prev_p.x;
        float  dy      = _next_p.y - _prev_p.y;
        double dist    = std::sqrt(dx * dx + dy * dy) + 1e-9;
        double altitue = pwa->get_area() * 2 / dist;
        if (altitue > altitude_threshold) {
            break;
        }
        if (pwa->get_area() < max_area) {
            pwa->set_area(max_area);
        } else {
            max_area = pwa->get_area();
        }
        remove_points.insert(pwa->get_point_ref());
        prev_pwa                   = pwa->get_prev();
        PointWithAreaF32* next_pwa = pwa->get_next();
        prev_pwa->set_next(next_pwa);
        next_pwa->set_prev(prev_pwa);

        // this is unsafe!
        prev_pwa->update_area();
        next_pwa->update_area();
        // adjust the priority!
        ++search_idx;
        --remain_size;
    }
    if (remove_points.size() == 0) {
        return removed_adjacent_polygon;
    }
    PolygonTypef32 ret_polygon;
    ret_polygon.reserve(removed_adjacent_polygon.size() - remove_points.size());
    for (size_t i = 0; i < removed_adjacent_polygon.size(); ++i) {
        if (remove_points.find(removed_adjacent_polygon[i]) != remove_points.end()) {
            ret_polygon.push_back(removed_adjacent_polygon[i]);
        }
    }
    return ret_polygon;
}

}   // namespace shape_simplier
}   // namespace segmentation
}   // namespace fish