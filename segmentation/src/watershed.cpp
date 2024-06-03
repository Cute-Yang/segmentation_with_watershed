#include "segmentation/watershed.h"
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "utils/logging.h"

namespace fish {
namespace segmentation {
namespace watershed {
namespace internal {
template<class T, typename = image_dtype_limit<T>>
bool get_neighbor_label_with_4_conn(ImageMat<T>& marker, int x, int y, T& last_label) {
    int         height     = marker.get_height();
    int         width      = marker.get_width();
    constexpr T zero_value = 0;   // avoid type cast in runtime!
    // find t he 4 neightbors
    size_t count = 0;
    T      none_zero_values[4];
    if (x > 0) [[likely]] {
        // left
        if (marker(y, x - 1) != zero_value) {
            none_zero_values[count] = marker(y, x - 1);
            ++count;
        }
    }
    if (x < width - 1) [[likely]] {
        // right
        if (marker(y, x + 1) != zero_value) {
            none_zero_values[count] = marker(y, x + 1);
            ++count;
        }
    }
    if (y > 0) [[likely]] {
        // top
        if (marker(y - 1, x) != zero_value) {
            none_zero_values[count] = marker(y - 1, x);
            ++count;
        }
    }
    // bottom!
    if (y < height - 1) [[likely]] {
        // if want to support negative,here should be <= zero_value
        if (marker(y + 1, x) != zero_value) {
            none_zero_values[count] = marker(y + 1, x);
            ++count;
        }
    }
    last_label = none_zero_values[0];
    bool neightbors_fg_are_same;
    switch (count) {
    case 0:
        // means all value are zeros
        neightbors_fg_are_same = false;
        break;
    case 1:
        // means only one element is none zero!
        neightbors_fg_are_same = true;
        break;
    case 2:
        // means 2 element is none zero
        neightbors_fg_are_same = (none_zero_values[0] == none_zero_values[1]);
        break;
    case 3:
        neightbors_fg_are_same = (none_zero_values[0] == none_zero_values[1]) &&
                                 (none_zero_values[1] == none_zero_values[2]);
        break;
    case 4:
        neightbors_fg_are_same = (none_zero_values[0] == none_zero_values[1]) &&
                                 none_zero_values[1] == none_zero_values[2] &&
                                 (none_zero_values[2] == none_zero_values[3]);
        break;
    // but it is unreachable!
    default: neightbors_fg_are_same = false; break;
    }
    return neightbors_fg_are_same;
}

template<class T>
bool get_neighbor_label_with_8_conn(ImageMat<T>& marker, int x, int y, T& last_label) {
    int         height     = marker.get_height();
    int         width      = marker.get_width();
    constexpr T zero_value = 0;
    size_t      none_zero_values[8];
    std::fill(none_zero_values, none_zero_values + 8, 0);
    size_t count   = 0;
    int    x_start = std::max(x - 1, 0);
    int    x_end   = std::min(x + 2, width);
    int    y_start = std::max(y - 1, 0);
    int    y_end   = std::min(y + 2, height);
    // we can flatten it
    for (int yy = y_start; yy < y_end; ++yy) {
        for (int xx = x_start; xx < x_end; ++xx) {
            if (xx == x && yy == y) [[unlikely]] {
                continue;
            }
            if (marker(yy, xx) != zero_value) {
                none_zero_values[count] = marker(yy, xx);
            }
        }
    }
    last_label = none_zero_values[0];
    // 直接计算方差,为0表示一定是相同的
    size_t mean_value =
        (none_zero_values[0] + none_zero_values[1] + none_zero_values[2] + none_zero_values[3] +
         none_zero_values[4] + none_zero_values[5] + none_zero_values[6] + none_zero_values[7]) /
        8;
    size_t variance = 0;
    for (size_t i = 0; i < count; ++i) {
        variance += (none_zero_values[i] - mean_value) * (none_zero_values[i] - mean_value);
    }
    return variance == 0UL;
}

// the template type of queue equals to image!
template<class T>
void add_neighbors_with_4_conn(WatershedQueueWrapper<T>& queue, int x, int y,
                               const ImageMat<T>& image) {
    int height = image.get_height();
    int width  = image.get_width();
    if (y > 0) [[likely]] {
        queue.add(x, y - 1, image(y - 1, x));
    }
    if (x > 0) [[likely]] {
        queue.add(x - 1, y, image(y, x - 1));
    }

    if (x < width - 1) [[likely]] {
        queue.add(x + 1, y, image(y, x + 1));
    }
    if (y < height - 1) [[likely]] {
        queue.add(x, y + 1, image(y + 1, x));
    }
}

//可以展开的
template<class T>
void add_neighbors_with_8_conn(WatershedQueueWrapper<T>& queue, int x, int y,
                               const ImageMat<T>& image) {
    int height  = image.get_height();
    int width   = image.get_width();
    int x_start = FISH_MAX(x - 1, 0);
    int x_end   = FISH_MIN(x + 2, width);
    int y_start = FISH_MAX(y - 1, 0);
    int y_end   = FISH_MIN(y + 2, height);
    for (int yy = y_start; yy < y_end; ++yy) {
        for (int xx = x_start; xx < x_end; ++xx) {
            if (xx == x && yy == y) [[unlikely]] {
                continue;
            }
            queue.add(xx, yy, image(yy, xx));
        }
    }
}
}   // namespace internal
template<class T1, class T2, NeighborConnectiveType conn>
Status::ErrorCode watershed_transform_impl(const ImageMat<T1>& image, ImageMat<T2>& marker,
                                           T1 min_threshold) {
    int height   = image.get_height();
    int width    = image.get_width();
    int channels = image.get_channels();
    if (channels != 1) {
        LOG_ERROR("channels {} is invalid ...", channels);
        return Status::ErrorCode::InvalidMatShape;
    }

    if (!marker.shape_equal(height, width, 1)) {
        marker.resize(height, width, true);
    }

    WatershedQueueWrapper<T1> wraped_queue;
    wraped_queue.initialize(image, marker, min_threshold, 0.4f);
    while (!wraped_queue.is_empty()) {
        const PixelWithValue<T1>& pixel_info = wraped_queue.get_top_pixel();
        int                       x          = pixel_info.x;
        int                       y          = pixel_info.y;
        T2                        last_label;
        // false,表示邻居存在两个以上不同前景,就不需要淹没
        bool neightbors_fg_are_same;
        if constexpr (conn == NeighborConnectiveType::Conn4) {
            // do 8 neight conn
            neightbors_fg_are_same =
                internal::get_neighbor_label_with_4_conn<T2>(marker, x, y, last_label);
        } else {
            neightbors_fg_are_same =
                internal::get_neighbor_label_with_8_conn<T2>(marker, x, y, last_label);
        }
        // now we need to remove the pixel!
        wraped_queue.remove_top_pixel();
        //如果neighbors有两种以上前景(或者无前景),修建大坝
        if (!neightbors_fg_are_same) {
            continue;
        }
        // 如果neighbors的前景相同,淹没当前像素
        marker(y, x) = last_label;
        // 广度优先扩散,添加其邻居节点
        if constexpr (conn == NeighborConnectiveType::Conn4) {
            internal::add_neighbors_with_4_conn<T1>(wraped_queue, x, y, image);
        } else {
            // use loop imp,can unroll it
            internal::add_neighbors_with_8_conn<T1>(wraped_queue, x, y, image);
        }
    }
    return Status::ErrorCode::Ok;
}

template<class T1, class T2, typename, typename>
Status::ErrorCode watershed_transform(const ImageMat<T1>& image, ImageMat<T2>& marker,
                                      T1 min_threshold, bool conn8) {
    Status::ErrorCode status;
    if (conn8) {
        status = watershed_transform_impl<T1, T2, NeighborConnectiveType::Conn8>(
            image, marker, min_threshold);
    } else {
        status = watershed_transform_impl<T1, T2, NeighborConnectiveType::Conn4>(
            image, marker, min_threshold);
    }
    return status;
}

template Status::ErrorCode watershed_transform<float, uint8_t>(const ImageMat<float>& image,
                                                               ImageMat<uint8_t>&     marker,
                                                               float min_threshold, bool conn8);

template Status::ErrorCode watershed_transform<float, uint16_t>(const ImageMat<float>& image,
                                                                ImageMat<uint16_t>&    marker,
                                                                float min_threshold, bool conn8);

template Status::ErrorCode watershed_transform<float, uint32_t>(const ImageMat<float>& image,
                                                                ImageMat<uint32_t>&    marker,
                                                                float min_threshold, bool conn8);

template Status::ErrorCode watershed_transform<float, float>(const ImageMat<float>& image,
                                                             ImageMat<float>&       marker,
                                                             float min_threshold, bool conn8);

}   // namespace watershed
}   // namespace segmentation
}   // namespace fish