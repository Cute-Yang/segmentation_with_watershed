#include "segmentation/morphological_transform.h"
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "core/mat_ops.h"
#include "utils/logging.h"
#include <cstdlib>

namespace fish {
namespace segmentation {
namespace morphological {
using namespace fish::core::mat_ops;
namespace internal {
template<class T> class CuteQueue {
private:
    T* array_ptr;
    T  head;
    T  tail;
    // size_t size;
    size_t capacity;

public:
    static constexpr size_t max_expansion = 1024 * 10;
    CuteQueue(size_t capacity)
        : head(0)
        , tail(0)
        // , size(0)
        , capacity(capacity) {
        size_t allocate_bytes = sizeof(T) * capacity;
        array_ptr             = reinterpret_cast<T*>(malloc(allocate_bytes));
    }

    CuteQueue()
        : head(0)
        , tail(0)
        , array_ptr(nullptr)
        // , size(0)
        , capacity(0) {}
    bool is_empty() const { return head == tail; }

    CuteQueue(const CuteQueue& rhs)
        : head(rhs.head)
        , tail(rhs.tail)
        , capacity(rhs.capacity) {
        size_t allocate_bytes = sizeof(T) * capacity;
        array_ptr             = reinterpret_cast<T*>(malloc(allocate_bytes));
        if (array_ptr == nullptr) {
            LOG_ERROR("fail to allocate memory with bytes {}", allocate_bytes);
            capacity = 0;
            return;
        } else {
            std::copy(rhs.array_ptr, rhs.array_ptr + capacity, array_ptr);
        }
    }

    CuteQueue(CuteQueue&& rhs)
        : head(rhs.head)
        , tail(rhs.tail)
        , capacity(rhs.capacity) {
        array_ptr     = rhs.array_ptr;
        rhs.array_ptr = nullptr;
        rhs.head      = 0;
        rhs.tail      = 0;
        rhs.capacity  = 0;
    }

    ~CuteQueue() {
        if (array_ptr != nullptr) {
            free(array_ptr);
            array_ptr = nullptr;
            head      = 0;
            tail      = 0;
            capacity  = 0;
        }
    }


    int remove() {
        ++head;
        return array_ptr[head - 1];
    }

    void add(int value) {
        if (tail < capacity) {
            array_ptr[tail] = value;
            ++tail;
            return;
        }

        // consider shift it!
        if (head != 0) {
            // shit to the begin of the array!
            if (tail > head) {
                std::copy(array_ptr + head, array_ptr + tail, array_ptr);
                tail -= head;
                head         = 0;
                array_ptr[0] = value;
                ++tail;
                return;
            }
        }

        size_t new_capacity  = std::max(capacity * 2, max_expansion);
        T*     new_array_ptr = new T[new_capacity];
        std::copy(array_ptr, array_ptr + capacity, new_array_ptr);
        capacity = new_capacity;
        delete[] array_ptr;
        array_ptr       = new_array_ptr;
        array_ptr[tail] = value;
        ++tail;
    }
};
using IntQueue = CuteQueue<int>;


template<bool push_queue>
int dilate_and_compare(ImageMat<float>& image_marker, const ImageMat<float>& image_mask,
                       bool reverse, IntQueue& queue) {
    int width  = image_marker.get_width();
    int height = image_marker.get_height();
    int x_start;
    int x_end;
    int y_start;
    int y_end;
    int increment;
    if (reverse) {
        increment = -1;
        x_start   = width - 1;
        x_end     = -1;
        y_start   = height - 1;
        y_end     = -1;
    } else {
        increment = 1;
        x_start   = 0;
        x_end     = width;
        y_start   = 0;
        y_end     = height;
    }
    int changes = 0;

    // handle the first line
    float previous_value = image_marker(y_start, x_start);
    float p1_value;
    float p2_value;
    float p3_value;
    float current_value;
    // the first row...
    for (int x = x_start; x < x_end - increment; x += increment) {
        // for the first line,p1,p2,p3 will set to current value,only need to compare the previous
        // and current is ok!
        float current_value = image_mask(y_start, x + increment, 0);
        if (previous_value > current_value) {
            float value_mask = image_mask(y_start, x + increment);
            float new_value  = (previous_value > value_mask) ? value_mask : previous_value;
            // set the current value with the min
            if (new_value > current_value) {
                current_value                        = new_value;
                image_marker(y_start, x + increment) = current_value;
                ++changes;
            }
        }
        previous_value = current_value;
        if (previous_value < current_value && previous_value < image_mask(y_start, x - increment)) {
            queue.add(y_start * width + x);
        }
    }

    for (int y = y_start + increment; y < y_end; y += increment) {
        // initialize the value of p1 and p2
        previous_value = image_marker(y, x_start);
        p2_value       = image_marker(y - increment, x_start);
        p1_value       = p2_value;
        int x          = x_start;
        for (; x < x_end - increment; x += increment) {
            //左上对角线值
            p3_value                 = image_marker(y - increment, x + increment);
            current_value            = image_marker(y, x);
            float neighbor_max_value = (p1_value > p2_value) ? p1_value : p2_value;
            neighbor_max_value = (p3_value > neighbor_max_value) ? p3_value : neighbor_max_value;
            neighbor_max_value =
                (previous_value > neighbor_max_value) ? previous_value : neighbor_max_value;
            if (current_value < neighbor_max_value) {
                float mask_value = image_mask(y, x);
                float new_value =
                    (neighbor_max_value > mask_value) ? mask_value : neighbor_max_value;
                //表达的意思就是,如果当前值比邻居值中的最大值和mask中值都小时,将这几个里面中的较小者赋给当前值
                if (new_value > current_value) {
                    current_value      = new_value;
                    image_marker(y, x) = current_value;
                    ++changes;
                }
            }
            bool add_to_queue = false;
            //如果当前值比邻居值大,且邻居小于对应位置的mask值,就将点添加到队列
            if (previous_value < current_value && previous_value < image_mask(y, x - increment)) {
                add_to_queue = true;
            } else {
                if (p1_value < current_value &&
                    p1_value < image_mask(y - increment, x - increment)) {
                    add_to_queue = true;
                } else if (p2_value < current_value && p2_value < image_mask(y - increment, x)) {
                    add_to_queue = true;
                } else if (p3_value < current_value &&
                           p3_value < image_mask(y - increment, x + increment)) {
                    add_to_queue = true;
                }
            }
            if (add_to_queue) {
                queue.add(y * width + x);
            }
            previous_value = current_value;
            p1_value       = p2_value;
            p2_value       = p3_value;
        }
        // the last column
        current_value            = image_marker(y, x_end - increment);
        p3_value                 = current_value;
        float neighbor_max_value = (p1_value > p2_value) ? p1_value : p2_value;
        neighbor_max_value       = (p3_value > neighbor_max_value) ? p3_value : neighbor_max_value;
        neighbor_max_value =
            (previous_value > neighbor_max_value) ? previous_value : neighbor_max_value;
        if (current_value < neighbor_max_value) {
            float mask_value = image_mask(y, x_end - increment);
            float new_value  = (neighbor_max_value > mask_value) ? mask_value : neighbor_max_value;
            if (new_value > current_value) {
                current_value                      = new_value;
                image_marker(y, x_end - increment) = current_value;
                ++changes;
            }
        }
        if constexpr (push_queue) {
            bool add_to_queue = false;
            // 如果当前值比邻居值大,且邻居小于对应位置的mask值,就将点添加到队列
            if (previous_value < current_value && previous_value < image_mask(y, x - increment)) {
                add_to_queue = true;
            } else {
                if (p1_value < current_value &&
                    p1_value < image_mask(y - increment, x - increment)) {
                    add_to_queue = true;
                } else if (p2_value < current_value && p2_value < image_mask(y - increment, x)) {
                    add_to_queue = true;
                } else if (p3_value < current_value &&
                           p3_value < image_mask(y - increment, x + increment)) {
                    add_to_queue = true;
                }
            }
            if (add_to_queue) {
                queue.add(y * width + x);
            }
        }
    }
    return changes;
}

FISH_ALWAYS_INLINE void process_point(ImageMat<float>&       image_marker,
                                      const ImageMat<float>& image_mask, int x, int y, float value,
                                      IntQueue& queue) {
    // if the marker value is minimum,ue the median value as the result!
    float temp_marker_value = image_marker(y, x);
    if (temp_marker_value < value) {
        float temp_mask_value = image_mask(y, x);
        if (temp_marker_value < temp_mask_value) {
            float new_value    = temp_mask_value < value ? temp_mask_value : value;
            image_marker(y, x) = new_value;
            queue.add(y * image_marker.get_width() + x);
        }
    }
}


// need to optimize this code!
void process_queue(ImageMat<float>& image_marker, const ImageMat<float>& image_mask,
                   IntQueue& queue) {
    int height = image_marker.get_height();
    int width  = image_marker.get_width();
    int x1     = 0;
    int x2     = width;
    int y1     = 0;
    int y2     = height;

    // int counter = 0;
    while (!queue.is_empty()) {
        // ++counter;
        int   flatten_index = queue.remove();
        int   x             = flatten_index % width;
        int   y             = flatten_index / width;
        float value         = image_marker(y, x, 0);
        // eight neighboor
        // bad code
        if (x > x1) [[likely]] {
            process_point(image_marker, image_mask, x - 1, y, value, queue);
            if (y > y1) [[likely]] {
                process_point(image_marker, image_mask, x - 1, y - 1, value, queue);
            }
            if (y < y2 - 1) [[likely]] {
                process_point(image_marker, image_mask, x - 1, y + 1, value, queue);
            }
        }
        if (x < x2 - 1) [[likely]] {
            process_point(image_marker, image_mask, x + 1, y, value, queue);
            if (y > y1) [[likely]] {
                process_point(image_marker, image_mask, x + 1, y - 1, value, queue);
            }
            if (y < y2 - 1) [[likely]] {
                process_point(image_marker, image_mask, x + 1, y + 1, value, queue);
            }
        }

        if (y > y1) [[likely]] {
            process_point(image_marker, image_mask, x, y - 1, value, queue);
        }
        if (y < y2 - 1) [[likely]] {
            process_point(image_marker, image_mask, x, y + 1, value, queue);
        }
    }
}

}   // namespace internal
Status::ErrorCode morphological_transform(ImageMat<float>&       image_marker,
                                          const ImageMat<float>& image_mask) {
    int height   = image_marker.get_height();
    int width    = image_marker.get_width();
    int channels = image_marker.get_channels();
    if (channels != 1) {
        return Status::ErrorCode::InvalidMatShape;
    }
    if (!image_marker.compare_shape(image_mask)) {
        return Status::ErrorCode::MatShapeMismatch;
    }
    int                element_size = height * width;
    internal::IntQueue empty_queue;
    int                n_changes =
        internal::dilate_and_compare<false>(image_marker, image_mask, false, empty_queue);
    double change_rate = static_cast<double>(n_changes) / static_cast<double>(element_size);
    constexpr double min_change_rate = 0.1;
    while (change_rate > min_change_rate) {
        // repeat
        dilate_and_compare<false>(image_marker, image_mask, true, empty_queue);
        n_changes   = dilate_and_compare<false>(image_marker, image_mask, false, empty_queue);
        change_rate = static_cast<double>(n_changes) / static_cast<double>(element_size);
    }
    // avoid allocate frequently!
    internal::IntQueue queue(element_size / 4);
    dilate_and_compare<true>(image_marker, image_mask, true, queue);
    process_queue(image_marker, image_mask, queue);
    return Status::Ok;
}

/**
 * Replace all potential local maxima - as determined by effectively comparing
 * the image with itself after
 * applying a 3x3 maximum filter - with the lowest possible value via
 * {@code setf(x, y, Float.NEGATIVE_INFINITY)}.
 * <p>
 * These can then be filled in by morphological reconstruction on the way to
 * finding 'true' maxima.
 **/
template<class T, typename = image_dtype_limit<T>>
Status::ErrorCode get_maximal_labels(const ImageMat<T>& image, ImageMat<T>& marked_maximum_image,
                                     T threshold, int x1, int y1, int x2, int y2) {
    constexpr T type_min_value = std::numeric_limits<T>::lowest();
    int         height         = image.get_height();
    int         width          = image.get_width();
    int         channels       = image.get_channels();
    if (channels != 1) {
        return Status::ErrorCode::InvalidMatChannle;
    }
    if (!marked_maximum_image.shape_equal(height, width, 1)) {
        marked_maximum_image.resize(height, width, 1, true);
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            marked_maximum_image(y, x) = image(y, x);
        }
    }

    // find the maximum data with 3x3 window!
    for (int y = y1 + 1; y < y2 - 1; ++y) {
        T value      = image(y, x1);
        T next_value = image(y, x1 + 1);
        for (int x = x1 + 1; x < x2 - 1; ++x) {
            T last_value = value;
            value        = next_value;
            next_value   = image(y, x + 1);
            // if value less than threhsold or less than prev and next value!
            if (value < threshold || value < last_value || value < next_value) {
                continue;
            }
            // if the value >= the 6 neightbor...
            if (value >= image(y - 1, x - 1) && value >= image(y - 1, x) &&
                value >= image(y - 1, x + 1) && value >= image(y + 1, x - 1) &&
                value >= image(y + 1, x) && value >= image(y + 1, x + 1)) {
                marked_maximum_image(y, x) = type_min_value;
            }
        }
    }
    return Status::ErrorCode::Ok;
}


template<class T, typename = image_dtype_limit<T>>
ImageMat<T> get_maximal_labels(const ImageMat<T>& image, T threshold, int x1, int y1, int x2,
                               int y2) {
    ImageMat<T> marked_maximum_image;
    get_maximal_labels(image, marked_maximum_image, threshold, x1, y1, x2, y2);
    return marked_maximum_image;
}


void find_regional_maxima(const ImageMat<float>& image, ImageMat<float>& marked_maximum_image,
                          float threshold) {
    int width  = image.get_width();
    int height = image.get_height();
    int x1 = 0, x2 = width, y1 = 0, y2 = height;
    // resue the memory!
    // find the mamimu and mark it with float min with 3x3 window...
    get_maximal_labels(image, marked_maximum_image, threshold, x1, y1, x2, y2);
    morphological_transform(marked_maximum_image, image);
    copy_image_mat(image, marked_maximum_image, ValueOpKind::DIFFERENCE);
}

ImageMat<float> find_regional_maxima(const ImageMat<float>& image, float threshold) {
    ImageMat<float> marked_maximum_image;
    find_regional_maxima(image, marked_maximum_image, threshold);
    return marked_maximum_image;
}

void find_regional_maxima_and_binarize(const ImageMat<float>& image,
                                       ImageMat<float>&       marked_maximum_image,
                                       ImageMat<uint8_t>& marked_mask, float threshold) {
    int width  = image.get_width();
    int height = image.get_height();
    // the valid rect! x1y1 x2y2
    int x1 = 0, x2 = width, y1 = 0, y2 = height;
    get_maximal_labels<float>(image, marked_maximum_image, threshold, x1, x2, y1, y2);
    morphological_transform(marked_maximum_image, image);
    compare_mat<float, MatCompareOpType::GREATER>(image, marked_maximum_image, marked_mask);
    ImageMat<uint8_t> mask = mat_ops::compare_mat<float, mat_ops::MatCompareOpType::GREATER>(
        image, marked_maximum_image);
}

void find_regional_maxima_and_binarize(const ImageMat<float>& image, ImageMat<uint8_t>& marked_mask,
                                       float threshold) {
    int width  = image.get_width();
    int height = image.get_height();

    // the valid rect! x1y1 x2y2
    int             x1 = 0, x2 = width, y1 = 0, y2 = height;
    ImageMat<float> marked_maximum_image(height, width, 1, MatMemLayout::LayoutRight);
    get_maximal_labels<float>(image, marked_maximum_image, threshold, x1, x2, y1, y2);
    morphological_transform(marked_maximum_image, image);
    compare_mat<float, MatCompareOpType::GREATER>(image, marked_maximum_image, marked_mask);
    ImageMat<uint8_t> mask = mat_ops::compare_mat<float, mat_ops::MatCompareOpType::GREATER>(
        image, marked_maximum_image);
}


ImageMat<uint8_t> find_regional_maxima_and_binarize(const ImageMat<float>& image, float threshold) {
    int width  = image.get_width();
    int height = image.get_height();
    // the valid rect! x1y1 x2y2
    int             x1 = 0, x2 = width, y1 = 0, y2 = height;
    ImageMat<float> marked_maximum_image =
        get_maximal_labels<float>(image, threshold, x1, x2, y1, y2);
    morphological_transform(marked_maximum_image, image);
    ImageMat<uint8_t> mask =
        compare_mat<float, mat_ops::MatCompareOpType::GREATER>(image, marked_maximum_image);
    return mask;
}

}   // namespace morphological
}   // namespace segmentation
}   // namespace fish
