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
        // here need to do some optimize,avoid do much copy...
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
        T*     new_array_ptr = reinterpret_cast<T*>(malloc(new_capacity * sizeof(T)));
        std::copy(array_ptr, array_ptr + capacity, new_array_ptr);
        capacity = new_capacity;
        free(array_ptr);
        array_ptr       = new_array_ptr;
        array_ptr[tail] = value;
        ++tail;
    }
};
using IntQueue = CuteQueue<int>;


// while the increasement is reversed,we need do some change!
template<bool push_queue>
int dilate_and_compare(ImageMat<float>& image_marker, const ImageMat<float>& image_mask,
                       bool reverse, IntQueue& queue) {
    if (image_marker.empty() || image_mask.empty()) {
        LOG_ERROR("the given mat can not be empty,so we will not run...");
        return -1;
    }
    if (!image_marker.compare_shape(image_mask)) {
        LOG_ERROR("the image_marker and image_mask should have same shape....");
        return -1;
    }
    int width    = image_marker.get_width();
    int height   = image_marker.get_height();
    int channels = image_marker.get_channels();
    if (channels != 1) {
        LOG_WARN("the image_marker should be single channel image,but get channels {}", channels);
    }

    if (height == 1 && width == 1) {
        LOG_INFO("nothinig to do with shape(1,1)");
        return 0;
    }
    float& bad_value = image_marker(1538, 0);

    int x_start;
    int x_end;
    int y_start;
    int y_end;
    int increment;

    if (reverse) {
        LOG_INFO("invoke dilate with reverse order...");
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
    float previous_value = image_marker(y_start, x_start);   // y-1,x
    float p1_value;                                          // y-1,x-1
    float p2_value;                                          // y-1,x
    float p3_value;                                          // y-1,x + 1
    float current_value;                                     // y,x

    // height,1
    if (width == 1) {
        LOG_INFO("apply dialting with shape ({},1)", height);
        // in this case,no prev,not p1 no p3,only p2
        // the only neighbor is x,y-1
        for (int y = y_start + increment; y != y_end; y += increment) {
            p2_value       = image_marker(y - increment, 0);
            current_value  = image_marker(y, 0);
            previous_value = current_value;
            if (current_value < p2_value) {
                float mask_value                = image_mask(y, 0);
                float minimum_of_neigh_and_mask = FISH_MIN(p2_value, mask_value);
                if (minimum_of_neigh_and_mask > current_value) {
                    current_value      = minimum_of_neigh_and_mask;
                    image_marker(y, 0) = minimum_of_neigh_and_mask;
                    ++changes;
                }
            }
            if constexpr (push_queue) {
                bool add_to_queue = false;
                if (p2_value < current_value && p2_value < image_mask(y - 1, 0)) {
                    add_to_queue = true;
                }
                if (add_to_queue) {
                    queue.add(y);
                }
            }
        }
        return changes;
    }
    // the first row...
    // in this case,we only need to compare the prev value and currentvalue is ok!
    // no need to compare the first value,because it always equal to itself!

    // height = 0,here we must use not equal!
    for (int x = x_start + increment; x != x_end; x += increment) {
        // for the first line,p1,p2,p3 will set to current value,only need to compare the previous
        // and current is ok!
        previous_value = image_marker(y_start, x - increment);
        current_value  = image_marker(y_start, x);
        // so the max neigh max value is alwasy previous value
        if (current_value < previous_value) {
            float mask_value = image_mask(y_start, x);
            // find the minimum value between neigh and mask value
            float minimum_of_neigh_and_mask = FISH_MIN(previous_value, mask_value);
            // set the current value with the minimum value between
            if (minimum_of_neigh_and_mask > current_value) {
                current_value            = minimum_of_neigh_and_mask;
                image_marker(y_start, x) = current_value;
                ++changes;
            }
        }
        // whether add the neigh to the queue!
        if constexpr (push_queue) {
            bool add_to_queue = (previous_value < current_value &&
                                 previous_value < image_mask(y_start, x - increment));
            if (add_to_queue) {
                queue.add(y_start * width + x);
            }
        }
    }


    // height 1->heigh -1
    for (int y = y_start + increment; y != y_end; y += increment) {
        // the first column only need to initialize the p2 and p3 value,no need p1 and prev!
        // this case only have 2 neigh! (exclude p1,prev) because now x=0
        p2_value      = image_marker(y - increment, x_start);               // y-1,x
        current_value = image_marker(y, x_start);                           // y,x
        p3_value      = image_marker(y - increment, x_start + increment);   // y-1,x+1

        float neigh_max_value = FISH_MAX(p2_value, p3_value);
        if (current_value < neigh_max_value) {
            float mask_value                = image_mask(y, x_start);
            float minimum_of_neigh_and_mask = FISH_MIN(mask_value, neigh_max_value);
            if (minimum_of_neigh_and_mask > current_value) {
                current_value            = minimum_of_neigh_and_mask;
                image_marker(y, x_start) = current_value;
                ++changes;
            }
        }

        // now the previous value equal to current value,so the condition is always false!
        if constexpr (push_queue) {
            bool add_to_queue = false;
            if (p2_value < current_value && p2_value < image_mask(y - increment, x_start)) {
                add_to_queue = true;
            } else if (p3_value < current_value &&
                       p3_value < image_mask(y - increment, x_start + increment)) {
                add_to_queue = true;
            }
            if (add_to_queue) {
                queue.add(y * width + x_start);
            }
        }


        // process the first column value...
        for (int x = x_start + increment; x != x_end - increment; x += increment) {
            // 左上对角线值
            // the x can not be last column...
            // the 4 neigh ^_^
            previous_value = image_marker(y, x - increment);
            p1_value       = image_marker(y - increment, x - increment);
            p2_value       = image_marker(y - increment, x);
            p3_value       = image_marker(y - increment, x + increment);
            current_value  = image_marker(y, x);

            if (y == 78 && x == 2192) {
                LOG_INFO("cute....");
            }

            float neigh_max_value = FISH_MAX(p1_value, p2_value);
            neigh_max_value       = FISH_MAX(neigh_max_value, p3_value);
            neigh_max_value       = FISH_MAX(neigh_max_value, previous_value);

            if (current_value < neigh_max_value) {
                float mask_value                = image_mask(y, x);
                float minimum_of_neigh_and_mask = FISH_MIN(mask_value, neigh_max_value);
                if (minimum_of_neigh_and_mask > current_value) {
                    current_value      = minimum_of_neigh_and_mask;
                    image_marker(y, x) = current_value;
                    ++changes;
                }
            }

            // which one is changed ^_^
            if constexpr (push_queue) {
                bool add_to_queue = false;
                if (previous_value < current_value &&
                    previous_value < image_mask(y, x - increment)) {
                    add_to_queue = true;
                } else if (p1_value < current_value &&
                           p1_value < image_mask(y - increment, x - increment)) {
                    add_to_queue = true;
                } else if (p2_value < current_value && p2_value < image_mask(y - increment, x)) {
                    add_to_queue = true;
                } else if (p3_value < current_value &&
                           p3_value < image_mask(y - increment, x + increment)) {
                    add_to_queue = true;
                }
                if (add_to_queue) {
                    queue.add(y * width + x);
                }
            }
        }

        // the last column
        current_value   = image_marker(y, x_end - increment);
        previous_value  = image_marker(y, x_end - 2 * increment);
        p1_value        = image_marker(y - increment, x_end - 2 * increment);
        p2_value        = image_marker(y - increment, x_end - increment);
        p3_value        = current_value;
        neigh_max_value = FISH_MAX(p1_value, p2_value);
        neigh_max_value = FISH_MAX(neigh_max_value, previous_value);

        if (current_value < neigh_max_value) {
            float mask_value                = image_mask(y, x_end - increment);
            float minimum_of_neigh_and_mask = FISH_MIN(mask_value, neigh_max_value);
            if (minimum_of_neigh_and_mask > current_value) {
                current_value                      = minimum_of_neigh_and_mask;
                image_marker(y, x_end - increment) = current_value;
                ++changes;
            }
        }
        if constexpr (push_queue) {
            if (y == 1538) {
                LOG_INFO("ssssssssss");
            }
            bool add_to_queue = false;
            if (previous_value < current_value &&
                previous_value < image_mask(y, x_end - 2 * increment)) {
                add_to_queue = true;
            } else if (p1_value < current_value &&
                       p1_value < image_mask(y - increment, x_end - 2 * increment)) {
                add_to_queue = true;
            } else if (p2_value < current_value &&
                       p2_value < image_mask(y - increment, x_end - increment)) {
                add_to_queue = true;
            }
            if (add_to_queue) {
                queue.add(y * width + (x_end - increment));
            }
        }
    }
    return changes;
}

FISH_ALWAYS_INLINE void process_point(ImageMat<float>&       image_marker,
                                      const ImageMat<float>& image_mask, int x, int y, float value,
                                      IntQueue& queue) {
    // if the marker value is minimum,ue the median value as the result!
    int   width             = image_marker.get_width();
    float temp_marker_value = image_marker(y, x);
    if (temp_marker_value < value) {
        float temp_mask_value = image_mask(y, x);
        if (temp_marker_value < temp_mask_value) {
            float minimum_value = FISH_MIN(temp_mask_value, value);
            image_marker(y, x)  = minimum_value;
            queue.add(y * width + x);
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
        internal::dilate_and_compare<false>(image_marker, image_mask, true, empty_queue);
        n_changes =
            internal::dilate_and_compare<false>(image_marker, image_mask, false, empty_queue);
        change_rate = static_cast<double>(n_changes) / static_cast<double>(element_size);
    }
    // avoid allocate frequently!
    internal::IntQueue queue(element_size / 4);
    internal::dilate_and_compare<true>(image_marker, image_mask, true, queue);
    internal::process_queue(image_marker, image_mask, queue);
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
            if (y == 78 && x == 2192) {
                LOG_INFO("cute");
            }
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
