#pragma once
#include "core/base.h"
#include "core/mat.h"
#include "utils/logging.h"
#include <array>
#include <cstddef>
#include <functional>
#include <queue>
#include <vector>

namespace fish {
namespace segmentation {
namespace watershed {
enum class NeighborConnectiveType : uint8_t { Conn8 = 0, Conn4 = 1 };
using namespace fish::core::mat;
constexpr uint8_t IS_QUEUED  = 1;
constexpr uint8_t NOT_QUEUED = 0;
template<class T> struct PixelWithValue {
    int    x;
    int    y;
    T      value;
    size_t count;
    PixelWithValue(int x_, int y_, T value_, size_t count_)
        : x(x_)
        , y(y_)
        , value(value_)
        , count(count_) {}
    PixelWithValue() = delete;
    PixelWithValue(const PixelWithValue<T>& rhs)
        : x(rhs.x)
        , y(rhs.y)
        , value(rhs.value)
        , count(rhs.count) {}

    PixelWithValue(PixelWithValue<T>&& rhs)
        : x(rhs.x)
        , y(rhs.y)
        , value(rhs.value)
        , count(rhs.count) {}
    PixelWithValue<T>& operator=(const PixelWithValue<T>& rhs) {
        x     = rhs.x;
        y     = rhs.y;
        value = rhs.value;
        count = rhs.count;
        return *this;
    }

    PixelWithValue<T>& operator=(PixelWithValue<T>&& rhs) {
        x     = rhs.x;
        y     = rhs.y;
        value = rhs.value;
        count = rhs.count;
        return *this;
    }

    // 在相同的像素值下,前面添加的元素表示越高优先级(count越小表示越大)
    // the std::less requires the lhs and rhs is const!
    bool less(const PixelWithValue<T>& rhs) const {
        if (value == rhs.value) {
            //所以这里比较是相反的(在我们实际的程序中,不可能出现两个count值一样的数据)
            return count > rhs.count;
        }
        return value < rhs.value;
    }


    //后面添加的元素,表示具有较低优先级(count越大表示越小)
    bool greater(const PixelWithValue<T>& rhs) const {
        if (value == rhs.value) {
            return count < rhs.count;
        }
        return value > rhs.value;
    }

    bool equal(const PixelWithValue<T>& rhs) const {
        return (value == rhs.value && count == rhs.count && x == rhs.x && y == rhs.x);
    }
    bool operator<(const PixelWithValue<T>& rhs) const { return less(rhs); }
    bool operator>(const PixelWithValue<T>& rhs) const { return greater(rhs); }
    bool operator==(const PixelWithValue<T>& rhs) const { return equal(rhs); }
};
using FloatPixelWithValue       = PixelWithValue<float>;
using UCharPixelWithValue       = PixelWithValue<unsigned char>;
using UIntPixelWithValue        = PixelWithValue<unsigned int>;
using UShortPixelWithValue      = PixelWithValue<unsigned short>;
using float_pixel_with_value_t  = FloatPixelWithValue;
using uchar_pixel_with_value_t  = UCharPixelWithValue;
using uint_pixel_with_value_t   = UIntPixelWithValue;
using ushort_pixel_with_value_t = UShortPixelWithValue;


// if T==float,the sizeof pxiel is 4 + 4+8 = 16 bytes
// if restore x,y,will pad to 24 bytes
template<class T> struct BetterPixelWithValue {
    uint32_t index;
    T        value;
    size_t   count;
    BetterPixelWithValue(uint32_t index_, T value_, size_t count)
        : index(index_)
        , value(value_)
        , count(count) {}
    BetterPixelWithValue()                                   = delete;
    BetterPixelWithValue(const BetterPixelWithValue<T>& rhs) = default;
    BetterPixelWithValue(BetterPixelWithValue<T>&& rhs)      = default;
    bool less(const BetterPixelWithValue<T>& rhs) const {
        if (value == rhs.value) {
            // the count never equal!
            return count > rhs.count;
        }
        return value < rhs.value;
    }
    bool equal(const BetterPixelWithValue<T>& rhs) const {
        return (value == rhs.value && count == rhs.count && index == rhs.index);
    }
    bool greater(const BetterPixelWithValue<T>& rhs) const {
        if (value == rhs.value) {
            return count < rhs.count;
        }
        return value > rhs.value;
    }

    //一定保证左操作数 op 右操作数时 返回true
    bool operator<(const BetterPixelWithValue<T>& rhs) const { return less(rhs); }
    bool operator>(const BetterPixelWithValue<T>& rhs) const { return greater(rhs); }
    bool operator==(const BetterPixelWithValue<T>& rhs) const { return equal(rhs); }
};

using BetterFloatPixelWithValue        = BetterPixelWithValue<float>;
using BetterUCharPixelWithValue        = BetterPixelWithValue<unsigned char>;
using BetterUIntPixelWithValue         = BetterPixelWithValue<unsigned int>;
using BetterUShortPixelWithValue       = BetterPixelWithValue<unsigned short>;
using better_float_pixel_with_value_t  = BetterFloatPixelWithValue;
using better_uchar_pixel_with_value_t  = BetterUCharPixelWithValue;
using better_uint_pixel_with_value_t   = BetterUIntPixelWithValue;
using better_ushort_pixel_with_value_t = BetterUShortPixelWithValue;

//理论上只需要1/8内存
class SaveMemoryLogicalMask {
private:
    int                  height;
    int                  width;
    std::vector<uint8_t> mask;

public:
    SaveMemoryLogicalMask(int height_, int width_)
        : height(height_)
        , width(width_)
        // 向上取整,防止越界
        , mask((height_ * width_ + 7) / 8, 0) {}

    template<bool value> void set_value(int x, int y) {
        int block_idx = (y * width + x) / 8;
        int bit_idx   = y * width + x - block_idx * 8;
        //将对应的bit位设置成1,其他位置不关心,所以|运算
        if constexpr (value) {
            mask[block_idx] |= bit_flags[bit_idx];
        } else {
            //将对应的bit位射0,同时其他位置不关心
            mask[block_idx] &= (~bit_flags[bit_idx]);
        }
    }

    bool get_value(int x, int y) {
        int block_idx = (y * width + x) / 8;
        int bit_idx   = y * width + x - block_idx * 8;
        //获取对应的bit位,其余设置成0,如果为0,表示该bit位位0,否则为1
        return (mask[block_idx] & bit_flags[bit_idx]) != 0;
    }

public:
    // static constexpr std::array<uint8_t, 8> bit_flags = {
    //     0b1, 0b10, 0b100, 0b1000, 0b10000, 0b100000, 0b1000000, 0b1000000};
    static constexpr std::array<uint8_t, 8> bit_flags = {1, 2, 4, 8, 16, 32, 64, 128};
};


template<class T> class WatershedQueueWrapper {
public:
    using pixel_ref_t       = PixelWithValue<T>&;
    using const_pixel_ref_t = const PixelWithValue<T>&;
    using pixel_ptr_t       = PixelWithValue<T>*;
    using const_pixel_ptr_t = const PixelWithValue<T>*;

private:
    //这里实际也可以使用引用,达到内存复用的母的
    std::priority_queue<PixelWithValue<T>, std::vector<PixelWithValue<T>>,
                        std::less<PixelWithValue<T>>>
        pixel_queue;
    //用来指示(x,y)处的元素是否已经进入过队列
    // std::vector<uint8_t> queued;
    ImageMat<uint8_t> queued;
    //计数器,用来表示像素的优先级
    size_t counter;

public:
    WatershedQueueWrapper()
        : pixel_queue()   //这种构造方式可能会带来严重扩容问题
        , queued()
        , counter(0) {
        LOG_INFO("you must invoke the intialize function to initialize it!");
    }
    //指定底层容器的容量,避免扩容
    // delete copy
    WatershedQueueWrapper(const WatershedQueueWrapper<T>& rhs) = delete;
    WatershedQueueWrapper(WatershedQueueWrapper<T>&& rhs)      = delete;

    WatershedQueueWrapper<T>& operator=(const WatershedQueueWrapper<T>& rhs) = delete;
    WatershedQueueWrapper<T>& operator=(WatershedQueueWrapper<T>&& rhs)      = delete;

    template<class MarkerType>
    void initialize(const ImageMat<T>& image, const ImageMat<MarkerType>& marker, T min_threshold,
                    float estimate_enqueue_rate) {
        constexpr MarkerType marker_zero = static_cast<MarkerType>(0);
        // 可以获取到其 container
        int height = image.get_height();
        int width  = image.get_width();
        // 1/5作为预留空间
        std::vector<PixelWithValue<T>> queue_container;
        if (estimate_enqueue_rate <= 0.0f || estimate_enqueue_rate >= 1.0f) {
            LOG_WARN("get invalid estimate_enqueue_rate %f,we will set 0.2 as default!",
                     estimate_enqueue_rate);
            estimate_enqueue_rate = 0.2f;
        }
        queue_container.reserve(
            static_cast<size_t>(static_cast<float>(height * width) * estimate_enqueue_rate));
        std::priority_queue<PixelWithValue<T>> temp_queue(std::less<PixelWithValue<T>>(),
                                                          std::move(queue_container));
        LOG_INFO("allocate {} elements for our quque container!", height * width / 5);
        queued.resize(height, width, 1, true);
        // fill with zero!
        queued.set_zero();
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (image(y, x) <= min_threshold) {
                    queued(y, x) = IS_QUEUED;
                    continue;
                }
                if (marker(y, x) != marker_zero) {
                    queued(y, x) = IS_QUEUED;
                } else if (marker(y, x + 1) != marker_zero || marker(y, x - 1) != marker_zero ||
                           marker(y - 1, x) != marker_zero || marker(y + 1, x) != marker_zero) {
                    queued(y, x) = IS_QUEUED;
                    //如果邻居中存在非0点,入队列
                    temp_queue.emplace(x, y, image(y, x), counter);
                    ++counter;
                }
            }
        }
        // swap the queue and temp queue
        LOG_INFO("try to reseve some memory to avoid expand....");
        pixel_queue.swap(temp_queue);
    }

    void add(int x, int y, T value) {
        if (queued(y, x) == NOT_QUEUED) {
            pixel_queue.emplace(x, y, value, counter);
            ++counter;
            queued(y, x) = IS_QUEUED;
        }
    }

    bool can_add_to_queue(int x, int y) { return queued(y, x) == NOT_QUEUED; }

    const PixelWithValue<T>& get_top_pixel() { return pixel_queue.top(); }
    void                     remove_top_pixel() { pixel_queue.pop(); }

    // will copy....
    PixelWithValue<T> get_top_pixel_safe() {
        PixelWithValue<T> pixel = pixel_queue.top();
        pixel_queue.pop();
    }

    bool is_empty() { return pixel_queue.empty(); }
};


template<class T1, class T2, typename = image_dtype_limit<T1>, typename = image_dtype_limit<T2>>
Status::ErrorCode watershed_transform(const ImageMat<T1>& image, ImageMat<T2>& marker,
                                      T1 min_threshold, bool conn8);

// if false,means that all neightbors are background value or neighbors are different!
// if ture,means,the none background values are same
// return the fg_same?
}   // namespace watershed
}   // namespace segmentation
}   // namespace fish