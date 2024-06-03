#include "image_proc/flood_filler.h"
#include "common/fishdef.h"
#include "core/mat.h"
#include "utils/logging.h"

namespace fish {
namespace image_proc {
namespace flood_filler {
FloodFiller::FloodFiller(int capacity)
    : stack_capacity(capacity)
    , stack_size(0)
    , max_value(0) {
    coordinate_stack = new Coordinate2d[capacity];
}

FloodFiller::FloodFiller()
    : stack_capacity(initialize_stack_size)
    , stack_size(0)
    , max_value(0) {

    coordinate_stack = new Coordinate2d[initialize_stack_size];
}
FloodFiller::~FloodFiller() {
    if (coordinate_stack != nullptr) {
        free(coordinate_stack);
    }
    // maybe not need!
    stack_size     = 0;
    stack_capacity = 0;
}

void FloodFiller::push(int x, int y) {
    // the stack size is the element nums currently,if it equals capacity,we need allocate new
    // buffer!
    if (stack_size == stack_capacity) [[unlikely]] {
        //每次采取2倍的扩容策略
        int           new_stack_capacity = 2 * stack_capacity;
        Coordinate2d* new_coordinate_stack =
            reinterpret_cast<Coordinate2d*>(std::malloc(sizeof(Coordinate2d) * new_stack_capacity));
        if (new_coordinate_stack == nullptr) [[unlikely]] {
            LOG_ERROR("fail to allocate memory with %ld bytes,will exit....",
                      sizeof(Coordinate2d) * new_stack_capacity);
        }
        std::copy(coordinate_stack, coordinate_stack + stack_capacity, new_coordinate_stack);
        //释放旧的空间
        free(coordinate_stack);
        coordinate_stack = new_coordinate_stack;
        stack_capacity   = new_stack_capacity;
    }
    coordinate_stack[stack_size].x = x;
    coordinate_stack[stack_size].y = y;
    ++stack_size;
}


// only for single channel!
template<class T>
FISH_ALWAYS_INLINE void FloodFiller::fill_line(ImageMat<T>& image, int x1, int x2, int y,
                                               T fill_value) {
    for (int x = x1; x < x2; ++x) {
        image(y, x) = fill_value;
    }
}

//这里传进来得是每一个像素得种类,1,2,3,开始计数得style
template<class T> bool FloodFiller::fill(ImageMat<T>& image, int x, int y, T new_color) {
    int height = image.get_height();
    int width  = image.get_width();
    // the new color can not
    T color = image(y, x);
    // fill curren pixel
    if (color == new_color) {
        return false;
    }
    stack_size = 0;
    this->push(x, y);
    while (true) {
        if (stack_size == 0) {
            break;
        }
        Coordinate2d coor = pop();
        x                 = coor.x;
        y                 = coor.y;
        if (image(y, x) != color) {
            continue;
        }
        int x1 = x;
        int x2 = x;
        while (x1 >= 0 && image(y, x1) == color) {
            --x1;
        }
        // find  the start
        ++x1;

        // no need to substract for the end point!
        while (x2 < width && image(y, x2) == color) {
            ++x2;
        }
        fill_line(image, x1, x2, y, new_color);
        // bool in_scanline = false;
        // above this line
        // be sure the y-1 not out of range!
        if (y > 1) {
            // bad code
            //  for (int i = x1; i < x2; ++i) {
            //      if (!in_scanline && image(i, y - 1) == color) {
            //          push(i, y - 1);
            //          in_scanline = true;
            //          // 如果连续的都为color的点就不用push了,需要push中间与间隔的color的点...
            //          //就是寻找下一行中有几段color,然后把每段的第一个点添加到队列中,下一轮继续从左右扩散
            //      } else if (in_scanline && image(i, y - 1) != color) {
            //          in_scanline = false;
            //      }
            //  }
            //  unroll it!
            if (image(y - 1, x1) == color) {
                push(x1, y - 1);
            }
            for (int i = x1 + 1; i < x2; ++i) {
                // 如果前一个不等于color,这个等于color,那么就将其添加进去
                if (image(y - 1, i - 1) != color && image(y - 1, i) == color) {
                    push(i, y - 1);
                }
            }
        }
        // belone this line
        // in_scanline = false;
        if (y < height - 1) {
            // for (int i = x1; i <= x2; ++i) {
            //     if (!in_scanline && image(i, y + 1 == color)) {
            //         push(x, y + 1);
            //         in_scanline = false;
            //         //这种就是将line划分成不同段的点
            //     } else if (in_scanline && image(i, y + 1) != color) {
            //         in_scanline = false;
            //     }
            // }
            if (image(y + 1, x1) == color) {
                push(x1, y + 1);
            }
            for (int i = x1 + 1; i < x2; ++i) {
                // 如果前一个不等于color,这个等于color,那么就将其添加进去
                if (image(y + 1, i - 1) != color && image(y + 1, i) == color) {
                    push(i, y + 1);
                }
            }
        }
    }
    return true;
}

template bool FloodFiller::fill<float>(ImageMat<float>& image, int x, int y, float new_color);
template bool FloodFiller::fill<uint8_t>(ImageMat<uint8_t>& image, int x, int y, uint8_t new_color);
template bool FloodFiller::fill<uint16_t>(ImageMat<uint16_t>& image, int x, int y,
                                          uint16_t new_color);
template bool FloodFiller::fill<uint32_t>(ImageMat<uint32_t>& image, int x, int y,
                                          uint32_t new_color);

template<class T> bool FloodFiller::fill_eight(ImageMat<T>& image, int x, int y, T new_color) {
    int height = image.get_height();
    int width  = image.get_width();
    T   color  = image(y, x);
    if (color == new_color) {
        return false;
    }
    stack_size = 0;
    push(x, y);
    while (true) {
        if (stack_size == 0) {
            break;
        }
        // find the start and end of line
        Coordinate2d coor = pop();
        x                 = coor.x;
        y                 = coor.y;
        int x1            = x;
        int x2            = x;
        while (x1 >= 0 && image(y, x1) == color) {
            --x1;
        }
        ++x1;
        while (x2 < width && image(y, x2) == color) {
            ++x2;
        }
        fill_line(image, x1, x2, y, new_color);
        // the above line,这种貌似是比较稀疏的
        if (y > 0) {
            //考虑他们的临界点
            if (x1 > 0 && image(x1 - 1, y - 1) == color) {
                push(x1 - 1, y - 1);
            }
            if (x2 < width && image(y - 1, x2 + 1) == color) {
                push(x2, y - 1);
            }
            // add the next line
            if (image(y - 1, x1) == color) {
                push(x1, y - 1);
            }
            for (int i = x1 + 1; x < x2; ++i) {
                if (image(y - 1, i - 1) != color && image(y - 1, i) == color) {
                    push(i, y - 1);
                }
            }
        }
        // the below line!
        if (y < height - 1) {
            if (x1 > 0 && image(y - 1, x1 - 1) == color) {
                push(x1 - 1, y + 1);
            }
            if (x2 < width && image(y + 1, x2 + 1) == color) {
                push(x2, y + 1);
            }
            // add the next line
            if (image(y + 1, x1) == color) {
                push(x1, y + 1);
            }
            for (int i = x1 + 1; x < x2; ++i) {
                if (image(y + 1, i - 1) != color && image(y + 1, i) == color) {
                    push(i, y + 1);
                }
            }
        }

        // the blow line
    }
    return true;
}

template bool FloodFiller::fill_eight<float>(ImageMat<float>& image, int x, int y, float new_color);
template bool FloodFiller::fill_eight<uint8_t>(ImageMat<uint8_t>& image, int x, int y,
                                               uint8_t new_color);
template bool FloodFiller::fill_eight<uint16_t>(ImageMat<uint16_t>& image, int x, int y,
                                                uint16_t new_color);
template bool FloodFiller::fill_eight<uint32_t>(ImageMat<uint32_t>& image, int x, int y,
                                                uint32_t new_color);
}   // namespace flood_filler
}   // namespace image_proc
}   // namespace fish
