#pragma once
#include "core/mat.h"


namespace fish {
namespace image_proc {
namespace flood_filler {
using namespace fish::core::mat;
class FloodFiller {
public:
    static constexpr size_t initialize_stack_size = 500;

private:
    size_t        stack_size;
    size_t        stack_capacity;
    Coordinate2d* coordinate_stack;
    int           max_value;

    void push(int x, int y);
    // int  pop_x();
    // int  pop_x_safe();
    // int  pop_y();
    // int  pop_y_safe();

    template<class T> void fill_line(ImageMat<T>& input_mat, int x1, int x2, int y, T fill_value);


    const Coordinate2d& pop() noexcept { return coordinate_stack[--stack_size]; }

public:
    FloodFiller(int capacity);
    FloodFiller();
    FloodFiller(const FloodFiller& rhs)            = delete;
    FloodFiller(FloodFiller&& rhs)                 = delete;
    FloodFiller& operator=(const FloodFiller& rhs) = delete;
    FloodFiller& operator=(FloodFiller&& rhs)      = delete;

    ~FloodFiller();

    template<class T> bool fill(ImageMat<T>& input_mat, int x, int y, T new_color);
    template<class T> bool fill_eight(ImageMat<T>& input_mat, int x, int y, T new_color);
};
}   // namespace flood_filler
}   // namespace image_proc
}   // namespace fish