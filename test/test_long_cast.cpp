#include <iostream>

int main() {
    long value = 32;
    int  x     = 48;
    int  y     = 64;
    int  width = 320;

    int64_t encode_value = (value << 32) | (y * width + x);

    int coor_value = static_cast<int>(encode_value);

    int decode_x = coor_value % width;
    int decode_y = coor_value / width;

    std::cout << "the decode x is " << decode_x << " the decode_y is " << decode_y << std::endl;
    return 0;
}