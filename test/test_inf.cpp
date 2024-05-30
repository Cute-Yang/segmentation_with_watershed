#include <cmath>
#include <iostream>
#include <limits>


int main() {
    float s1 = std::numeric_limits<double>::infinity();

    bool s1_inf = std::isinf(s1);
    if (s1_inf) {
        std::cout << "the max background is an inf value" << std::endl;
    } else {
        std::cout << "the max background is not an inf value" << std::endl;
    }
    std::cout << s1 << std::endl;

    std::cout << s1 / 0.00015 << std::endl;
    return 0;
}