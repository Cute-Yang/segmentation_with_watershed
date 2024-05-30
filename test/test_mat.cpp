#include "core/mat.h"
#include "utils/logging.h"
#include <cstdint>
#include <iostream>
using namespace fish::core::mat;
struct S {
    static void static_func() { std::cout << "this is the static method of S" << std::endl; }
};

template<class T> struct Cao {
    static void gg() { T::static_func(); }
};


template<class T> void foo() {
    LOG_INFO("hahahahhhhhhhh");
    ImageMat<T> s1(32, 32, 1);
    s1.template compare_shape<T>(s1);
}


int main() {
    ImageMat<float> m1(32, 32, 1);
    m1(0, 0, 0) = 1417.32f;
    m1(0, 3, 0) = 47.95f;
    m1(0, 5, 0) = 47.2f;
    m1.compare_shape<float>(m1);
    std::cout << m1 << std::endl;
    ImageMat<uint8_t> m2(32, 32, 1);
    convert_mat(m1, m2);

    std::cout << m2 << std::endl;
    return 0;
}
