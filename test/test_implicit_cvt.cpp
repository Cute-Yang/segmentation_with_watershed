#include <iostream>

void foo(unsigned char x) {
    std::cout << "invoke func int foo" << std::endl;
}


void foo(double x) {
    std::cout << "invoke func double foo" << std::endl;
}

int main() {
    foo(13);
    return 0;
}