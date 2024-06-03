#include <iostream>
#include <vector>


struct Test {
    long a, b, c, d;
    Test() { std::cout << "Test" << std::endl; }
    ~Test() { std::cout << "~Test" << std::endl; }
    Test(const Test&) { std::cout << "Test copy" << std::endl; }
    Test(Test&&) noexcept { std::cout << "Test move" << std::endl; }
};

int main(int argc, const char* argv[]) {
    std::vector<Test> ve;
    ve.emplace_back();
    ve.emplace_back();
    ve.emplace_back();
    return 0;
}
