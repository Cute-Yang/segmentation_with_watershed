#include "s1.h"
#include <chrono>
#include <iostream>
#include <thread>

//由此可以得出一个结论,我们的动态库在被不同的进程使用的时候，其全局变量和静态区数据一定会被拷贝一份
static int s1 = 0;

void foo() {
    for (int i = 0; i < 10; ++i) {
        std::cout << "the value of s1 is " << s1 << std::endl;
        ++s1;
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}
