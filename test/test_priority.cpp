#include <iostream>
#include <queue>


int main() {
    std::priority_queue<int*> p1;
    int*                      s = new int(32);
    p1.push(s);
    p1.push(new int(3));
    p1.push(new int(9));
    p1.push(new int(4));
    int size = p1.size();
    *s       = 128;
    for (int i = 0; i < size; ++i) {
        int* vptr = p1.top();
        std::cout << "addr=" << vptr << " value=" << *vptr << std::endl;
        p1.pop();
    }
    return 0;
}