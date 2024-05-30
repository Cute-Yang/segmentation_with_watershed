#pragma once
#include <chrono>

namespace fish {
namespace utils {
class TickTok {
private:
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

public:
    void tick() { start = std::chrono::steady_clock::now(); }

    void tock() { end = std::chrono::steady_clock::now(); }

    double compute_elapsed_milli() {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};
}   // namespace utils
}   // namespace fish
