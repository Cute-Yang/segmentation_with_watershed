#include "image_proc/guassian_blur.h"
#include "utils/logging.h"
#include <chrono>

using namespace fish::image_proc::guassian_blur;
using namespace fish::image_proc::guassian_blur::internal;
int main() {
    auto     s1       = std::chrono::steady_clock::now();
    double   sigma    = 2.7;
    uint32_t k_radius = compute_k_radius(sigma, GUASSIAN_HIGH_ACC);
    LOG_INFO("the k_radius is {}", k_radius);
    auto kernel = compute_kernel(sigma, k_radius, 27);
    LOG_INFO("the size of kernel is {}", kernel.size());
    for (size_t i = 0; i < kernel.size(); ++i) {
        LOG_INFO("idx={} value={}", i, kernel[i]);
    }

    double r = 0.0;
    for (size_t i = 0; i < k_radius; ++i) {
        LOG_INFO("idx = {},value = {}", i, r);
        r += kernel[k_radius - 1 - i];
    }

    auto kernel2 = compute_kernel(sigma, k_radius, 27);
    for (size_t i = 0; i < kernel2.size(); ++i) {
        LOG_INFO("idx = {},value = {}", i, kernel2[i]);
    }
    return 0;
}