#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include <cmath>
#include <vector>

namespace fish {
namespace image_proc {
namespace guassian_blur {
using namespace fish::core::mat;
namespace internal {
constexpr float GUASSIAN_BLUR_SIGMA = 2.0f;
constexpr float GUASSIAN_LOW_ACC    = 0.002f;
constexpr float GUASSIAN_HIGH_ACC   = 0.0002f;

constexpr int GUASSIAN_K_RADIUS_LIMIT = 50;

inline int compute_k_radius(double sigma, double acc) {
    return static_cast<int>(std::ceil((sigma * std::sqrt(-2 * std::log(acc))))) + 1;
}

inline int clip_k_radius(int max_k_radius, int k_radius) {
    constexpr int k_radius_limit = 50;
    max_k_radius                 = FISH_MIN(max_k_radius, k_radius_limit);
    k_radius                     = FISH_MIN(k_radius, max_k_radius);
    return k_radius;
}

void compute_kernel_sum(double sigma, int k_radius, int max_k_radius, float* kernel);
FISH_EXPORTS inline std::vector<float> compute_kernel_sum(double sigma, int k_radius,
                                                          int max_k_radius) {
    std::vector<float> kernel(2 * k_radius);
    compute_kernel_sum(sigma, k_radius, max_k_radius, kernel.data());
    return kernel;
}

FISH_EXPORTS std::vector<float> compute_kernel_sum(float* kernel, int kernel_size);

// the kernel size is 2 * k_radius - 1
// compute the full guassian kernel,for better performence!
void compute_kernel(double sigma, int k_radius, int max_k_radius, float* kernel);
FISH_EXPORTS inline std::vector<float> compute_kernel(double sigma, int k_radius,
                                                      int max_k_radius) {
    std::vector<float> kernel(k_radius);
    compute_kernel(sigma, k_radius, max_k_radius, kernel.data());
    return kernel;
}



void       compute_downscale_kernel(int block_size, float* kernel);
inline int compute_downscale_kernel_size(int block_size) {
    return 3 * block_size;
}
FISH_EXPORTS inline std::vector<float> compute_downscale_kernel(int block_size) {
    int                downscale_kernel_size = compute_downscale_kernel_size(block_size);
    std::vector<float> downscale_kernel(downscale_kernel_size);
    compute_downscale_kernel(block_size, downscale_kernel.data());
    return downscale_kernel;
}


void       compute_upscale_kernel(int block_size, float* kernel);
inline int compute_upscale_kernel_size(int block_size) {
    return 4 * block_size;
}
FISH_EXPORTS inline std::vector<float> compute_upscale_kernel(int block_size) {
    int                upscale_kernel_size = compute_upscale_kernel_size(block_size);
    std::vector<float> upscaled_kernel(upscale_kernel_size);
    compute_upscale_kernel(block_size, upscaled_kernel.data());
    return upscaled_kernel;
}
}   // namespace internal


template<class T, typename = dtype_limit<T>>
Status::ErrorCode guassian_blur_2d(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                                   double sigma);

template<class T, typename = dtype_limit<T>>
Status::ErrorCode guassian_blur_2d(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                                   double sigma_x, double sigma_y);

}   // namespace guassian_blur



}   // namespace image_proc
}   // namespace fish