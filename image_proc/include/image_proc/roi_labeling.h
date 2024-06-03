#pragma once
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "image_proc/polygon.h"
#include <vector>

namespace fish {
namespace image_proc {
namespace roi_labeling {
using namespace fish::core::mat;
using namespace fish::image_proc::polygon;

enum class FloodNeighConnType { Conn4 = 0, Conn8 = 1 };
struct PolygonWithMask {
    PolygonType       poly;
    ImageMat<uint8_t> mask;
    int               x0;
    int               y0;

    PolygonWithMask() = default;
    PolygonWithMask(const PolygonType& poly_, const ImageMat<uint8_t>& mask_, int x0_, int y0_)
        : poly(poly_)
        , mask(mask_)
        , x0(x0_)
        , y0(y0_) {}
    PolygonWithMask(PolygonType&& poly_, ImageMat<uint8_t>&& mask_, int x0_, int y0_)
        : poly(std::move(poly_))
        , mask(std::move(mask_))
        , x0(x0_)
        , y0(y0_) {}

    PolygonWithMask(const PolygonWithMask& rhs)
        : poly(rhs.poly)
        , mask(rhs.mask)
        , x0(rhs.x0)
        , y0(rhs.y0) {}

    PolygonWithMask(PolygonWithMask&& rhs)
        : poly(std::move(rhs.poly))
        , mask(std::move(rhs.mask))
        , x0(rhs.x0)
        , y0(rhs.y0) {}
};

struct PolyMask {
    int               x1;   // the x coor of left upper
    int               y1;   // the y coor of right bottom!
    ImageMat<uint8_t> mask;
    PolyMask()
        : x1(0)
        , y1(0)
        , mask() {}
    PolyMask(int x1_, int y1_, const ImageMat<uint8_t>& mask_)
        : x1(x1_)
        , y1(y1_)
        , mask(mask_) {}

    PolyMask(int x1_, int y1_, ImageMat<uint8_t>&& mask_)
        : x1(x1_)
        , y1(y1_)
        , mask(std::move(mask_)) {}

    PolyMask(const PolyMask& rhs)
        : x1(rhs.x1)
        , y1(rhs.y1)
        , mask(rhs.mask) {}

    PolyMask(PolyMask&& rhs) noexcept
        : x1(rhs.x1)
        , y1(rhs.y1)
        , mask(std::move(rhs.mask)) {
        rhs.x1 = 0;
        rhs.y1 = 0;
    }
};

// the impl of label image!
Status::ErrorCode compute_image_label(const ImageMat<float>& image, ImageMat<uint16_t>& label_image,
                                      float threshold, bool conn_8);

Status::ErrorCode compute_image_label(const ImageMat<float>& image, ImageMat<uint32_t>& label_image,
                                      float threshold, bool conn_8);

Status::ErrorCode compute_image_label(const ImageMat<uint8_t>& image,
                                      ImageMat<uint16_t>& label_image, float threshold,
                                      bool conn_8);

Status::ErrorCode compute_image_label(const ImageMat<uint8_t>& image,
                                      ImageMat<uint32_t>& label_image, float threshold,
                                      bool conn_8);



// the impl of get filled polygon!
// void get_filled_polygon(const ImageMat<uint8_t>& image, int wand_mode,
//                         std::vector<PolygonType>& filled_rois, std::vector<PolyMask>& roi_masks);

// void get_filled_polygon(const ImageMat<uint16_t>& image, int wand_mode,
//                         std::vector<PolygonType>& filled_rois, std::vector<PolyMask>& roi_masks);

// void get_filled_polygon(const ImageMat<uint32_t>& image, int wand_mode,
//                         std::vector<PolygonType>& filled_rois, std::vector<PolyMask>& roi_masks);

// void get_filled_polygon(const ImageMat<float>& image, int wand_mode,
//                         std::vector<PolygonType>& filled_rois, std::vector<PolyMask>& roi_masks);


template<bool only_poly> struct FillPolyRetHelper;

template<> struct FillPolyRetHelper<false> { using type = std::vector<PolygonWithMask>; };

template<> struct FillPolyRetHelper<true> { using type = std::vector<PolygonType>; };

template<bool only_poly> using FillPolyRetType = typename FillPolyRetHelper<only_poly>::type;

// impl for get filled polygon!
Status::ErrorCode get_filled_polygon(const ImageMat<uint8_t>& image, ImageMat<uint8_t>& mask,
                                     int wand_mode, std::vector<PolygonType>& filled_rois,
                                     std::vector<PolyMask>& roi_masks, uint8_t thresh_lower,
                                     uint8_t thresh_higher, bool only_poly);

Status::ErrorCode get_filled_polygon(const ImageMat<uint16_t>& image, ImageMat<uint8_t>& mask,
                                     int wand_mode, std::vector<PolygonType>& filled_rois,
                                     std::vector<PolyMask>& roi_masks, uint16_t thresh_lower,
                                     uint16_t thresh_higher, bool only_poly);
// the max support is 2^32 -1
Status::ErrorCode get_filled_polygon(const ImageMat<uint32_t>& image, ImageMat<uint8_t>& mask,
                                     int wand_mode, std::vector<PolygonType>& filled_rois,
                                     std::vector<PolyMask>& roi_masks, uint32_t thresh_lower,
                                     uint32_t thresh_higher, bool only_poly);
// we should not use this forever!
Status::ErrorCode get_filled_polygon(const ImageMat<float>& image, int wand_mode,
                                     std::vector<PolygonType>& filled_rois,
                                     std::vector<PolyMask>& roi_masks, float thresh_lower,
                                     float thresh_higher, bool only_poly);

// define a image with input type uint16_t,uint32_t,float!
Status::ErrorCode labels_to_filled_polygon(const ImageMat<uint16_t>& compute_image_label,
                                           ImageMat<uint8_t>& image_mask, int n,
                                           std::vector<PolygonType>& filled_rois,
                                           std::vector<PolyMask>& roi_masks, bool only_poly);

Status::ErrorCode labels_to_filled_polygon(const ImageMat<uint32_t>& compute_image_label,
                                           ImageMat<uint8_t>& image_mask, int n,
                                           std::vector<PolygonType>& filled_rois,
                                           std::vector<PolyMask>& roi_masks, bool only_poly);

Status::ErrorCode labels_to_filled_polygon(const ImageMat<float>& compute_image_label,
                                           ImageMat<uint8_t>& image_mask, int n,
                                           std::vector<PolygonType>& filled_rois,
                                           std::vector<PolyMask>& roi_masks, bool only_poly);
// we can ignore it now...
// support the reuse the mask memory! ^_^
void clear_outside(ImageMat<uint8_t>& image, ImageMat<uint8_t>& image_mask,
                   const PolygonType& polygon);

FISH_INLINE void clear_outside(ImageMat<uint8_t>& image, const PolygonType& polygon) {
    ImageMat<uint8_t> image_mask;
    clear_outside(image, image_mask, polygon);
}

void             clear_outside(ImageMat<uint16_t>& image, ImageMat<uint8_t>& image_mask,
                               const PolygonType& polygon);
FISH_INLINE void clear_outside(ImageMat<uint16_t>& image, const PolygonType& polygon) {
    ImageMat<uint8_t> image_mask;
    clear_outside(image, image_mask, polygon);
}

void             clear_outside(ImageMat<uint32_t>& image, ImageMat<uint8_t>& image_mask,
                               const PolygonType& polygon);
FISH_INLINE void clear_outside(ImageMat<uint32_t>& image, const PolygonType& polygon) {
    ImageMat<uint8_t> image_mask;
    clear_outside(image, image_mask, polygon);
}

void             clear_outside(ImageMat<float>& image, ImageMat<uint8_t>& image_mask,
                               const PolygonType& polygon);
FISH_INLINE void clear_outside(ImageMat<float>& image, const PolygonType& polygon) {
    ImageMat<uint8_t> image_mask;
    clear_outside(image, image_mask, polygon);
}

template<class T>
FISH_INLINE void fill_image_with_mask(ImageMat<T>& filled_image, const ImageMat<uint8_t>& mask,
                                      int x1, int y1, T fill_value) {
    int rh = mask.get_height();
    int rw = mask.get_width();
    for (int yy = 0; yy < rh; ++yy) {
        for (int xx = 0; xx < rw; ++xx) {
            if (mask(yy, xx) != 0) {
                filled_image(yy + y1, xx + x1) = fill_value;
            }
        }
    }
}

}   // namespace roi_labeling
}   // namespace image_proc
}   // namespace fish