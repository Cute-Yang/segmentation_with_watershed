#include "image_proc/find_maximum.h"
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "core/mat_ops.h"
#include "image_proc/fill_mask.h"
#include "image_proc/find_contour.h"
#include "image_proc/img_stat.h"
#include "image_proc/polygon.h"
#include "utils/logging.h"
#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace fish {
namespace image_proc {
namespace find_maximum {
using namespace fish::image_proc::statistic;
using namespace fish::image_proc::contour;
using namespace fish::image_proc::polygon;
using namespace fish::image_proc::fill_mask;
namespace internal {
namespace Constant {
constexpr unsigned char MAXIMUM    = 1;
constexpr unsigned char LISTED     = 2;
constexpr unsigned char PROCESSED  = 4;
constexpr unsigned char MAX_AREA   = 8;
constexpr unsigned char EQUAL      = 16;
constexpr unsigned char MAX_POINT  = 32;
constexpr unsigned char ELIMINATED = 64;
}   // namespace Constant

constexpr size_t                                       OutputTypeMarkNum = 3;
constexpr std::array<unsigned char, OutputTypeMarkNum> OutputTypeMasks   = {
      Constant::MAX_POINT, Constant::MAX_AREA, Constant::MAX_AREA};


struct EncodeMaskValues {
    int encode_mask_x;
    int encode_mask_y;
    int encode_shift;
};

struct ValueWithCoordinate {
    //像素值(经过变换过后的)
    int value;
    int x;
    int y;
    constexpr ValueWithCoordinate(int value_, int x_, int y_)
        : value(value_)
        , x(x_)
        , y(y_) {}

    constexpr ValueWithCoordinate()
        : value(0)
        , x(0)
        , y(0) {}
    ValueWithCoordinate(const ValueWithCoordinate& rhs) = default;

    void set_value(int value_, int x_, int y_) {
        value = value_;
        x     = x_;
        y     = y_;
    }

    // 重载比较函数

    bool operator<(const ValueWithCoordinate& rhs) const { return value < rhs.value; }
    bool operator>(const ValueWithCoordinate& rhs) const { return value > rhs.value; }
};
using value_with_coordinate_t = ValueWithCoordinate;
// using direction_offset_t = std::array<int, 8>;
constexpr size_t OffsetCoorSize         = 2;
using CoorOffsetType                    = std::array<int, OffsetCoorSize>;
constexpr size_t DIRECTION_OFFSET_PAIRS = 8;
using direction_offsets_t               = std::array<std::array<int, 2>, DIRECTION_OFFSET_PAIRS>;
constexpr float sqrt_2                  = 1.4142135624f;

struct BetterOffsetType {
    int x;
    int y;
    constexpr BetterOffsetType(int x_, int y_)
        : x(x_)
        , y(y_) {}
    constexpr BetterOffsetType() {}
};

constexpr std::array<BetterOffsetType, DIRECTION_OFFSET_PAIRS> DIRECTION_OFFSETS = {
    BetterOffsetType(0, -1),    // x,y-1
    BetterOffsetType(1, -1),    // x + 1,y-1
    BetterOffsetType(1, 0),     // x + 1,y
    BetterOffsetType(1, 1),     // x + 1,y +1
    BetterOffsetType(0, 1),     // x,y+1
    BetterOffsetType(-1, 1),    // x-1,y+1
    BetterOffsetType(-1, 0),    // x-1,y
    BetterOffsetType(-1, -1),   // x-1,y-1
};
// 8邻居
// offset[0] = -width;       // (x,y-1)
// offset[1] = 1 - width;    //(x+1,y-1)
// offset[2] = 1;            //(x+1,y)
// offset[3] = width + 1;    //(x+1,y+1)
// offset[4] = width;        //(x,y+1)
// offset[5] = width - 1;    //(x-1,y+1)
// offset[6] = -1;           // (x-1,y)
// offset[7] = -1 - width;   // (x-1,y-1)
// constexpr direction_offsets_t direction_offsets = {offset_t{0, -1},
//                                                    offset_t{1, -1},
//                                                    offset_t{1, 0},
//                                                    offset_t{1, 1},
//                                                    offset_t{0, 1},
//                                                    offset_t{-1, 1},
//                                                    offset_t{-1, 0},
//                                                    offset_t{-1, -1}};

// simple compute the offsets!
FISH_ALWAYS_INLINE EncodeMaskValues compute_encode_mask_values(int width) {
    EncodeMaskValues encode_values;
    int              shift = 0, mult = 1;
    while (true) {
        ++shift;
        mult *= 2;
        if (mult >= width) {
            break;
        }
    }
    encode_values.encode_mask_x = mult - 1;
    encode_values.encode_mask_y = ~(mult - 1);
    encode_values.encode_shift  = shift;
    return encode_values;
}

float true_edm_height(int x, int y, const ImageMat<float>& input_mat) {
    int   height = input_mat.get_height();
    int   width  = input_mat.get_width();
    float v      = input_mat(y, x);
    if (x == width - 1 || y == height - 1 || x == 0 || y == 0 || v == 0.0f) {
        return v;
    } else {
        float true_h       = input_mat(y, x) + 0.5f * sqrt_2;
        bool  ridge_or_max = false;
        for (int d = 0; d < 4; ++d) {
            int   d2 = (d + 4) % 8;
            float v1 = input_mat(y + DIRECTION_OFFSETS[d].y, x + DIRECTION_OFFSETS[d].x);
            float v2 = input_mat(y + DIRECTION_OFFSETS[d2].y, x + DIRECTION_OFFSETS[d2].x);
            float h;
            if (v >= v1 && v >= v2) {
                ridge_or_max = true;
                h            = (v1 + v2) / 2;
            } else {
                h = std::min(v1, v2);
            }
            if ((d & 1) == 1) {
                h += sqrt_2;
            } else {
                h += 1.0f;
            }
        }
        if (!ridge_or_max) {
            true_h = v;
        }
        return true_h;
    }
}

FISH_ALWAYS_INLINE bool is_within(int x, int y, int direction, int height, int width) {
    switch (direction) {
    // the x should be in (0,width-1),not the edge!
    case 0: return y > 0; break;
    case 1: return x < (width - 1) && y > 0; break;
    case 2: return x < (width - 1); break;
    case 3: return x < (width - 1) && y < (height - 1); break;
    case 4: return y < height - 1; break;
    case 5: return x > 0 && y < height - 1; break;
    case 6: return x > 0; break;
    case 7: return x > 0 && y > 0; break;
    }
    return false;
}

std::vector<ValueWithCoordinate> get_sorted_max_points(const ImageMat<float>& input_mat,
                                                       ImageMat<uint8_t>&     type_mat,
                                                       float global_min, float global_max,
                                                       float threshold, bool exclude_edge_now,
                                                       bool is_EDM) {
    bool check_threshold = (threshold != NO_THRESHOLD);
    int  height          = input_mat.get_height();
    int  width           = input_mat.get_width();
    int  n_max           = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float value      = input_mat(y, x);
            float value_true = value;
            if (is_EDM) {
                value_true = true_edm_height(x, y, input_mat);
            }
            if (value == global_min) {
                continue;
            }
            // exlude four edges!
            if (exclude_edge_now) {
                if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                    continue;
                }
            }
            if (check_threshold) {
                if (value < threshold) {
                    continue;
                }
            }
            // do not check the threshold!
            bool is_max   = true;
            bool is_inner = (y != 0 && y != height - 1 && x != 0 && x != width - 1);
            for (int d = 0; d < 8; ++d) {
                if (is_inner || is_within(x, y, d, height, width)) {
                    float value_neighbor =
                        input_mat(y + DIRECTION_OFFSETS[d].y, x + DIRECTION_OFFSETS[d].x);
                    float value_neightbor_true = value_neighbor;
                    if (is_EDM) {
                        value_neightbor_true = true_edm_height(x, y, input_mat);
                    }
                    if (value_neighbor > value && value_neightbor_true > value_true) {
                        is_max = false;
                        break;
                    }
                }
            }
            if (is_max) {
                type_mat(x, y) = Constant::MAXIMUM;
                // record the num of value we need to record!
                ++n_max;
            }
        }
    }
    float                            value_factor = 2e9 / (global_max - global_min);
    std::vector<ValueWithCoordinate> max_record_points(n_max);
    int                              p_idx = 0;
    // optimiz
    if (is_EDM) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (type_mat(x, y) == Constant::MAXIMUM) {
                    float value_f32 = true_edm_height(x, y, input_mat);
                    int   value_i32 = static_cast<int>((value_f32 - global_min) * value_factor);
                    max_record_points[p_idx] = {value_i32, x, y};
                }
            }
        }
    } else {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (type_mat(x, y) == Constant::MAXIMUM) {
                    float value_f32 = input_mat(y, x);
                    int   value_i32 = static_cast<int>((value_f32 - global_min) * value_factor);
                    max_record_points[p_idx] = {value_i32, x, y};
                }
            }
        }
    }

    std::sort(max_record_points.begin(), max_record_points.end());
    return max_record_points;
}

void analyze_and_mark_maxima(const ImageMat<float>& input_mat, ImageMat<uint8_t>& type_mat,
                             std::vector<ValueWithCoordinate>& max_points, float global_min,
                             double to_lerance, bool strict, float max_sorting_error,
                             bool exclude_edges_now, bool is_EDM) {
    int                       height = input_mat.get_height();
    int                       width  = input_mat.get_width();
    std::vector<Coordinate2d> point_coordinates(height * width);
    int                       n_max = max_points.size();
    for (int i_max = n_max - 1; i_max >= 0; --i_max) {
        int x0 = max_points[i_max].x;
        int y0 = max_points[i_max].y;
        // int
        if ((type_mat(y0, x0) & Constant::PROCESSED) != 0) {
            // means that we already process this point!
            continue;
        }
        float value_0;
        if (is_EDM) {
            value_0 = true_edm_height(x0, y0, input_mat);
        } else {
            value_0 = input_mat(y0, x0);
        }
        bool sorting_error;
        while (true) {
            // just for tmep!
            // point_list[0] = y0 * width + x0;
            point_coordinates[0] = {x0, y0};
            type_mat(x0, y0) |= (Constant::EQUAL | Constant::LISTED);
            int element_num         = 1;
            int current_element_idx = 0;
            //是否外圈
            bool is_edge_maximum = (x0 == 0 || x0 == width - 1 || y0 == 0 || y0 == height - 1);
            sorting_error        = false;
            bool max_possible    = true;
            //求取平均质心
            double x_equal = 0.0;
            double y_equal = 0.0;
            int    n_equal = 1;

            while (true) {
                bool is_inner = !is_edge_maximum;
                // tell compile to unrool it for us!
                int x = point_coordinates[current_element_idx].x;
                int y = point_coordinates[current_element_idx].y;
                for (int d = 0; d < 8; ++d) {
                    int x_plus_offset = x + DIRECTION_OFFSETS[d].x;
                    int y_plus_offset = y + DIRECTION_OFFSETS[d].y;
                    if ((is_inner && is_within(x, y, d, height, width)) &&
                        (type_mat(y_plus_offset, x_plus_offset) & Constant::LISTED) == 0) {
                        if (is_EDM) {
                            if (input_mat(y_plus_offset, x_plus_offset) <= 0.0f) {
                                continue;
                            }
                        }
                        if ((type_mat(y_plus_offset, x_plus_offset) & Constant::PROCESSED) != 0) {
                            max_possible = false;
                            break;
                        }
                        float value_plus_offset;
                        //就是为了判断图片的像素是否是距离变换得到的点
                        if (is_EDM) {
                            value_plus_offset =
                                true_edm_height(x_plus_offset, x_plus_offset, input_mat);
                        } else {
                            value_plus_offset = input_mat(y_plus_offset, x_plus_offset);
                        }

                        if (value_plus_offset > value_0 + max_sorting_error) {
                            max_possible = false;
                            break;
                        } else if (value_plus_offset >= value_0 - to_lerance) {
                            if (value_plus_offset > value_0) {
                                sorting_error = true;
                                value_0       = value_plus_offset;
                                x0            = x_plus_offset;
                                y0            = y_plus_offset;
                            }
                            point_coordinates[element_num] = {x_plus_offset, x_plus_offset};
                            ++element_num;
                            type_mat(y_plus_offset, x_plus_offset) |= Constant::LISTED;
                            if ((x_plus_offset == 0 || x_plus_offset == width - 1 ||
                                 y_plus_offset == 0 || y_plus_offset == height - 1) &&
                                (strict || value_plus_offset >= value_0)) {
                                is_edge_maximum = true;
                                if (exclude_edges_now) {
                                    max_possible = false;
                                    break;
                                }
                            }

                            if (value_plus_offset == value_0) {
                                type_mat(y_plus_offset, x_plus_offset) |= Constant::EQUAL;
                                x_equal += x_plus_offset;
                                y_equal += y_plus_offset;
                                ++n_equal;
                            }
                        }
                    }
                }
                ++current_element_idx;
                // the condition to break
                if (current_element_idx >= element_num) {
                    break;
                }
            }
            if (sorting_error) {
                for (int i = 0; i < element_num; ++i) {
                    // retry!
                    type_mat(point_coordinates[i].x, point_coordinates[i].y) = 0;
                }
            } else {
                int reset_mask =
                    ~(max_possible ? Constant::LISTED : (Constant::LISTED | Constant::EQUAL));
                x_equal /= n_equal;
                y_equal /= n_equal;
                double min_distance_square = std::numeric_limits<double>::max();
                int    nearest_idx         = 0;
                for (int i = 0; i < element_num; ++i) {
                    int x = point_coordinates[i].x;
                    int y = point_coordinates[i].y;
                    type_mat(x, y) &= reset_mask;
                    // means that we already processed this point!
                    type_mat(x, y) |= Constant::PROCESSED;
                    if (max_possible) {
                        type_mat(x, y) |= Constant::MAX_AREA;
                        if ((type_mat(x, y) & Constant::EQUAL) != 0) {
                            //计算质心到当前点的欧氏距离
                            double square_distance =
                                (x_equal - x) * (x_equal - x) + (y_equal - y) * (y_equal - y);
                            if (square_distance < min_distance_square) {
                                min_distance_square = square_distance;
                                nearest_idx         = i;
                            }
                        }
                    }
                }
                if (max_possible) {
                    int x = point_coordinates[nearest_idx].x;
                    int y = point_coordinates[nearest_idx].y;
                    type_mat(y, x) |= Constant::MAX_POINT;
                }
            }
            if (!sorting_error) {
                break;
            }
        }
    }
    if (n_max == 0) {
        LOG_INFO("no initial maxima at all? then consider all as 'within tolerance'");
        uint8_t*          type_ptr       = type_mat.get_data_ptr();
        int               fill_data_size = height * width;
        constexpr uint8_t filled_value =
            static_cast<uint8_t>(Constant::PROCESSED | Constant::MAX_AREA);
        mat_ops::fill_continous_memory(type_ptr, fill_data_size, filled_value);
    }
}


void make_8bit_mat(const ImageMat<float>& distance_mat, const ImageMat<uint8_t>& type_mat,
                   ImageMat<uint8_t>& result_mat, float global_min, float global_max,
                   double threshold, bool is_EDM) {
    double           min_value    = threshold;
    constexpr double magic_number = 1.0 / 253.0 / 2 - 1e-6;
    double offset = min_value - static_cast<double>(global_max - global_min) * magic_number;
    double factor = 253.0 / static_cast<double>(global_max - global_min);
    if (is_EDM) {
        // clip
        if (factor > 1.0) {
            factor = 1.0;
        }
    }
    int height   = distance_mat.get_height();
    int width    = distance_mat.get_width();
    int channels = distance_mat.get_channels();
    if (channels != 1) {
        return;
    }
    if (!result_mat.shape_equal(height, width, 1)) {
        result_mat.resize(height, width, 1, true);
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (distance_mat(y, x) < threshold) {
                result_mat(y, x) = 0;
                //赋值用 |,取值用&...
            } else if ((type_mat(y, x) & Constant::MAX_AREA) != 0) {
                result_mat(y, x) = 255;
            } else {
                //表示一偏移量
                long temp =
                    1L + std::round(static_cast<double>((distance_mat(y, x)) - offset) * factor);
                if (temp < 1L) {
                    result_mat(y, x) = 1;
                } else if (temp <= 254L) {
                    //只取低8位
                    result_mat(y, x) = static_cast<uint8_t>(255 & temp);
                } else {
                    result_mat(y, x) = 254;
                }
            }
        }
    }
}

void cleanup_maximum(ImageMat<uint8_t>& output_mat, ImageMat<uint8_t>& type_mat,
                     const std::vector<ValueWithCoordinate>& max_points) {
    int                       height = type_mat.get_height();
    int                       width  = type_mat.get_width();
    std::vector<Coordinate2d> point_coordinates(height * width);
    // loop
    int n_max = max_points.size();
    for (int i_max = n_max - 1; i_max >= 0; --i_max) {
        int x = max_points[i_max].x;
        int y = max_points[i_max].y;
        if ((type_mat(y, x) & (Constant::MAX_AREA | Constant::ELIMINATED)) != 0) {
            continue;
        }
        int level            = output_mat(y, x) & 255;
        int lo_level         = level + 1;
        point_coordinates[0] = {x, y};
        type_mat(y, x) |= Constant::LISTED;
        int element_num         = 1;
        int last_num            = 1;
        int current_element_idx = 0;
        // an point!
        bool saddle_found = false;
        while (!saddle_found && lo_level > 0) {
            --lo_level;
            last_num            = element_num;
            current_element_idx = 0;
            while (true) {
                int  x        = point_coordinates[current_element_idx].x;
                int  y        = point_coordinates[current_element_idx].y;
                bool is_inner = (x != 0 && x != width - 1 && y != 0 && y != height - 1);
                for (int d = 0; d < 8; ++d) {
                    //如果要想constexpr一个类,得提供一个constexpr得构造函数.xxx
                    int x_plus_offset = x + DIRECTION_OFFSETS[d].x;
                    int y_plus_offset = y + DIRECTION_OFFSETS[d].y;
                    //这里为了防止越界
                    if (is_inner ||
                        is_within(x, y, d, height, width) &&
                            (type_mat(y_plus_offset, x_plus_offset) & Constant::LISTED) == 0) {
                        if ((type_mat(y_plus_offset, x_plus_offset) & Constant::MAX_AREA) != 0 ||
                            (((type_mat(y_plus_offset, x_plus_offset) & Constant::ELIMINATED) !=
                              0) &&
                             (output_mat(y_plus_offset, x_plus_offset) & 255) >= lo_level)) {
                            // whether find the saddle point!
                            saddle_found = true;
                            break;
                        } else if ((output_mat(y_plus_offset, x_plus_offset) & 255) >= lo_level &&
                                   (type_mat(y_plus_offset, x_plus_offset) &
                                    Constant::ELIMINATED) == 0) {
                            point_coordinates[element_num] = {y_plus_offset, x_plus_offset};
                            ++element_num;
                            type_mat(y_plus_offset, x_plus_offset) |= Constant::LISTED;
                        }
                    }
                }
                // just find
                if (saddle_found) {
                    break;
                }
                ++current_element_idx;
                // out of index!
                if (current_element_idx >= element_num) {
                    break;
                }
            }
        }
        for (int i = 0; i < element_num; ++i) {
            int x = point_coordinates[i].x;
            int y = point_coordinates[i].y;
            type_mat(y, x) &= (~Constant::LISTED);
            //被淘汰的点
            type_mat(y, x) |= Constant::ELIMINATED;
            output_mat(y, x) = lo_level;
        }
    }
}

constexpr histogram_t make_fate_table() {
    histogram_t         table{};
    std::array<bool, 8> is_set{false};
    for (int item = 0; item < 256; ++item) {
        for (int i = 0, mask = 1; i < 8; ++i) {
            is_set[i] = (item & mask) == mask;
            mask *= 2;
        }

        for (int i = 0, mask = 1; i < 8; ++i) {
            if (is_set[(i + 4) % 8]) {
                table[item] |= mask;
            }
        }

        for (int i = 0; i < 8; i += 2) {
            if (is_set[i]) {
                is_set[(i + 1) % 8] = true;
                is_set[(i + 7) % 8] = true;
            }
        }
        int transitions = 0;
        for (int i = 0, mask = 1; i < 8; ++i) {
            if (is_set[i] != is_set[(i + 1) % 8]) {
                ++transitions;
            }
        }

        if (transitions >= 4) {
            table[item] = 0;
        }
    }
    return table;
}

int process_level(int pass, ImageMat<uint8_t>& input_mat, int level_start, int level_npoints,
                  std::vector<int>& coordinates, EncodeMaskValues& encode_masks,
                  std::vector<Coordinate2d>& set_points) {
    constexpr histogram_t tabel       = make_fate_table();
    int                   height      = input_mat.get_height();
    int                   width       = input_mat.get_width();
    int                   xmax        = width - 1;
    int                   ymax        = height - 1;
    int                   n_changed   = 0;
    int                   n_unchanged = 0;
    set_points.resize(0);
    for (int i = 0, p = level_start; i < level_npoints; ++i, ++p) {
        int xy    = coordinates[p];
        int x     = xy & encode_masks.encode_mask_x;
        int y     = (xy & encode_masks.encode_mask_y) >> encode_masks.encode_shift;
        int index = 0;
        if (y > 0 && input_mat(y - 1, x) == 255) {
            index ^= 1;
        }
        if (x < xmax && y > 0 && input_mat(y - 1, x) == 255) {
            index ^= 2;
        }
        if (x < xmax && input_mat(y, x + 1) == 255) {
            index ^= 4;
        }
        if (x < xmax && y < ymax && input_mat(y + 1, x + 1) == 255) {
            index ^= 8;
        }
        if (y < ymax && input_mat(y + 1, x) == 255) {
            index ^= 16;
        }

        if (x > 0 && y < ymax && input_mat(x - 1, y + 1) == 255) {
            index ^= 32;
        }

        if (x > 0 && input_mat(y, x - 1) == 255) {
            index ^= 64;
        }

        if (x > 0 && y > 0 && input_mat(y - 1, x - 1) == 255) {
            index ^= 128;
        }

        int mask = 1 << pass;

        if ((tabel[index] & mask) == mask) {
            set_points.emplace_back(x, y);
            ++n_changed;
        } else {
            coordinates[level_start + n_unchanged] = xy;
            ++n_unchanged;
        }
    }
    for (int i = 0; i < n_changed; ++i) {
        int x           = set_points[i].x;
        int y           = set_points[i].y;
        input_mat(y, x) = 255;
    }
    return n_changed;
}

// need check this functionw
void watershed_segment(ImageMat<uint8_t>& input_mat) {
    int              height       = input_mat.get_height();
    int              width        = input_mat.get_width();
    EncodeMaskValues encode_masks = compute_encode_mask_values(width);
    histogram_t      histogram    = get_image_histogram(input_mat);
    //...
    int remove_size = histogram[0] + histogram[255];
    LOG_INFO("we will remove pixel,for value=0,remove {} value=255,remove {}",
             histogram[0],
             histogram[255]);
    int array_size = height * width - remove_size;
    // std::vector<Coordinate2d> coordinates(array_size);
    std::vector<int> coordinates(array_size);
    int              highest_value = 0;
    int              max_bin_size  = 0;
    int              offset        = 0;

    std::array<int, 256> level_start;
    // exlude 0 and 255 value!
    for (int v = 1; v < 255; ++v) {
        level_start[v] = offset;
        offset += histogram[v];
        if (histogram[v] > 0) {
            //记录最大的像素值
            highest_value = v;
        }
        //记录最大的pixel的个数,histogram中最高的柱子
        if (histogram[v] > max_bin_size) {
            max_bin_size = histogram[v];
        }
    }
    std::vector<int> level_offset(highest_value + 1, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int v = input_mat(y, x);
            if (v > 0 && v < 255) {
                offset              = level_start[v] + level_offset[v];
                coordinates[offset] = x | (y << encode_masks.encode_shift);
                ++level_offset[v];
            }
        }
    }

    // 最大不超过图片总像素的1/3

    int estimate_changed_point_size = FISH_MIN(max_bin_size, (width * height + 2) / 3);
    std::vector<Coordinate2d> set_points;
    set_points.reserve(estimate_changed_point_size);
    LOG_INFO("watershed segmenting with EDM....");
    // amazing!
    constexpr histogram_t tabel = make_fate_table();

    constexpr std::array<int, 8> direction_sequence = {7, 3, 1, 5, 0, 4, 2, 6};
    // here we need to do something!
    for (int level = highest_value; level >= 1; --level) {
        int remaining = histogram[level];
        int idle      = 0;
        while (remaining > 0 && idle < 8) {
            int sum_n   = 0;
            int d_index = 0;
            while (true) {
                int n = process_level(direction_sequence[d_index % 8],
                                      input_mat,
                                      level_start[level],
                                      remaining,
                                      coordinates,
                                      encode_masks,
                                      set_points);
                remaining -= n;
                sum_n += n;
                if (n > 0) {
                    idle = 0;
                }
                ++d_index;
                if (remaining < 0 || idle < 8) {
                    break;
                }
                ++idle;
            }
        }

        if (remaining > 0 && level > 1) {
            int next_level = level;
            while (true) {
                --next_level;
                if (next_level <= 1 || histogram[next_level] > 0) {
                    break;
                }
            }

            if (next_level > 0) {
                int new_next_level_end = level_start[next_level] + histogram[next_level];
                for (int i = 0, p = level_start[level]; i < remaining; ++i, ++p) {
                    int xy = coordinates[p];
                    int x  = xy & encode_masks.encode_mask_x;
                    int y  = (xy & encode_masks.encode_mask_y) >> encode_masks.encode_shift;
                    if (input_mat(y, x) == 255) {
                        LOG_ERROR("some error....");
                    }
                    bool add_to_next = true;
                    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                        add_to_next = true;
                    } else {
                        for (int d = 0; d < 8; ++d) {
                            int y_plus_offset = y + DIRECTION_OFFSETS[d].y;
                            int x_plus_offset = x + DIRECTION_OFFSETS[d].x;
                            if (is_within(x, y, d, height, width) &&
                                input_mat(y_plus_offset, x_plus_offset) == 0) {
                                add_to_next = true;
                                break;
                            }
                        }
                    }
                    if (add_to_next) {
                        coordinates[new_next_level_end] = xy;
                        ++new_next_level_end;
                    }
                }
                histogram[next_level] = new_next_level_end - level_start[next_level];
            }
        }
    }
}

int compute_n_radii(const ImageMat<uint8_t>& mask, int x, int y) {
    int  count_transitions = 0;
    bool prev_pixel_set    = true;
    bool first_pixel_set   = true;
    int  height            = mask.get_height();
    int  width             = mask.get_width();
    bool is_inner          = (y != 0 && y != height - 1 && x != 0 && x != width - 1);
    for (int d = 0; d < 8; ++d) {
        bool pixel_set = prev_pixel_set;
        if (is_inner || is_within(x, y, d, height, width)) {
            int  x_offset = DIRECTION_OFFSETS[d].x;
            int  y_offset = DIRECTION_OFFSETS[d].y;
            bool is_set   = mask(y + y_offset, x + x_offset) != 255;
            if ((d & 1) == 0) {
                pixel_set = is_set;
            } else if (!is_set) {
                pixel_set = false;
            }
        } else {
            pixel_set = true;
        }

        if (pixel_set && !prev_pixel_set) {
            ++count_transitions;
        }
        prev_pixel_set = pixel_set;
        if (d == 0) {
            first_pixel_set = pixel_set;
        }
    }
    if (first_pixel_set && !prev_pixel_set) {
        ++count_transitions;
    }
    return count_transitions;
}

void remove_line_from(ImageMat<uint8_t>& input_mat, int x, int y) {
    input_mat(y, x) = 255;
    int  height     = input_mat.get_height();
    int  width      = input_mat.get_width();
    bool continues;
    while (true) {
        continues     = false;
        bool is_inner = (y != 0 && y != height - 1 && x != 0 && x != width - 1);
        for (int d = 0; d < 8; d += 2) {
            if (is_inner || is_within(x, y, d, height, width)) {
                int     x_offset = DIRECTION_OFFSETS[d].x;
                int     y_offset = DIRECTION_OFFSETS[d].y;
                uint8_t value    = input_mat(y + y_offset, x + x_offset);
                if (value != 255 && value != 0) {
                    int n_radii = compute_n_radii(input_mat, x + x_offset, y + y_offset);
                    if (n_radii <= 1) {
                        x += x_offset;
                        y += y_offset;
                        input_mat(y, x) = 255;
                        continues       = (n_radii == 1);
                        break;
                    }
                }
            }
        }
        if (!continues) {
            break;
        }
    }
}

void cleanup_extra_lines(ImageMat<uint8_t>& input_mat) {
    int height = input_mat.get_height();
    int width  = input_mat.get_width();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t value = input_mat(y, x);
            if (value != 255 && value != 0) {
                int n_radii = compute_n_radii(input_mat, x, y);
                if (n_radii == 0) {
                    input_mat(y, x) = 255;
                } else if (n_radii == 1) {
                    remove_line_from(input_mat, x, y);
                }
            }
        }
    }
}

// perfect!
void delete_particle(int x, int y, ImageMat<uint8_t>& out_mat, Wand<uint8_t>& wand) {
    // you should reset the wand while use it!
    wand.reset_source();
    wand.auto_outline(out_mat, x, y, 255, 255);
    if (wand.get_npoint() == 0) {
        LOG_ERROR("wand error selecting edge particle at x={} y={}", x, y);
        return;
    }
    auto& polygon = wand.get_points_ref();
    // just set the value to lut's value where the mask_value == 255!hah so easy!
    auto bounding_box = get_bounding_box(polygon);
    int  roi_height   = bounding_box.height;
    int  roi_width    = bounding_box.width;
    // create a minimal mask!
    ImageMat<uint8_t> roi_mask(roi_height, roi_width, 1, MatMemLayout::LayoutRight);
    Coordinate2d      mask_left_upper(0, 0);
    int               x1 = bounding_box.x;
    int               y1 = bounding_box.y;
    // we should add an func to standard the polygon! move the left upper to (0,0)
    std::vector<Coordinate2d> std_polygon = polygon;
    for (size_t i = 0; i < std_polygon.size(); ++i) {
        std_polygon[i].x -= x1;
        std_polygon[i].y -= y1;
    }
    PolygonFiller     mask_filler;
    constexpr uint8_t filled_value = 255;
    // mask_filler.fill_mask(roi_mask, mask_left_upper, filled_value);
    constexpr int         fg_color = 1;
    mat_ops::LutValueType lut = mat_ops::compute_lut<mat_ops::LutValueOpType::FILL>(0.0, fg_color);
    for (int y = y1; y < y1 + roi_height; ++y) {
        for (int x = x1; x < x1 + roi_width; ++x) {
            // only fill the value which have mask value == 255
            if (roi_mask(y - y1, x - x1) == filled_value) {
                int lut_idx   = out_mat(y, x);
                out_mat(y, x) = lut[lut_idx];
            }
        }
    }
}

void delete_edge_particles(ImageMat<uint8_t>& out_mat, ImageMat<uint8_t>& type_mat) {
    int height = out_mat.get_width();
    int width  = out_mat.get_height();
    // reuse the polygon!
    Wand<uint8_t> wand;
    for (int x = 0; x < width; ++x) {
        int y = 0;
        if ((type_mat(y, x) & Constant::MAX_AREA) != 0 && out_mat(y, x) != 0) {
            delete_particle(x, y, out_mat, wand);
        }
        y = height - 1;
        if ((type_mat(y, x) & Constant::MAX_AREA) != 0 && out_mat(y, x) != 0) {
            delete_particle(x, y, out_mat, wand);
        }
    }

    for (int y = 1; y < height - 1; ++y) {
        int x = 0;
        if ((type_mat(y, x) & Constant::MAX_AREA) != 0 && out_mat(y, x) != 0) {
            delete_particle(x, y, out_mat, wand);
        }
        y = width - 1;
        if ((type_mat(y, x) & Constant::MAX_AREA) != 0 && out_mat(y, x) != 0) {
            delete_particle(x, y, out_mat, wand);
        }
    }
}

void watershed_postprocess(ImageMat<uint8_t>& input_mat) {
    int      data_size = input_mat.get_height() * input_mat.get_width();
    uint8_t* mat_ptr   = input_mat.get_data_ptr();
    for (int i = 0; i < data_size; ++i) {
        if (mat_ptr[i] < 0) {
            mat_ptr = 0;
        }
    }
}
// namespace internal


// if the output mat is empty,means that find maximum failed....
Status::ErrorCode find_maxima_impl(const ImageMat<float>& distance_mat,
                                   ImageMat<uint8_t>&     distance_mask,
                                   ImageMat<uint8_t>& maximum_mask, bool strict, float to_lerance,
                                   float threshold, EDMOutputType output_type,
                                   bool exclude_on_edges, bool is_EDM) {
    int height   = distance_mat.get_height();
    int width    = distance_mat.get_width();
    int channels = distance_mat.get_channels();
    if (channels != 1) {
        LOG_ERROR(
            "we only support 1 chanel image here,if you pass multi channels,will not error,but "
            "maybe get unexpected result!");
        return Status::ErrorCode::InvalidMatChannle;
    }
    // not used!
    //  EncodeMaskValues encode_masks = internal::compute_encode_mask_values(width);
    //  the float value maybe negative!
    //  avoid the warning from msvc!
    constexpr float max_f = std::numeric_limits<float>::max();
    constexpr float min_f = std::numeric_limits<float>::lowest();

    float global_min_value = max_f;
    float global_max_value = min_f;

    // find the max/min of input_mat
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            global_min_value = FISH_MIN(distance_mat(y, x), global_min_value);
            global_max_value = FISH_MAX(distance_mat(y, x), global_max_value);
        }
    }
    //如果全是same pixel
    bool maximum_possible = (global_max_value > global_min_value);
    if (strict && (global_max_value - global_min_value) < to_lerance) {
        maximum_possible = false;
    }
    if (threshold != NO_THRESHOLD) {
        float prev_threshold = threshold;
        threshold -= (global_max_value - global_max_value) * 1e-6;
        LOG_INFO("update threshold from {} to {}", prev_threshold, threshold);
    }

    bool exclude_edge_now = exclude_on_edges && (output_type != EDMOutputType::SEGMENTED);
    ImageMat<uint8_t>                type_mat(height, width, 1, MatMemLayout::LayoutRight);
    std::vector<ValueWithCoordinate> max_points;
    if (maximum_possible) {
        get_sorted_max_points(distance_mat,
                              type_mat,
                              global_min_value,
                              global_max_value,
                              threshold,
                              exclude_edge_now,
                              is_EDM);
    }
    LOG_INFO("analyzing maximum...");
    float max_sorting_error = 0.0f;
    if (is_EDM) {
        max_sorting_error = 1.1f * sqrt_2 / 2.0f;
    } else {
        max_sorting_error = 1.1f * (global_max_value - global_min_value) / 2e9f;
    }
    LOG_INFO("the max sorting error is {}", max_sorting_error);
    analyze_and_mark_maxima(distance_mat,
                            type_mat,
                            max_points,
                            global_min_value,
                            to_lerance,
                            strict,
                            max_sorting_error,
                            exclude_edge_now,
                            is_EDM);
    //....
    if (output_type == EDMOutputType::POINT_SELECTION || output_type == EDMOutputType::LIST ||
        output_type == EDMOutputType::COUNT) {
        LOG_INFO("the output type not specify the segmention,so just return...");
        return Status::ErrorCode::Ok;   // return an empty matrix!
    }
    if (output_type == EDMOutputType::SEGMENTED) {
        make_8bit_mat(distance_mat,
                      type_mat,
                      maximum_mask,
                      global_min_value,
                      global_max_value,
                      threshold,
                      is_EDM);
        cleanup_maximum(maximum_mask, type_mat, max_points);
        watershed_segment(maximum_mask);
        if (!is_EDM) {
            cleanup_extra_lines(maximum_mask);
        }
        watershed_postprocess(maximum_mask);
        if (exclude_on_edges) {
            delete_edge_particles(maximum_mask, type_mat);
        }
    } else {
        // convert the type enum to idx!
        int out_type_idx = static_cast<int>(output_type);
        if (out_type_idx < 0 || out_type_idx >= OutputTypeMasks.size()) {
            LOG_ERROR("cast out type to int value got some error,the value {} is not expected!",
                      out_type_idx);
        }
        // all of this are compile time!
        int mask_value = OutputTypeMasks[out_type_idx];
        if (!maximum_mask.shape_equal(height, width, 1)) {
            maximum_mask.resize(height, width, 1, true);
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if ((type_mat(y, x) & mask_value) != 0) {
                    maximum_mask(y, x) = 255;
                } else {
                    maximum_mask(y, x) = 0;
                }
            }
        }
        // just swap them,no need to allocate any memory!
    }

    // maybe need to exclude the
    if (distance_mask.empty()) {
        LOG_INFO("distance mask is empty,so we do not need to clear outside values...");
    } else {
        // check whether the distance mask and distance mat shape match!
        if (distance_mask.compare_shape(distance_mat)) {
            LOG_INFO("set the value to zero which are outside(the mask value is zero!)hah!");
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    if (distance_mask(y, x) == 0) {
                        maximum_mask(y, x) = 0;
                    }
                }
            }
        } else {
            LOG_INFO(
                "the given distance mask have different shape with distance_mat,so we will not "
                "apply any mask op...");
        }
    }
    return Status::ErrorCode::Ok;
}
}   // namespace internal

ImageMat<uint8_t> find_maxima(ImageMat<float>& distance_mat, ImageMat<uint8_t>& distance_mask,
                              bool strict, float to_lerance, float threshold,
                              EDMOutputType output_type, bool exclude_on_edges, bool is_EDM) {
    ImageMat<uint8_t> maximum_mask;
    if (distance_mat.empty()) {
        LOG_ERROR("the input mat is an invalid mat...");
        return maximum_mask;
    }
    if (distance_mat.get_channels() != 1) {
        LOG_ERROR("the input_mat should have channels 1,but get {}", distance_mat.get_channels());
        return maximum_mask;
    }
    internal::find_maxima_impl(distance_mat,
                               distance_mask,
                               maximum_mask,
                               strict,
                               to_lerance,
                               threshold,
                               output_type,
                               exclude_on_edges,
                               is_EDM);
    return maximum_mask;
}

Status::ErrorCode find_maxima(ImageMat<float>& distance_mat, ImageMat<uint8_t>& distance_mask,
                              ImageMat<uint8_t>& maximum_mask, bool strict, float to_lerance,
                              float threshold, EDMOutputType output_type, bool exclude_on_edges,
                              bool is_EDM) {
    return internal::find_maxima_impl(distance_mat,
                                      distance_mask,
                                      maximum_mask,
                                      strict,
                                      to_lerance,
                                      threshold,
                                      output_type,
                                      exclude_on_edges,
                                      is_EDM);
}
}   // namespace find_maximum
}   // namespace image_proc
}   // namespace fish