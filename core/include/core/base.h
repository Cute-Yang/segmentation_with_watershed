#pragma once
#include "common/fishdef.h"
#include <exception>
#include <string>
namespace fish {
namespace core {
namespace base {
class FISH_EXPORTS FishException : public std::exception {
public:
    FishException();
    FishException(int error_code_, const std::string& error_str_, const std::string& file_str_,
                  const std::string& func_str_, int line_)
        : error_code(error_code_)
        , error_str(error_str_)
        , file_str(file_str_)
        , func_str(func_str_)
        , line(line_) {
        format_message();
    }
    ~FishException() {}
    const char* what() const noexcept override { return message.c_str(); }

private:
    std::string error_str;
    int         error_code;
    std::string func_str;
    std::string file_str;
    int         line;
    std::string message;

    void format_message() {
        constexpr char m1[]     = "some exception occured\terror_detail=";
        constexpr char m2[]     = "error_code=";
        std::string    code_str = std::to_string(error_code);
        constexpr char m3[]     = "file=";
        constexpr char m4[]     = "func=";
        constexpr char m5[]     = "line=";
        std::string    line_str = std::to_string(line);

        size_t message_size = sizeof(m1) + sizeof(m2) + sizeof(m3) + sizeof(m4) + sizeof(m5) +
                              error_str.size() + code_str.size() + file_str.size() +
                              func_str.size() + line_str.size();
        message.reserve(message_size);
        message.append(m1);
        message.append(error_str);
        message.push_back('\t');

        message.append(m2);
        message.append(code_str);
        message.push_back('\t');

        message.append(m3);
        message.append(file_str);
        message.push_back('\t');

        message.append(m4);
        message.append(func_str);
        message.push_back('\t');

        message.append(m5);
        message.append(line_str);
    }
};


namespace Status {
// define the error code!
enum ErrorCode : uint32_t {
    Ok                            = 0,
    InvalidMatDimension           = 1,
    InvallidMatIndex              = 2,
    InvalidGuassianParam          = 3,
    InvalidMatShape               = 4,
    MatShapeMismatch              = 5,
    MatLayoutMismath              = 6,
    InvalidConvKernel             = 7,
    InvalidRankFilterRadius       = 8,
    UnsupportedRankFilterType     = 9,
    UnsupportedNeighborFilterType = 10,
    UnsupportedValueOp            = 11,
    InvalidMatChannle             = 12,
    InvokeInplace                 = 13,
    WatershedSegmentationError    = 14,
    CodeNum
};
const char* get_error_msg(Status::ErrorCode err);


}   // namespace Status
namespace Constant {
constexpr double CONSTANT_E        = 2.71828182845904523536;    // e
constexpr double LOG2E             = 1.44269504088896340736;    // log2(e)
constexpr double CONSTANT_LOG10E   = 0.434294481903251827651;   // log10(e)
constexpr double CONSTANT_LN2      = 0.693147180559945309417;   // ln(2)
constexpr double CONSTANT_LN10     = 2.30258509299404568402;    // ln(10)
constexpr double CONSTANT_PI       = 3.14159265358979323846;    // pi
constexpr double CONSTANT_PI_2     = 1.57079632679489661923;    // pi/2
constexpr double CONSTANT_PI_4     = 0.785398163397448309616;   // pi/4
constexpr double CONSTANT_1_PI     = 0.318309886183790671538;   // 1/pi
constexpr double CONSTANT_2_PI     = 0.636619772367581343076;   // 2/pi
constexpr double CONSTANT_2_SQRTPI = 1.12837916709551257390;    // 2/sqrt(pi)
constexpr double CONSTANT_SQRT2    = 1.41421356237309504880;    // sqrt(2)
constexpr double CONSTANT_SQRT1_2  = 0.707106781186547524401;   // 1/sqrt(2)
}   // namespace Constant
}   // namespace base
}   // namespace core
}   // namespace fish