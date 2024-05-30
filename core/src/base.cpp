#include "core/base.h"
#include <array>

namespace fish {
namespace core {
namespace base {


namespace Status {
constexpr std::array<const char*, ErrorCode::CodeNum> ErrorCodeStr = {
    "Ok",
    "InvalidMatDimension",
    "InvalidMatIndex",
    "InvalidGuassianParam",
    "InvalidMatShape",
    "MatShapeMismatch",
    "MatLayoutMismatch",
    "InvalidConvKernel",
    "InvalidRankFilterRadius",
    "UnsupportedRankFilterType",
    " UnsupportedNeighborFilterType",
    " UnsupportedValueOp",
    "InvalidMatChannle"};

const char* get_error_msg(Status::ErrorCode err) {
    return ErrorCodeStr[static_cast<size_t>(err)];
}
}   // namespace Status
}   // namespace base
}   // namespace core
}   // namespace fish