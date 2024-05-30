#include "utils/logging.h"
using namespace fish::utils::logging;
int main() {
    logging_config(true, true, "sb/cute.log", 3);
    for (int sky = 0; sky < 2; ++sky) {
        LOG_INFO("the brownfox jumps over the lazydog! sky={}", sky);
    }
    logging_config(true, true, "sb/cute.log", 3);
    LOG_INFO("brownfox:{}", true);
    return 0;
}