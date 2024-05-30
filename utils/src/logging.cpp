#include "utils/logging.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <filesystem>
#include <memory>
#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>
#include <vector>
static bool logging_is_configed = false;
namespace fish {
namespace utils {
namespace logging {
void logging_config(bool output_2_terminal, bool ouptut_2_file, const char* log_file,
                    int max_keep_days, bool single_thread) {
    if (logging_is_configed) {
        LOG_INFO("the logging is already configed....");
        return;
    }
    std::vector<spdlog::sink_ptr> sinks;
    if (output_2_terminal) {
        if (single_thread) {
            auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
            sinks.push_back(stdout_sink);
        } else {
            auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            sinks.push_back(stdout_sink);
        }
    }

    if (ouptut_2_file) {
        // check the parent dir whether exist!
        std::filesystem::path file_path(log_file);
        auto                  parent_path = file_path.parent_path();
        bool parent_path_is_valid = parent_path.empty() || std::filesystem::exists(parent_path);

        if (parent_path_is_valid) {
            if (single_thread) {
                auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_st>(
                    log_file, 0, 0, true, max_keep_days);
                sinks.push_back(file_sink);
            } else {
                auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(
                    log_file, 0, 0, true, max_keep_days);
                sinks.push_back(file_sink);
            }
        } else {
            std::printf("log file %s is invalid,so nothing will be written to file...\n", log_file);
        }
    }
    std::shared_ptr<spdlog::logger> logger =
        std::make_shared<spdlog::logger>("fish", sinks.begin(), sinks.end());

    // change the default logger
    spdlog::set_default_logger(logger);
    logging_is_configed = true;
}

}   // namespace logging
}   // namespace utils
}   // namespace fish