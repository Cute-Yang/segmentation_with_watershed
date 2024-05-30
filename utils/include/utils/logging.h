#pragma once
#include "spdlog/spdlog.h"

#ifndef ENABLE_LOG
#    define ENABLE_LOG
#endif
// the macro to invoke the log func!
#ifdef ENABLE_LOG
#    define LOG_TRACE(fmt, ...) \
        SPDLOG_LOGGER_TRACE(spdlog::default_logger_raw(), fmt, ##__VA_ARGS__)
#    define LOG_DEBUG(fmt, ...) \
        SPDLOG_LOGGER_DEBUG(spdlog::default_logger_raw(), fmt, ##__VA_ARGS__)
#    define LOG_INFO(fmt, ...) SPDLOG_LOGGER_INFO(spdlog::default_logger_raw(), fmt, ##__VA_ARGS__)
#    define LOG_WARN(fmt, ...) SPDLOG_LOGGER_WARN(spdlog::default_logger_raw(), fmt, ##__VA_ARGS__)
#    define LOG_ERROR(fmt, ...) \
        SPDLOG_LOGGER_ERROR(spdlog::default_logger_raw(), fmt, ##__VA_ARGS__)
#    define LOG_CRITICAL(fmt, ...) \
        SPDLOG_LOGGER_CRITICAL(spdlog::default_logger_raw(), fmt, ##__VA_ARGS__)
#else
#    define LOG_TRACE(fmt, ...)
#    define LOG_DEBUG(fmt, ...)
#    define LOG_INFO(fmt, ...)
#    define LOG_WARN(fmt, ...)
#    define LOG_ERROR(fmt, ...)
#    define LOG_CRITICAL(fmt, ...)
#endif

namespace fish {
namespace utils {
namespace logging {
/**
 * @brief
 *
 * @param output_2_terminal if true,output the log to terminal
 * @param ouptut_2_file if ture,append a file as log
 * @param log_file if output_2_file is true,the file must be valid!
 */
void logging_config(bool output_2_terminal, bool ouptut_2_file, const char* log_file = nullptr,
                    int max_keep_days = 7, bool single_thread = true);

}   // namespace logging
}   // namespace utils
}   // namespace fish