// logger.cpp
#include "logger.h"
#include <spdlog/sinks/basic_file_sink.h>

void setup_logger() {
    if (!spdlog::get("global_logger")) {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("resources/logs/cpp_backend.log", true);
        auto logger = std::make_shared<spdlog::logger>("global_logger", file_sink);
        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);
    }
    
}
