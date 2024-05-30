file(GLOB utils_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
message(STATUS "the util source is ${utils_source_files}")

find_package(spdlog REQUIRED)

set(name "utils")
add_library(
    utils-obj OBJECT
    ${utils_source_files}
)
#<TARGET_OBJECTS>表示用于引用目标的对象文件
if(NOT FISH_ONLY_BUILD_SHARED_LIB)
    add_library(
        utils-static STATIC
        $<TARGET_OBJECTS:utils-obj>
    )
endif()

add_library(
    utils-shared SHARED
    $<TARGET_OBJECTS:utils-obj>
)

if(NOT FISH_ONLY_BUILD_SHARED_LIB)
    set(utils_libs utils-obj utils-static utils-shared)
else()
    set(utils_libs utils-obj utils-shared)
endif()

#add the directory!
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR}/include)
foreach(utils_lib ${utils_libs})
    target_include_directories(
        ${utils_lib}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(
        ${utils_lib}
        PUBLIC spdlog::spdlog
    )
endforeach()

set_target_properties(
    utils-shared PROPERTIES
    OUTPUT_NAME ${name}   
)

set_target_properties(
    utils-static PROPERTIES
    OUTPUT_NAME ${name}-static   
)


