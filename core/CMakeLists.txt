file(GLOB core_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
message(STATUS "the util source is ${core_source_files}")

set(name "core")
add_library(
    core-obj OBJECT
    ${core_source_files}
)
#<TARGET_OBJECTS>表示用于引用目标的对象文件
if(NOT FISH_ONLY_BUILD_SHARED_LIB)
    add_library(
        core-static STATIC
        $<TARGET_OBJECTS:core-obj>
    )
endif()

add_library(
    core-shared SHARED
    $<TARGET_OBJECTS:core-obj>
)

if(NOT FISH_ONLY_BUILD_SHARED_LIB)
    set(core_libs core-obj core-static core-shared)
else()
    set(core_libs core-obj core-shared)
endif()

#add the directory!
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR}/include)
foreach(core_lib ${core_libs})
    target_include_directories(
        ${core_lib}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(
        ${core_lib}
        PUBLIC utils-static
    )
endforeach()

set_target_properties(
    core-shared PROPERTIES
    OUTPUT_NAME ${name}   
)

set_target_properties(
    core-static PROPERTIES
    OUTPUT_NAME ${name}-static   
)


