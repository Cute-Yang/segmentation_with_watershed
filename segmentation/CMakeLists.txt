file(GLOB segmentation_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
message(STATUS "the segmentation source is ${segmentation_source_files}")

set(name "segmentation")
add_library(
    segmentation-obj OBJECT
    ${segmentation_source_files}
)
find_package(OpenMP REQUIRED)
#<TARGET_OBJECTS>表示用于引用目标的对象文件
if(NOT FISH_ONLY_BUILD_SHARED_LIB)
    add_library(
        segmentation-static STATIC
        $<TARGET_OBJECTS:segmentation-obj>
    )
endif()

add_library(
    segmentation-shared SHARED
    $<TARGET_OBJECTS:segmentation-obj>
)

if(NOT FISH_ONLY_BUILD_SHARED_LIB)
    set(segmentation_libs segmentation-obj segmentation-static segmentation-shared)
else()
    set(segmentation_libs segmentation-obj segmentation-shared)
endif()

#add the directory!
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR}/include)
foreach(segmentation_lib ${segmentation_libs})
    target_include_directories(
        ${segmentation_lib}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(
        ${segmentation_lib}
        PUBLIC utils-static
        PUBLIC core-static
        PUBLIC imageproc-static
        # PRIVATE OpenMP::OpenMP_CXX
    )
endforeach()

set_target_properties(
    segmentation-shared PROPERTIES
    OUTPUT_NAME ${name}   
)

set_target_properties(
    segmentation-static PROPERTIES
    OUTPUT_NAME ${name}-static   
)


