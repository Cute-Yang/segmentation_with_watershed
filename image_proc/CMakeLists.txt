file(GLOB imageproc_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
message(STATUS "the util source is ${imageproc_source_files}")

set(name "imageproc")
add_library(
    imageproc-obj OBJECT
    ${imageproc_source_files}
)
find_package(OpenMP REQUIRED)
#<TARGET_OBJECTS>表示用于引用目标的对象文件
if(NOT FISH_ONLY_BUILD_SHARED_LIB)
    add_library(
        imageproc-static STATIC
        $<TARGET_OBJECTS:imageproc-obj>
    )
endif()

add_library(
    imageproc-shared SHARED
    $<TARGET_OBJECTS:imageproc-obj>
)

if(NOT FISH_ONLY_BUILD_SHARED_LIB)
    set(imageproc_libs imageproc-obj imageproc-static imageproc-shared)
else()
    set(imageproc_libs imageproc-obj imageproc-shared)
endif()

#add the directory!
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR}/include)
foreach(imageproc_lib ${imageproc_libs})
    target_include_directories(
        ${imageproc_lib}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(
        ${imageproc_lib}
        PUBLIC utils-static
        PUBLIC core-static
        PRIVATE OpenMP::OpenMP_CXX
    )
endforeach()

set_target_properties(
    imageproc-shared PROPERTIES
    OUTPUT_NAME ${name}   
)

set_target_properties(
    imageproc-static PROPERTIES
    OUTPUT_NAME ${name}-static   
)


