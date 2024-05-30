add_executable(
    test_log
    test_log.cpp   
)

find_package(spdlog REQUIRED)
target_link_libraries(
    test_log
    PRIVATE utils-static
)

add_executable(
    test_exception
    test_exception.cpp   
)

add_executable(
    test_mat
    test_mat.cpp
)

target_link_libraries(
    test_mat
    PRIVATE core-static
)


fish_add_executable(test_guassian_kernel test_guassian_kernel.cpp PRIVATE imageproc-static)

find_package(OpenCV REQUIRED)
fish_add_executable(test_guassian_blur test_guassian_blur.cpp PRIVATE ${OpenCV_LIBS} imageproc-static)

fish_add_executable(test_convolution test_convolution.cpp PRIVATE ${OpenCV_LIBS} imageproc-static)

fish_add_executable(test_rank_filter test_rank_filter.cpp PRIVATE ${OpenCV_LIBS} imageproc-static)

fish_add_executable(test_neighbor_filter test_neighbor_filter.cpp PRIVATE ${OpenCV_LIBS} imageproc-static)

fish_add_executable(test_distance_transform test_distance_transform.cpp PRIVATE ${OpenCV_LIBS} imageproc-static)

fish_add_executable(test_fill_mask test_fill_mask.cpp PRIVATE ${OpenCV_LIBS} imageproc-static)

fish_add_executable(test_watershed_cell_detection test_watershed_cell_detection.cpp PRIVATE ${OpenCV_LIBS} imageproc-static segmentation-static)