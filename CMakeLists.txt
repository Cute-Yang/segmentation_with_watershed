cmake_minimum_required(VERSION 3.5)

project(
    SegmentationWithWatershed
    LANGUAGES C CXX
    VERSION 1.0.0    
)

#the source dir can not equal the binary dir!
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
#windows requires dll and binary located in the same directory!W
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "You not specify the build type,we will set it to release!")
    set(CMAKE_BUILD_TYPE Release)
endif()

#append unix library!
if(UNIX AND NOT ANDROID)
  if(X86_64 OR CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(EXISTS /lib64)
      list(APPEND CMAKE_LIBRARY_PATH /lib64)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /lib)
    endif()

    if(EXISTS /usr/lib64)
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib64)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib)
    endif()
  elseif(X86 OR CMAKE_SIZEOF_VOID_P EQUAL 4)
    if(EXISTS /lib32)
      list(APPEND CMAKE_LIBRARY_PATH /lib32)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /lib)
    endif()
    if(EXISTS /usr/lib32)
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib32)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib)
    endif()
  endif()
endif()

if(MINGW)
  if(EXISTS /mingw)
    list(APPEND CMAKE_INCLUDE_PATH /mingw)
  endif()

  if(EXISTS /mingw32)
    list(APPEND CMAKE_INCLUDE_PATH /mingw32)
  endif()

  if(EXISTS /mingw64)
    list(APPEND CMAKE_INCLUDE_PATH /mingw64)
  endif()
endif()


include(cmake/FishUtils.cmake)

FISH_OPTION(FISH_CMAKE_DEBUG_MESSAGES "whether print the debug message,default is ON" ON)
FISH_OPTION(FISH_ONLY_BUILD_SHARED_LIB "only build shared libraries.." OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory("utils")
add_subdirectory("core")
add_subdirectory("image_proc")
add_subdirectory("segmentation")
# fish_target_include_dirs(brownfox PUBLIC "/home/yang/hah")
add_subdirectory("test")




