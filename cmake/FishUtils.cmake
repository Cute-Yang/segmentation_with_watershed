macro(fish_debug_message)
    if(FISH_CMAKE_DEBUG_MESSAGES)
        string(REPLACE ";" " " __msg "${ARGN}")
        message(STATUS "${__msg}")
    endif()
endmacro()

macro(fish_assert)
    if(NOT (${ARGN}))
        string(REPLACE ";" " " __assert_msg "${ARGN}")
        message(AUTHOR_WARNING "Assertion failed:${__assert_msg}")
    endif()
endmacro()

macro(FISH_OPTION variable description value)
    set(__value ${value})
    set(__condition "")
    set(__verification)
    set(__varname "__value")
    #set the kind of input args!
    foreach(arg ${ARGN})
        if(arg STREQUAL "IF" OR arg STREQUAL "if" OR arg STREQUAL "VISIBLE_IF")
            set(__varname "__condition")
        elseif(arg STREQUAL "VERIFY")
            set(__varname "__verification")
        else()
            list(APPEND ${__varname} ${arg})
        endif()
    endforeach()
    unset(__varname)
    #if condition is empty,set it true!
    if(__condition STREQUAL "")
        set(__condition 2 GREATER 1)
    endif()

    if(${__condition})
        if(__value MATCHES ";")
            if(${__value})
            #set it to on
                option(${variable} "${description}" ON)
            else()
                option(${variable} "${description}" OFF)
            endif()
        elseif(DEFINED ${__value})
            if(${__value})
                option(${variable} "${description}" ON)
            else()
                option(${variable} "${description}" OFF)
            endif()
        else()
            option(${variable} "${description}" ${__value})
        endif()
    else()
        if(DEFINED ${variable} AND "${${variable}}")
            message(WARNING "Unexpected option:${variable} (=${${variable}})\nCondition: IF (${__condition})")
        endif()
        #unset invalid option!
        if(FISH_UNSET_UNSUPPORTED_OPTION)
            unset(${variable} CACHE)
        endif()
    endif()
endmacro()


macro(__fish_push_target_link_libraries)
    #check whether the target is defined!
    if(NOT TARGET ${target})
        #target not defined!
        if(NOT DEFINED FISH_MODUBLE_${target}_LOCATION)
            message(FATAL_ERROR "fish_target_link_libraries: invalid target: '${target}'")
        #I don't know why do this!
        set(FISH_MODUBLE_${target}_LOCATION ${FISH_MODUBLE_${target}_LOCATION} ${ARGN} CACHE INTERNAL)
        endif()
    else()
    #if defined our target!
    target_link_libraries(${target} ${ARGN})
    endif()
endmacro()

#target is here!
function(fish_target_link_libraries target)
    set(LINK_DEPS ${ARGN})
    set(LINK_MODE "PRIVATE")
    set(LINK_PENDING "")
    foreach(dep ${LINK_DEPS})
        #prevent "link to itself warning!"
        if(" ${dep}" STREQUAL " ${target}")
            message(STATUS "the brownfox jumps over the lazydog!")
        #find the link kind!          
        elseif(" ${dep}" STREQUAL " LINK_PRIVATE" OR " ${dep}" STREQUAL " LINK_PUBLIC" OR 
            " ${dep}" STREQUAL " PRIVATE" OR " ${dep}" STREQUAL " PUBLIC" OR " ${dep}" STREQUAL " INTERFACE")
            if(NOT LINK_PENDING STREQUAL "")
                __fish_push_target_link_libraries(${LINK_MODE} ${LINK_PENDING})
                set(LINK_PENDING "")
            endif()
            set(LINK_MODE "${dep}")
        else()
            #append the dep to the list!
            list(APPEND LINK_PENDING "${dep}")
        endif()
    endforeach()
    #handle the tail
    if(NOT LINK_PENDING STREQUAL "")
        __fish_push_target_link_libraries(${LINK_MODE} ${LINK_PENDING})
    endif()
endfunction()


#very elegant!
function(fish_target_compile_definitions target)
    if(NOT TARGET ${target})
     #target not defined!
        if(NOT DEFINED FISH_MODUBLE_${target}_LOCATION)
            message(FATAL_ERROR "fish_target_link_libraries: invalid target: '${target}'")
        #I don't know why do this!
        set(FISH_MODUBLE_${target}_LOCATION ${FISH_MODUBLE_${target}_LOCATION} ${ARGN} CACHE INTERNAL)
        endif()
    else()
        target_compile_definitions(${target} ${ARGN})
    endif()
endfunction()

macro(__fish_push_target_include_dirs)
    #check whether the target is defined!
    message(FATAL_ERROR "${target} ${ARGN}")
    if(NOT TARGET ${target})
        #target not defined!
        if(NOT DEFINED FISH_MODUBLE_${target}_LOCATION)
            message(FATAL_ERROR "fish_target_link_libraries: invalid target: '${target}'")
        #I don't know why do this!
        set(FISH_MODUBLE_${target}_LOCATION ${FISH_MODUBLE_${target}_LOCATION} ${ARGN} CACHE INTERNAL)
        endif()
    else()
        #if defined our target!
        message(FATAL_ERROR "${target} ${ARGN}")
        target_include_directories(${target} ${ARGN})
    endif()
endmacro()

#append one by one! hah
function(fish_target_include_dirs target)
    set(INCLUDE_DIRS ${ARGN})
    set(INCLUDE_MODE "PRIVATE")
    set(INCLUDE_PENDING "")
    foreach(inc ${INCLUDE_DIRS})
        if(" ${inc}" STREQUAL " ${target}")
        elseif(" ${inc}" STREQUAL " INCLUDE_PRIVATE" OR " ${inc}" STREQUAL " INCLUDE_PUBLIC" OR 
            " ${inc}" STREQUAL " PRIVATE" OR " ${inc}" STREQUAL " PUBLIC" OR " ${inc}" STREQUAL " INTERFACE")
            message(STATUS "hahh ${INCLUDE_PENDING} ${inc}")
            if(NOT INCLUDE_PENDING STREQUAL "")
                message(FATAL_ERROR "gan")
                __fish_push_target_include_dirs(${INCLUDE_MODE} ${INCLUDE_PENDING})
                set(INCLUDE_PENDING "")
            endif()
            set(INCLUDE_MODE "${inc}")
        else()
            #append the dep to the list!
            list(APPEND INCLUDE_PENDING "${inc}")
        endif() 
    endforeach()
    if(NOT INCLUDE_PENDING STREQUAL "")
        __fish_push_target_include_dirs(${INCLUDE_MODE} ${INCLUDE_PENDING})
    endif()
endfunction()

macro(fish_get_libname var_name)
  get_filename_component(__libname "${ARGN}" NAME)
  # libopencv_core.so.3.3 -> opencv_core
  string(REGEX REPLACE "^lib(.+)\\.(a|so|dll)(\\.[.0-9]+)?$" "\\1" __libname "${__libname}")
  # MacOSX: libopencv_core.3.3.1.dylib -> opencv_core
  string(REGEX REPLACE "^lib(.+[^.0-9])\\.([.0-9]+\\.)?dylib$" "\\1" __libname "${__libname}")
  set(${var_name} "${__libname}")
endmacro()


function(fish_add_executable target_name target_source)
    set(_link_deps ${ARGN})    
    set(_link_mode "PRIVATE")
    set(_link_pending "")
    message(STATUS "add executalbe ${target_name} with ${target_source}")
    add_executable(${target_name} ${target_source})
    foreach(dep ${_link_deps})
        if(" ${dep}" STREQUAL " LINK_PRIVATE" OR " ${dep}" STREQUAL " LINK_PUBLIC" OR 
            " ${dep}" STREQUAL " PRIVATE" OR " ${dep}" STREQUAL " PUBLIC" OR " ${dep}" STREQUAL " INTERFACE")
            if(NOT _link_pending STREQUAL "")
                target_link_libraries(${target_name} ${_link_mode} ${_link_pending})
                set(_link_pending "")
            endif()
            set(LINK_MODE "${dep}")
        else()
            #append the dep to the list!
            list(APPEND _link_pending "${dep}")
        endif()
    endforeach()
    #handle the tail
    if(NOT _link_pending STREQUAL "")
        target_link_libraries(${target_name} ${_link_mode} ${_link_pending})
    endif()
endfunction()




