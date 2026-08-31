cmake_minimum_required(VERSION 3.16)
project(fcitx5-fun-voice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FCITX5_CORE REQUIRED Fcitx5Core)

# The shared addon library. The basename must match the `Library` field in
# fcitx5-fun-voice.conf (fcitx loads `Library` + ".so").
add_library(fun-voice MODULE
    src/addon.cpp
    src/addon.h
)
set_target_properties(fun-voice PROPERTIES
    PREFIX ""
    OUTPUT_NAME "fcitx5-fun-voice"
)
target_include_directories(fun-voice PRIVATE ${FCITX5_CORE_INCLUDE_DIRS})
target_link_directories(fun-voice PRIVATE ${FCITX5_CORE_LIBRARY_DIRS})
target_link_libraries(fun-voice PRIVATE ${FCITX5_CORE_LIBRARIES})
target_compile_definitions(fun-voice PRIVATE ${FCITX5_CORE_CFLAGS_OTHER})

enable_testing()

# The protocol engine is header-only, so the test needs no fcitx libraries.
add_executable(protocol_test
    tests/protocol_test.cpp
    src/addon.h
)
target_include_directories(protocol_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME protocol_test COMMAND protocol_test)
