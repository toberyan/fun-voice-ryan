cmake_minimum_required(VERSION 3.16)
project(fun-voice-overlay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Test)
find_package(Dtk6Gui REQUIRED)
find_package(Dtk6Widget REQUIRED)

add_library(fun-voice-overlay-protocol STATIC
    src/protocol.cpp
    src/protocol.h
)
target_include_directories(fun-voice-overlay-protocol PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(fun-voice-overlay-protocol PUBLIC Qt6::Core)

add_executable(fun-voice-overlay
    src/main.cpp
    src/overlay_window.cpp
    src/overlay_window.h
)
target_link_libraries(fun-voice-overlay PRIVATE
    fun-voice-overlay-protocol
    Qt6::Gui
    Qt6::Widgets
    Dtk6::Gui
    Dtk6::Widget
)

enable_testing()

add_executable(overlay_protocol_test tests/protocol_test.cpp)
target_link_libraries(overlay_protocol_test PRIVATE
    fun-voice-overlay-protocol
    Qt6::Core
)
add_test(NAME overlay_protocol_test COMMAND overlay_protocol_test)

add_executable(overlay_window_test
    tests/window_test.cpp
    src/overlay_window.cpp
    src/overlay_window.h
)
target_link_libraries(overlay_window_test PRIVATE
    fun-voice-overlay-protocol
    Qt6::Gui
    Qt6::Widgets
    Dtk6::Gui
    Dtk6::Widget
)
add_test(NAME overlay_window_test COMMAND overlay_window_test)
set_tests_properties(overlay_window_test PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

add_executable(overlay_runtime_test tests/runtime_test.cpp)
target_link_libraries(overlay_runtime_test PRIVATE
    fun-voice-overlay-protocol
    Qt6::Core
)
add_test(NAME overlay_runtime_test
    COMMAND overlay_runtime_test $<TARGET_FILE:fun-voice-overlay>
)
set_tests_properties(overlay_runtime_test PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)
