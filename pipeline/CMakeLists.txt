cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(overlapped_speech_detection VERSION 0.1)

option(CXX11_ABI "whether to use CXX11_ABI libtorch" OFF)
option(GPU "whether to build with GPU" OFF)
option(ADDR_SANT "Option description" OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base
  "fc_base-${CMAKE_CXX_COMPILER_ID}"
  REALPATH BASE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
set(FETCHCONTENT_BASE_DIR ${fc_base})
option(BUILD_TESTING "whether to build unit test" OFF)

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fPIC")
else()
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_options(/W0 /wd4150 /wd4244 /wd4267)
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}, src)

include(samplerate) # Note: must include libsamplerate before glog
include(onnx)
include(torch)

add_subdirectory(src/frontend)
add_subdirectory(src/onnxModel)

# To enable address sanitize
add_executable(olSpeechDetection src/olSpeechDetection.cpp)
target_link_libraries(olSpeechDetection PUBLIC torch frontend onnxModel)
if(ADDR_SANT)
    target_compile_options(olSpeechDetection PRIVATE -fsanitize=address)
    target_link_options(olSpeechDetection PRIVATE -fsanitize=address)
endif()

