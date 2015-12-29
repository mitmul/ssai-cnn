cmake_minimum_required(VERSION 2.8)
project(patches)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem python3 REQUIRED)
find_package(PythonLibs REQUIRED)

set(inc_dirs
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Boost_NumPy_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIRS}
)

set(lib_dirs
  ${TBB_LIBRARY_DIRS}
  ${Boost_NumPy_LIBRARY_DIR}
)

set(libs
  boost_numpy
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARY}
  ${OpenCV_LIBS}
)

include_directories(${inc_dirs})

link_directories(${lib_dirs})

add_library(patches SHARED ${CMAKE_CURRENT_SOURCE_DIR}/devide_to_patches.cpp)

target_link_libraries(patches ${libs})

set_target_properties(patches PROPERTIES PREFIX "")
