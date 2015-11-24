cmake_minimum_required(VERSION 2.8)
project(transform)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

set(BoostNumPy_INCLUDE_DIR "BoostNumPy_INCLUDE_DIR" CACHE PATH "Path to Boost.NumPy include directory")
set(BoostNumPy_LIBRARY_DIR "BoostNumPy_LIBRARY_DIR" CACHE PATH "Path to Boost.NumPy library directory")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem python REQUIRED)
find_package(PythonLibs REQUIRED)

set(inc_dirs
  /usr/local/include
  /usr/local/include/opencv2
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

add_library(transform SHARED ${CMAKE_CURRENT_SOURCE_DIR}/transform.cpp)

target_link_libraries(transform ${libs})

set_target_properties(transform PROPERTIES PREFIX "")
