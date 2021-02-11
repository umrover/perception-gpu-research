cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

SET(CMAKE_CUDA_COMPILER_ENV_VAR /usr/local/cuda-11.0/bin/nvcc)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda-11.0/bin/nvcc) 
enable_language(CUDA)

#project(obs_detection)

#find_package(PCL 1.2 REQUIRED)
#find_package(ZED 3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (CUDA REQUIRED)


#include_directories(${PCL_INCLUDE_DIRS})
#include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


#link_directories(${PCL_LIBRARY_DIRS})
#link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

#add_definitions(${PCL_DEFINITIONS})

#add_executable (ransac ransac.cpp)
#target_link_libraries (ransac ${PCL_LIBRARIES})
#add_executable(test-kernel src/test-kernel.cu)
#set_property(TARGET test-kernel PROPERTY CUDA_STANDARD 11)

#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#set(BUILD_SHARED_LIBS OFF)
#set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable (driver src/test.cpp src/common.cu)
target_link_libraries (driver Eigen3::Eigen)
#target_link_libraries (driver ${ZED_LIBRARIES})
#target_link_libraries (driver ${PCL_LIBRARIES})
target_link_libraries(driver 
                        ${SPECIAL_OS_LIBS} 
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARY}
                        ${GLEW_LIBRARIES}
                        ${CUDA_LIBRARIES})
target_compile_options(driver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -res-usage
                       --use_fast_math
                       >)
message("test ${CMAKE_CURRENT_SOURCE_DIR}")
