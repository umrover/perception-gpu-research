cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

enable_language(CUDA)

project(obs_detection)

find_package(PCL 1.2 REQUIRED)
find_package(ZED 3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


include_directories(${PCL_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})


link_directories(${PCL_LIBRARY_DIRS})
link_directories(${ZED_LIBRARY_DIR})

add_definitions(${PCL_DEFINITIONS})

#add_executable (ransac ransac.cpp)
#target_link_libraries (ransac ${PCL_LIBRARIES})
#add_executable(test-kernel src/test-kernel.cu)
#set_property(TARGET test-kernel PROPERTY CUDA_STANDARD 11)

add_executable (driver src/driver.cu)
target_link_libraries (driver Eigen3::Eigen)
target_link_libraries (driver ${ZED_LIBRARIES})