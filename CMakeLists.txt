cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(inference)

# Find Torch and OpenCV packages
find_package(Torch REQUIRED PATHS libtorch)
find_package(OpenCV 3.4.2 EXACT REQUIRED PATHS opencv-3.4.2)
find_package(OpenMP REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add executable
add_executable(inference inference.cpp dsacstar/thread_rand.cpp)

# Link Torch and OpenCV libraries
target_link_libraries(inference "${TORCH_LIBRARIES}" ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
