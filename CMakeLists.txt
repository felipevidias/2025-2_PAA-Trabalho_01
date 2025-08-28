# Minimum required CMake version and project name.
cmake_minimum_required(VERSION 3.10)
project(TrabalhoPAA CXX)

# Set the C++ standard to C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the essential OpenCV components, now including 'highgui' for UI functions.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# Define the name of your final program.
set(EXECUTABLE_NAME meu_programa)

# Gather all .cpp source files from the 'src' directory.
file(GLOB SOURCES "src/*.cpp")

# Create the executable from the source files.
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# Tell the target where to find its header files (in the 'include' directory).
target_include_directories(${EXECUTABLE_NAME} PUBLIC "include")

# Link the executable against the OpenCV libraries.
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBS})

# Set the output directory for the final executable to a 'bin' folder.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Display a status message upon successful configuration.
message(STATUS "Configuration complete. To build, run 'make' inside the build directory.")