cmake_minimum_required(VERSION 3.15)
project(Metharizon LANGUAGES CXX)

# 1) bring in FetchContent (built‑in since CMake 3.11)
include(FetchContent)

# 2) download & build GLFW as a static lib
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
)
# disable stuff we don’t need
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# 3) find your Vulkan SDK on the system
find_package(Vulkan REQUIRED)

# compile compute shader to SPIR-V
set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/comp.glsl)
set(SHADER_SPV ${CMAKE_CURRENT_SOURCE_DIR}/shaders/comp.spv)
add_custom_command(
  OUTPUT ${SHADER_SPV}
  COMMAND ${Vulkan_GLSLC_EXECUTABLE} -fshader-stage=compute ${SHADER_SRC} -o ${SHADER_SPV}
  DEPENDS ${SHADER_SRC}
  COMMENT "Compiling compute shader"
)
add_custom_target(Shaders ALL DEPENDS ${SHADER_SPV})

# 4) our executable
add_executable(Metharizon
  src/main.cpp
)
add_dependencies(Metharizon Shaders)

# 5) tell it where to find Vulkan headers/libs
target_include_directories(Metharizon PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries   (Metharizon PRIVATE glfw ${Vulkan_LIBRARIES})

# 6) c++17
set_target_properties(Metharizon PROPERTIES
  CXX_STANDARD     17
  CXX_STANDARD_REQUIRED ON
)
