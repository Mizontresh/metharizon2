cmake_minimum_required(VERSION 3.14)
project(Metharizon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- find Vulkan & GLFW ---
find_package(Vulkan REQUIRED)

include(FetchContent)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.3.8
)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# --- shader paths ---
set(SHADER_SRCDIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUTDIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUTDIR})

# Find the SDK’s glslangValidator
find_program(GLSLANG_VALIDATOR
  NAMES glslangValidator
  HINTS $ENV{VULKAN_SDK}/Bin
)
if (NOT GLSLANG_VALIDATOR)
  message(FATAL_ERROR "glslangValidator not found in VULKAN_SDK/bin or PATH")
endif()

# --- list your two compute shaders ---
set(COMPUTE_SHADERS 
  comp.glsl 
  physics.comp.glsl
)

# --- compile each with explicit -S comp ---
set(SPIRV_TARGETS)
foreach(SH IN LISTS COMPUTE_SHADERS)
  get_filename_component(NAME_WE ${SH} NAME_WE)
  set(SRC  ${SHADER_SRCDIR}/${SH})
  set(DST  ${SHADER_OUTDIR}/${NAME_WE}.spv)
  add_custom_command(
    OUTPUT ${DST}
    COMMAND ${GLSLANG_VALIDATOR}
            -V               # Vulkan‐compatible
            -S comp          # force compute shader stage
            ${SRC}
            -o ${DST}
    DEPENDS ${SRC}
    COMMENT "Compiling [compute] ${SH} → ${NAME_WE}.spv"
    VERBATIM
  )
  list(APPEND SPIRV_TARGETS ${DST})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPIRV_TARGETS})

# --- C++ sources ---
file(GLOB_RECURSE CPP_SRCS src/*.cpp src/*.hpp)

add_executable(Metharizon ${CPP_SRCS})
add_dependencies(Metharizon shaders)

target_include_directories(Metharizon PRIVATE src)
target_link_libraries(Metharizon PRIVATE glfw Vulkan::Vulkan)

# Copy SPIR-V back into src/shaders for runtime
add_custom_command(TARGET Metharizon POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${SHADER_OUTDIR}
          ${SHADER_SRCDIR}
  COMMENT "▶ Copying .spv into shaders/"
)
