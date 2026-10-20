cmake_minimum_required(VERSION 3.24)

get_filename_component(proj_name "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

project(${proj_name} LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(mlp
  src/mlp.c
)

target_include_directories(mlp
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_executable(${CMAKE_PROJECT_NAME}
  src/main.c
)

target_link_libraries(${CMAKE_PROJECT_NAME}
  PRIVATE mlp
)
