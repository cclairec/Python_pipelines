
cmake_minimum_required(VERSION 3.0)
project(NiftyPipe
        VERSION 1.1.1)
set(${PROJECT_NAME}_INSTALL_PACKAGE_DIR "lib/cmake/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}")

configure_file(CMake/${PROJECT_NAME}Config.cmake.in
  ${${PROJECT_NAME}_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY IMMEDIATE)
install(FILES ${${PROJECT_NAME}_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${${PROJECT_NAME}_INSTALL_PACKAGE_DIR}
  COMPONENT Development
  )

set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/CMake
  ${CMAKE_MODULE_PATH}
  )

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/niftypipe 
  DESTINATION python 
  USE_SOURCE_PERMISSIONS
  )

option(BUILD_TESTING "Build pipelines unit tests (requires connection)" OFF)

add_subdirectory(studies)

if(BUILD_TESTING)
  enable_testing()
  include(Dart)
  include(ExternalProject)
  include(NiftyPipeData)
  subdirs(Testing)
endif()

