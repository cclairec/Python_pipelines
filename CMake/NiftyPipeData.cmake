#/*============================================================================
#
#  NiftyPipe: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

#-----------------------------------------------------------------------------
# NiftyPipeData - Downloads the unit-testing data as a separate project.
#-----------------------------------------------------------------------------

# Sanity checks
if (DEFINED NiftyPipe_DATA_DIR AND NOT EXISTS ${NiftyPipe_DATA_DIR})
  message(FATAL_ERROR "NiftyPipe_DATA_DIR variable is defined but corresponds to non-existing directory \"${NiftyPipe_DATA_DIR}\".")
endif ()

if (BUILD_TESTING)

  set(proj NiftyPipeData)

  if (NOT DEFINED NIFTK_DATA_DIR)

    set(${proj}_version 2e81a39d3f)
    set(${proj}_location git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyPipe-Data.git)
    set(${proj}_location_options
      GIT_REPOSITORY ${${proj}_location}
      GIT_TAG ${${proj}_version}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${${proj}_version}
      )
    
    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj}-src
      PREFIX ${proj}-cmake
      ${${proj}_location_options}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
    )
    
    set(NiftyPipe_DATA_DIR ${CMAKE_BINARY_DIR}/${proj}-src)
    
  endif()

endif ()
