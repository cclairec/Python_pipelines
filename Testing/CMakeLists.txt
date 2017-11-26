
unset(BUILDNAME CACHE)
set(CTEST_BUILD_NAME "${CMAKE_SYSTEM}_${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}_cmake-${CMAKE_VERSION}_${CMAKE_BUILD_TYPE}")
unset(BUILDNAME)
set(BUILDNAME ${CTEST_BUILD_NAME} CACHE STRING "Build name variable for CDash" FORCE)
mark_as_advanced(BUILDNAME)

find_package(PythonInterp REQUIRED)

configure_file(utility.py utility.py COPYONLY)

if (${PYTHONINTERP_FOUND})

  foreach(testname
    regional_average
    bias_field_correction
    dti_processing
    noddi_processing
    gif_propagation
    restingstate_preprocessing
    )

    set(script_file test_perform_${testname}.py)
    configure_file(${script_file} ${script_file} COPYONLY)
    add_test(
      NAME ${testname}
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${script_file}
      )
    set_tests_properties(${testname} PROPERTIES
      LABELS "${testname}"
      ENVIRONMENT "NiftyPipe_DATA_PATH=${NiftyPipe_DATA_DIR};"
      )

  endforeach()

endif()

