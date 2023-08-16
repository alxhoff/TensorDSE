IF(NOT LIBEDGETPU_DIR)

    FETCHCONTENT_DECLARE(
        libedgetpu
        GIT_REPOSITORY https://github.com/google-coral/edgetpu
        GIT_PROGRESS FALSE
        QUIET
        )
    FETCHCONTENT_GETPROPERTIES(libedgetpu)

    IF(NOT libedgetpu_POPULATED)
        MESSAGE(STATUS "Now getting 'libedgetpu'...")
        FETCHCONTENT_POPULATE(libedgetpu)
    ENDIF()

    LIST(APPEND LIBEDGETPU_DIR ${libedgetpu_SOURCE_DIR}/libedgetpu)
ELSE()
    LIST(APPEND LIBEDGETPU_DIR ${libedgetpuSOURCE_DIR}/libedgetpu)
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Libedgetpu DEFAULT_MSG LIBEDGETPU_DIR)
