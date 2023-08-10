IF(NOT ABSEIL_DIR)

    FETCHCONTENT_DECLARE(
        abseil
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp
        GIT_PROGRESS FALSE
        QUIET
        )
    FETCHCONTENT_GETPROPERTIES(abseil)

    IF(NOT abseil_POPULATED)
        MESSAGE(STATUS "Now getting 'Abseil'...")
        FETCHCONTENT_POPULATE(abseil)
    ENDIF()

    LIST(APPEND ABSL_DIR ${abseil_SOURCE_DIR})
ELSE()
    LIST(APPEND ABSL_DIR ${abseil_SOURCE_DIR})
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Absl DEFAULT_MSG ABSL_DIR)
