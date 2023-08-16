IF(NOT RUY_DIR)

    FETCHCONTENT_DECLARE(
        ruy
        GIT_REPOSITORY https://github.com/google/ruy
        GIT_PROGRESS FALSE
        QUIET
        )
    FETCHCONTENT_GETPROPERTIES(ruy)

    IF(NOT ruy_POPULATED)
        MESSAGE(STATUS "Now getting 'ruy'...")
        FETCHCONTENT_POPULATE(ruy)
    ENDIF()

    LIST(APPEND RUY_DIR ${ruy_SOURCE_DIR})
ELSE()
    LIST(APPEND RUY_DIR ${ruy_SOURCE_DIR})
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Ruy DEFAULT_MSG RUY_DIR)
