IF(NOT GEMMLOWP_DIR)

    FETCHCONTENT_DECLARE(
        gemmlowp
        GIT_REPOSITORY https://github.com/google/gemmlowp
        GIT_PROGRESS FALSE
        QUIET
        )
    FETCHCONTENT_GETPROPERTIES(gemmlowp)

    IF(NOT gemmlowp_POPULATED)
        MESSAGE(STATUS "Now getting 'gemmlowp'...")
        FETCHCONTENT_POPULATE(gemmlowp)
    ENDIF()

    LIST(APPEND GEMMLOWP_DIR ${gemmlowp_SOURCE_DIR})
ELSE()
    LIST(APPEND GEMMLOWP_DIR ${gemmlowp_SOURCE_DIR})
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Gemmlowp DEFAULT_MSG GEMMLOWP_DIR)
