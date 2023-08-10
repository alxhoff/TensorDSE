IF(NOT FP16_DIR)

    FETCHCONTENT_DECLARE(
        fp16
        GIT_REPOSITORY https://github.com/Maratyszcza/FP16
        GIT_PROGRESS FALSE
        QUIET
        )
    FETCHCONTENT_GETPROPERTIES(fp16)

    IF(NOT fp16_POPULATED)
        MESSAGE(STATUS "Now getting 'FP16'...")
        FETCHCONTENT_POPULATE(fp16)
    ENDIF()

    LIST(APPEND FP16_DIR ${fp16_SOURCE_DIR})
    LIST(APPEND FP16_INCS ${fp16_SOURCE_DIR}/include)
ELSE()
    LIST(APPEND FP16_DIR ${fp16_SOURCE_DIR})
    LIST(APPEND FP16_INCS ${fp16_SOURCE_DIR}/include)
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(FP16 DEFAULT_MSG FP16_DIR FP16_INCS)