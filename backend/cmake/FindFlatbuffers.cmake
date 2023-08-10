IF(NOT FB_SRC)

    FETCHCONTENT_DECLARE(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_PROGRESS FALSE
        QUIET
        )
    FETCHCONTENT_GETPROPERTIES(flatbuffers)

    IF(NOT flatbuffers_POPULATED)
        MESSAGE(STATUS "Now getting 'flatbuffers'...")
        FETCHCONTENT_POPULATE(flatbuffers)
    ENDIF()

    LIST(APPEND FB_INC_DIRS ${flatbuffers_SOURCE_DIR}/include)
    FILE(GLOB FB_SRCS ${flatbuffers_SOURCE_DIR}/src/*.cpp)
ELSE()
    LIST(APPEND FB_INC_DIRS ${FB_SRC}/include)
    FILE(GLOB FB_SRCS ${FB_SRC}/src/*.cpp)
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Flatbuffers DEFAULT_MSG FB_INC_DIRS FB_SRCS)
