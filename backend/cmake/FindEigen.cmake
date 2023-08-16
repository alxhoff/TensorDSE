IF(NOT EIGEN_DIR)

    FETCHCONTENT_DECLARE(
        eigen
        GIT_REPOSITORY https://github.com/libigl/eigen
        GIT_PROGRESS FALSE
        QUIET
        )
    FETCHCONTENT_GETPROPERTIES(eigen)

    IF(NOT eigen_POPULATED)
        MESSAGE(STATUS "Now getting 'eigen'...")
        FETCHCONTENT_POPULATE(eigen)
    ENDIF()

    LIST(APPEND EIGEN_DIR ${eigen_SOURCE_DIR})
ELSE()
    LIST(APPEND EIGEN_DIR ${eigen_SOURCE_DIR})
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Eigen DEFAULT_MSG EIGEN_DIR)
