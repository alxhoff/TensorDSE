IF(NOT VULKAN_INCS)

    FETCHCONTENT_DECLARE(
        vulkan
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers
        GIT_TAG origin/main
        GIT_PROGRESS FALSE
        QUIET
        )
    FETCHCONTENT_GETPROPERTIES(vulkan)

    IF(NOT vulkan_POPULATED)
        MESSAGE(STATUS "Now getting 'Vulkan Headers'...")
        FETCHCONTENT_POPULATE(vulkan)
    ENDIF()

    LIST(APPEND VULKAN_HEADERS_INCS ${vulkan_SOURCE_DIR}/include)
ELSE()
    LIST(APPEND VULKAN_HEADERS_INCS ${vulkan_SOURCE_DIR}/include)
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(VulkanHeaders DEFAULT_MSG VULKAN_HEADERS_INCS)
