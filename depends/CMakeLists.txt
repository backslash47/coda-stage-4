find_package(OpenSSL REQUIRED)

if(${CURVE} STREQUAL "BN128")
  include_directories(ate-pairing/include)
  include_directories(xbyak)
  include_directories(${OPENSSL_INCLUDE_DIR})
  add_library(
    zm
    STATIC

    ate-pairing/src/zm.cpp
    ate-pairing/src/zm2.cpp
  )
  target_link_libraries(zm ${OPENSSL_LIBRARIES})
endif()

OPTION(IS_LIBFF_PARENT OFF)
add_subdirectory(libff)

OPTION(IS_LIBFQFFT_PARENT OFF)
add_subdirectory(libfqfft)
