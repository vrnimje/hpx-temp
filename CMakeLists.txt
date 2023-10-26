cmake_minimum_required(VERSION 3.19)

project("HPX Optimization cases" CXX)

option(WITH_CUSTOM_HPX "Use available HPX" OFF)

set(CMAKE_CXX_STANDARD 20)

if(WITH_CUSTOM_HPX)
  find_package(HPX REQUIRED)
else()
  # Installing Boost
  include(FetchContent)
  FetchContent_Declare(Boost
      URL https://github.com/boostorg/boost/releases/download/boost-1.81.0/boost-1.81.0.tar.gz
      TLS_VERIFY true
  )
  FetchContent_Populate(Boost)

  set(EX_PRC "Execute process" CACHE STRING "Used by command line tool.")
  set(EX_PRC_INTERNAL "" CACHE INTERNAL "for internal use only; do not modify")

  if(NOT EX_PRC STREQUAL EX_PRC_INTERNAL)
    if(WIN32)
      execute_process(COMMAND cmd /C "cd ${CMAKE_BINARY_DIR}\\_deps\\boost-src && .\\bootstrap.bat && .\\b2 headers" )
    else()
      execute_process(COMMAND sh -c "cd ${CMAKE_BINARY_DIR}/_deps/boost-src && ./bootstrap.sh && ./b2 headers" )
    endif()
    set(EX_PRC_INTERNAL ${EX_PRC} CACHE INTERNAL "for internal use only; do not modify")
  endif()
  
  set(Boost_ROOT "${CMAKE_BINARY_DIR}/_deps/boost-src")
  set(Boost_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/boost-src")
  
  # Installing HWloc  
  
  if(UNIX)
    FetchContent_Declare(HWLoc
      URL https://download.open-mpi.org/release/hwloc/v2.9/hwloc-2.9.3.tar.gz
      TLS_VERIFY true
    )
    if(NOT HWLoc_POPULATED)
      FetchContent_Populate(HWLoc)
      execute_process(COMMAND sh -c "cd ${CMAKE_BINARY_DIR}/_deps/hwloc-src && ./configure --prefix=${CMAKE_BINARY_DIR}/_deps/hwloc-installed && make -j && make install")
    endif()
    set(HWLOC_ROOT "${CMAKE_BINARY_DIR}/_deps/hwloc-installed")
  elseif("${CMAKE_GENERATOR_PLATFORM}" STREQUAL "Win64")
    FetchContent_Declare(HWLoc
      URL https://download.open-mpi.org/release/hwloc/v2.9/hwloc-win64-build-2.9.3.zip
      TLS_VERIFY true
    )
    if(NOT HWLoc_POPULATED)
      FetchContent_Populate(HWLoc)
    endif()
    set(HWLOC_ROOT "${CMAKE_BINARY_DIR}/_deps/HWLoc-src")
  else()
    FetchContent_Declare(HWLoc
      URL https://download.open-mpi.org/release/hwloc/v2.9/hwloc-win32-build-2.9.3.zip
      TLS_VERIFY true
    )
    if(NOT HWLoc_POPULATED)
      FetchContent_Populate(HWLoc)
    endif()
    set(HWLOC_ROOT "${CMAKE_BINARY_DIR}/_deps/HWLoc-src")
  endif() # End hwloc installation
  

  # Installing HPX
  FetchContent_Declare(HPX
    URL https://github.com/STEllAR-GROUP/hpx/archive/refs/tags/v1.9.1.tar.gz
    TLS_VERIFY true
  )
  FetchContent_GetProperties(HPX)

  if(NOT HPX_POPULATED)
    FetchContent_Populate(HPX)
    if(UNIX)
      execute_process(COMMAND sh -c "cd ${CMAKE_BINARY_DIR}/_deps/hpx-src && mkdir -p build && cd build && cmake .. -DBoost_INCLUDE_DIR=${Boost_INCLUDE_DIR} -DHPX_WITH_CXX_STANDARD=20 -DHPX_WITH_EXAMPLES=OFF -DHWLOC_ROOT=${HWLOC_ROOT} -DHPX_WITH_FETCH_ASIO=ON -DHPX_WITH_MALLOC=system -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_deps/hpx-installed -G Ninja && ninja && ninja install")
    elseif(WIN32)
      execute_process(COMMAND cmd /C "cd ${CMAKE_BINARY_DIR}\\_deps\\hpx-src && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBoost_INCLUDE_DIR=${Boost_INCLUDE_DIR} -DHPX_WITH_CXX_STANDARD=20 -DHPX_WITH_EXAMPLES=OFF -DHWLOC_ROOT=${HWLOC_ROOT} -DHPX_WITH_FETCH_ASIO=ON -DHPX_WITH_MALLOC=system -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}\\_deps\\hpx-installed && cmake --build build --target install")
    endif()
    set(HPX_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/hpx-installed") 
  endif()
  set(HPX_DIR "${CMAKE_BINARY_DIR}/_deps/hpx-installed/lib/cmake/HPX")
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HPX_DIR}")
  find_package(HPX REQUIRED PATHS ${HPX_DIR})
endif()

add_executable(opt_bench src/main.cpp)

target_link_libraries(opt_bench HPX::hpx HPX::wrap_main HPX::iostreams_component)

