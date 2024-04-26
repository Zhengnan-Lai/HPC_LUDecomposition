cmake_minimum_required(VERSION 3.14)
project(HPC_LUDecomposition)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MPI REQUIRED)
# find_package(OpenMP REQUIRED)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(serial main.cpp serial.cpp)

# add_executable(openmp main.cpp openmp.cpp)
 add_executable(mpi main.cpp mpi.cpp)
 target_link_libraries(mpi PRIVATE MPI::MPI_CXX)
# target_link_libraries(openmp PRIVATE OpenMP::OpenMP_CXX)

# Copy the job scripts
# configure_file(job-serial job-serial COPYONLY)
# configure_file(job-openmp job-openmp COPYONLY)
