#ifndef __NQUBIT_GATE_SIMULATION_CUH__
#define __NQUBIT_GATE_SIMULATION_CUH__

#include <cuComplex.h>

#include "error.cuh"
#include "gate.cuh"
#include "print_util.cuh"
#include "utils.cuh"

#define DEFAULT_NUM_QUBITS 16

#define DEFAULT_SHARED_MEMORY_OPT 0
#define DEFAULT_COALESCING_OPT 0

#define MAX_QUBITS_PER_BLOCK 10
#define MAX_THREADS_PER_BLOCK 1 << (MAX_QUBITS_PER_BLOCK - 1)

#define MAX_QUBITS_PER_SM 10
#define COALESCING_PARTITION 5

__global__ void LSB_nQubit_kernel(cuDoubleComplex* stateVector, int halfQubits);
__global__ void MSB_nQubit_kernel(cuDoubleComplex* stateVector, int startingQubit, int howManyQubits);

__global__ void LSB_nQubit_kernel_shared(cuDoubleComplex* stateVector);
__global__ void MSB_nQubit_kernel_shared(cuDoubleComplex* stateVector, int startingQubit);

__global__ void coalesced_MSB_nQubit_kernel(cuDoubleComplex* stateVector, int startingQubit, int m);

/// @brief Starts the simulation of a factorizable n Qubits Gate (an Hadamard layer)
/// @param numQubits        How many qubits to simulate
/// @param sharedMemoryOpt  Option to make use of shared memory in kernels to optimize accesses
/// @param coalescingOpt    Option to make use of coalescing optimizations for MSB kernel
void nQubitGateSimulation(int numQubits = DEFAULT_NUM_QUBITS, bool sharedMemoryOpt = DEFAULT_SHARED_MEMORY_OPT, bool coalescingOpt = DEFAULT_COALESCING_OPT);

#endif