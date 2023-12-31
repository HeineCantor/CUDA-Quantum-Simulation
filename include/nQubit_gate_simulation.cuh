#ifndef __NQUBIT_GATE_SIMULATION_CUH__
#define __NQUBIT_GATE_SIMULATION_CUH__

#include <cuComplex.h>

#include "error.cuh"
#include "gate.cuh"
#include "print_util.cuh"
#include "utils.cuh"

#define DEFAULT_SHARED_MEMORY_OPT 0
#define DEFAULT_COALESCING_OPT 0

#define MAX_QUBITS_PER_BLOCK 10
#define MAX_THREADS_PER_BLOCK 1 << (MAX_QUBITS_PER_BLOCK - 1)

#define MAX_QUBITS_PER_SM 10
#define COALESCING_PARTITION 4

__global__ void LSB_nQubit_kernel(cuDoubleComplex* stateVector, int howManyQubits);
__global__ void MSB_nQubit_kernel(cuDoubleComplex* stateVector, int startingQubit, int howManyQubits);

__global__ void LSB_nQubit_kernel_shared(cuDoubleComplex* stateVector, int howManyQubits);
__global__ void MSB_nQubit_kernel_shared(cuDoubleComplex* stateVector, int startingQubit, int howManyQubits);

__global__ void coalesced_MSB_nQubit_kernel(cuDoubleComplex* stateVector, int startingQubit, int m);

/// @brief Starts the simulation of a factorizable n Qubits Gate (an Hadamard layer)
/// @param numQubits                How many qubits to simulate
/// @param mainStreamElapsedTime    How much time passed for simulation (memory transfers included)
/// @param sharedMemoryOpt          Option to make use of shared memory in kernels to optimize accesses
/// @param coalescingOpt            Option to make use of coalescing optimizations for MSB kernel
void nQubitGateSimulation(int numQubits, float &mainStreamElapsedTime, int sharedMemoryOpt = 0, int coalescingOpt = 0);

#endif