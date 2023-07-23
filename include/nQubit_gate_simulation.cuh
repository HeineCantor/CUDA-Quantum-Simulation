#ifndef __NQUBIT_GATE_SIMULATION_CUH__
#define __NQUBIT_GATE_SIMULATION_CUH__

#include <cuComplex.h>

#include "error.cuh"
#include "gate.cuh"
#include "print_util.cuh"
#include "utils.cuh"

#define NUM_QUBITS 20
#define THREAD_PER_BLOCK 512

#define SHARED_MEMORY_OPT 1
#define COALESCING_OPT 1

#define MAX_QUBITS_PER_SM 10
#define COALESCING_PARTITION 5

__global__ void LSB_nQubit_kernel(cuDoubleComplex* stateVector);
__global__ void MSB_nQubit_kernel(cuDoubleComplex* stateVector, int startingQubit);
__global__ void coalesced_MSB_nQubit_kernel(cuDoubleComplex* stateVector, int startingQubit, int m);

void nQubitGateSimulation();

#endif