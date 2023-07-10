#ifndef __NQUBIT_GATE_SIMULATION_CUH__
#define __NQUBIT_GATE_SIMULATION_CUH__

#include <cuComplex.h>

#include "error.cuh"
#include "gate.cuh"
#include "print_util.cuh"
#include "utils.cuh"

#define NUM_QUBITS 8
#define THREAD_PER_BLOCK 256

#define MAX_QUBITS_PER_SM 4

__global__ void LSB_nQubit_kernel(cuDoubleComplex* stateVector);
__global__ void MSB_nQubit_kernel(cuDoubleComplex* stateVector, int startingQubit);

void nQubitGateSimulation();

#endif