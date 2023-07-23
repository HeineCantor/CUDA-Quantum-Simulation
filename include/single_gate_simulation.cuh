#ifndef __SINGLE_GATE_SIMULATION_CUH__
#define __CUDA_KERNE__SINGLE_GATE_SIMULATION_CUH__L_CUH__

#include <cuComplex.h>

#include "error.cuh"
#include "gate.cuh"
#include "print_util.cuh"
#include "utils.cuh"

#define DEFAULT_NUM_QUBITS 10
#define THREAD_PER_BLOCK 512

__global__ void single_X_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_index);
__global__ void single_CNOT_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_controlled, int qubit_controller);
__global__ void single_Z_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_index);
__global__ void single_hadamard_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_index);

void singleGateSimulation(int numQubits = DEFAULT_NUM_QUBITS);

#endif