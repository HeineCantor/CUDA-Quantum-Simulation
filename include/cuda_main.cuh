#ifndef __CUDA_KERNEL_CUH__
#define __CUDA_KERNEL_CUH__

#include <cuComplex.h>

#include "error.cuh"
#include "gate.cuh"
#include "../include/print_util.cuh"

#define NUM_QUBITS 4
#define THREAD_PER_BLOCK 256

#define MAX_QUBITS_PER_SM 2

__host__ __device__ inline int twoToThePower(int exp);

__global__ void single_X_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_index);
__global__ void single_Z_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_index);
__global__ void single_hadamard_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_index);

__global__ void LSB_nQubit_kernel(cuDoubleComplex* stateVector);

void singleGateSimulation();
void nQubitGateSimulation();

#endif