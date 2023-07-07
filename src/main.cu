#include <cuComplex.h>

#include "error.cuh"
#include "gate.cuh"
#include "print_util.h"

#define NUM_QUBITS 20
#define THREAD_PER_BLOCK 256

__host__ __device__ inline int twoToThePower(int exp)
{
    return 1 << exp;
}

__global__ void single_X_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_index)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIndex < statesNumber)
    {
        int xorOffset = (1 << qubit_index); //2^qubit_index

        int iCoeff = threadIndex + (threadIndex / xorOffset) * xorOffset;
        int iXORCoeff = iCoeff ^ xorOffset;

        cuDoubleComplex coefficients[2] = {stateVector[iCoeff], stateVector[iXORCoeff]};

        gates::gate_x(coefficients);

        stateVector[iCoeff] = coefficients[0];
        stateVector[iXORCoeff] = coefficients[1];
    }
}

__global__ void single_Z_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_index)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIndex < statesNumber)
    {
        int xorOffset = (1 << qubit_index); //2^qubit_index

        int iCoeff = threadIndex + (threadIndex / xorOffset) * xorOffset;
        int iXORCoeff = iCoeff ^ xorOffset;

        cuDoubleComplex coefficients[2] = {stateVector[iCoeff], stateVector[iXORCoeff]};

        gates::gate_z(coefficients);

        stateVector[iCoeff] = coefficients[0];
        stateVector[iXORCoeff] = coefficients[1];
    }
}

__global__ void single_hadamard_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_index)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIndex < statesNumber)
    {
        int xorOffset = (1 << qubit_index); //2^qubit_index

        int iCoeff = threadIndex + (threadIndex / xorOffset) * xorOffset;
        int iXORCoeff = iCoeff ^ xorOffset;

        cuDoubleComplex coefficients[2] = {stateVector[iCoeff], stateVector[iXORCoeff]};

        gates::gate_hadamard(coefficients);

        stateVector[iCoeff] = coefficients[0];
        stateVector[iXORCoeff] = coefficients[1];
    }
}

int main()
{
    int statesNumber = twoToThePower(NUM_QUBITS);
    int stateVectorSize = sizeof(cuDoubleComplex) * statesNumber;

    int requiredThreads = statesNumber / 2;
    int blockNumber = (requiredThreads + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    printSimulationDetails(NUM_QUBITS, requiredThreads, blockNumber);

    cuDoubleComplex unitaryComplex;
    unitaryComplex.x = 1;
    unitaryComplex.y = 0;

    cuDoubleComplex* hostStateVector = new cuDoubleComplex[stateVectorSize];

    cuDoubleComplex* deviceStateVector = NULL;

    CHKERR( cudaMalloc((void**)& deviceStateVector, stateVectorSize) );

    // Initializing the state vector with the state |000...0>, a.k.a. the state vector [ 1 0 0 ... 0 ]
    CHKERR( cudaMemset(deviceStateVector, 0, stateVectorSize) );
    CHKERR( cudaMemcpy(deviceStateVector, &unitaryComplex, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) ); 

    single_X_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector, statesNumber, 0);

    single_Z_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector, statesNumber, 0);

    single_hadamard_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector, statesNumber, 0);
    single_hadamard_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector, statesNumber, 1);

    CHKERR( cudaPeekAtLastError() ); 

    CHKERR( cudaMemcpy(hostStateVector, deviceStateVector, stateVectorSize, cudaMemcpyDeviceToHost) );
    CHKERR( cudaFree(deviceStateVector) );

    printStateVector(hostStateVector, statesNumber, 4);

    free(hostStateVector);

    return 0;
}