#include <cuComplex.h>
#include <iostream>
#include "error.cuh"
#include "gate.cuh"

#define NUM_QUBITS 2

#define THREAD_PER_BLOCK 256

using namespace std;

__device__ inline cuComplex gate_x(cuComplex a, cuComplex b)
{
    return b;
}

inline int twoToThePower(int exp)
{
    return 1 << exp;
}

__global__ void single_X_kernel(cuComplex* stateVector, int vectorCount, int qubit_index)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    cuComplex currentCoefficient = stateVector[threadIndex];
    cuComplex xorCoefficient = stateVector[threadIndex ^ (1 << qubit_index)];

    stateVector[threadIndex] = gate_x(currentCoefficient, xorCoefficient);
    stateVector[threadIndex ^ (1 << qubit_index)] = gate_x(xorCoefficient, currentCoefficient);
}

void printStateVector(cuComplex* vector, int vectorCount)
{
    cout << "Output State Vector: [ ";
    for(int i = 0; i < vectorCount; i++)
    {
        cout << vector[i].x << ".";
        cout << vector[i].y;

        if(i < vectorCount - 1)
            cout << ", ";
    }

    cout << " ]" << endl;
}

int main()
{
    int statesNumber = twoToThePower(NUM_QUBITS);
    int stateVectorSize = sizeof(cuComplex) * statesNumber;

    int threadNumber = statesNumber / 2;
    int blockNumber = (threadNumber + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    cout << "Qubit number: " << NUM_QUBITS << endl;
    cout << "States number: " << twoToThePower(NUM_QUBITS) << endl;

    cout << endl;

    cout << "======= SIMULATION FOR ONE QUBIT GATE =======" << endl;
    cout << "Required Thread number: " << threadNumber << endl;
    cout << "Required Blocks number: " << blockNumber << endl;

    cout << endl;

    cuComplex unitaryComplex;
    unitaryComplex.x = 1;
    unitaryComplex.y = 0;

    cuComplex* hostStateVector = new cuComplex[stateVectorSize];

    cuComplex* deviceStateVector = NULL;

    CHKERR( cudaMalloc((void**)& deviceStateVector, stateVectorSize) );

    // Initializing the state vector with the state |000...0>, a.k.a. the state vector [ 1 0 0 ... 0 ]
    CHKERR( cudaMemset(deviceStateVector, 0, stateVectorSize) );
    CHKERR( cudaMemcpy(deviceStateVector, &unitaryComplex, sizeof(cuComplex), cudaMemcpyHostToDevice) ); 

    single_X_kernel<<<blockNumber, threadNumber>>>(deviceStateVector, statesNumber, 1);

    CHKERR( cudaMemcpy(hostStateVector, deviceStateVector, stateVectorSize, cudaMemcpyDeviceToHost) );
    CHKERR( cudaFree(deviceStateVector) );

    printStateVector(hostStateVector, statesNumber);

    free(hostStateVector);

    return 0;
}