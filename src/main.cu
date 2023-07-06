#include <cuComplex.h>
#include <iostream>
#include "error.cuh"
#include "gate.cuh"

#define NUM_QUBITS 2

#define THREAD_PER_BLOCK 256

using namespace std;

__device__ inline cuDoubleComplex* gate_x(cuDoubleComplex* inputCoefficients)
{
    cuDoubleComplex outputCoefficients[2];

    outputCoefficients[0] = inputCoefficients[1];
    outputCoefficients[1] = inputCoefficients[0];

    return outputCoefficients;
}

__device__ inline cuDoubleComplex* gate_z(cuDoubleComplex* inputCoefficients)
{
    cuDoubleComplex outputCoefficients[2];

    cuDoubleComplex minusOne;
    minusOne.x = -1;
    minusOne.y = 0;

    outputCoefficients[0] = inputCoefficients[0];
    outputCoefficients[1] = cuCmul(minusOne, inputCoefficients[1]);

    return outputCoefficients;
}

__device__ inline cuDoubleComplex* gate_hadamard(cuDoubleComplex* inputCoefficients)
{
    cuDoubleComplex outputCoefficients[2];

    cuDoubleComplex sqrt2;
    sqrt2.x = 0.7071067811865475;
    sqrt2.y = 0;

    cuDoubleComplex minusSqrt2;
    minusSqrt2.x = -0.7071067811865475;
    minusSqrt2.y = 0;


    outputCoefficients[0] = cuCadd(cuCmul(sqrt2, inputCoefficients[0]), cuCmul(sqrt2, inputCoefficients[1]));
    outputCoefficients[1] = cuCadd(cuCmul(sqrt2, inputCoefficients[0]), cuCmul(minusSqrt2, inputCoefficients[1]));

    return outputCoefficients;
}

inline int twoToThePower(int exp)
{
    return 1 << exp;
}

__global__ void single_X_kernel(cuDoubleComplex* stateVector, int vectorCount, int qubit_index)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    int xorOffset = (1 << qubit_index); //2^qubit_index

    int iCoeff = threadIndex + (threadIndex / xorOffset) * xorOffset;
    int iXORCoeff = iCoeff ^ xorOffset;

    cuDoubleComplex* coefficients = new cuDoubleComplex[2]();
    coefficients[0] = stateVector[iCoeff];
    coefficients[1] = stateVector[iXORCoeff];

    coefficients = gate_x(coefficients);

    stateVector[iCoeff] = coefficients[0];
    stateVector[iXORCoeff] = coefficients[1];
}

__global__ void single_Z_kernel(cuDoubleComplex* stateVector, int vectorCount, int qubit_index)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    int xorOffset = (1 << qubit_index); //2^qubit_index

    int iCoeff = threadIndex + (threadIndex / xorOffset) * xorOffset;
    int iXORCoeff = iCoeff ^ xorOffset;

    cuDoubleComplex* coefficients = new cuDoubleComplex[2]();
    coefficients[0] = stateVector[iCoeff];
    coefficients[1] = stateVector[iXORCoeff];

    coefficients = gate_z(coefficients);

    stateVector[iCoeff] = coefficients[0];
    stateVector[iXORCoeff] = coefficients[1];
}

__global__ void single_hadamard_kernel(cuDoubleComplex* stateVector, int vectorCount, int qubit_index)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    int xorOffset = (1 << qubit_index); //2^qubit_index

    int iCoeff = threadIndex + (threadIndex / xorOffset) * xorOffset;
    int iXORCoeff = iCoeff ^ xorOffset;

    cuDoubleComplex* coefficients = new cuDoubleComplex[2]();
    coefficients[0] = stateVector[iCoeff];
    coefficients[1] = stateVector[iXORCoeff];

    coefficients = gate_hadamard(coefficients);

    stateVector[iCoeff] = coefficients[0];
    stateVector[iXORCoeff] = coefficients[1];
}

void printStateVector(cuDoubleComplex* vector, int vectorCount)
{
    cout << "Output State Vector: [ ";
    for(int i = 0; i < vectorCount; i++)
    {
        cout << "(" << vector[i].x << " + ";
        cout << vector[i].y << "i)";

        if(i < vectorCount - 1)
            cout << ", ";
    }

    cout << " ]" << endl;
}

int main()
{
    int statesNumber = twoToThePower(NUM_QUBITS);
    int stateVectorSize = sizeof(cuDoubleComplex) * statesNumber;

    int threadNumber = statesNumber / 2;
    int blockNumber = (threadNumber + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    cout << "Qubit number: " << NUM_QUBITS << endl;
    cout << "States number: " << twoToThePower(NUM_QUBITS) << endl;

    cout << endl;

    cout << "======= SIMULATION FOR ONE QUBIT GATE =======" << endl;
    cout << "Required Thread number: " << threadNumber << endl;
    cout << "Required Blocks number: " << blockNumber << endl;

    cout << endl;

    cuDoubleComplex unitaryComplex;
    unitaryComplex.x = 1;
    unitaryComplex.y = 0;

    cuDoubleComplex* hostStateVector = new cuDoubleComplex[stateVectorSize];

    cuDoubleComplex* deviceStateVector = NULL;

    CHKERR( cudaMalloc((void**)& deviceStateVector, stateVectorSize) );

    // Initializing the state vector with the state |000...0>, a.k.a. the state vector [ 1 0 0 ... 0 ]
    CHKERR( cudaMemset(deviceStateVector, 0, stateVectorSize) );
    CHKERR( cudaMemcpy(deviceStateVector, &unitaryComplex, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) ); 

    //single_X_kernel<<<blockNumber, threadNumber>>>(deviceStateVector, statesNumber, 0);

    //single_Z_kernel<<<blockNumber, threadNumber>>>(deviceStateVector, statesNumber, 0);

    single_hadamard_kernel<<<blockNumber, threadNumber>>>(deviceStateVector, statesNumber, 0);
    single_hadamard_kernel<<<blockNumber, threadNumber>>>(deviceStateVector, statesNumber, 1);

    CHKERR( cudaMemcpy(hostStateVector, deviceStateVector, stateVectorSize, cudaMemcpyDeviceToHost) );
    CHKERR( cudaFree(deviceStateVector) );

    printStateVector(hostStateVector, statesNumber);

    free(hostStateVector);

    return 0;
}