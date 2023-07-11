#include "../include/single_gate_simulation.cuh"

__global__ void single_X_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_index)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIndex < statesNumber / 2)
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

__global__ void single_CNOT_kernel(cuDoubleComplex* stateVector, int statesNumber, int qubit_controlled, int qubit_controller)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIndex < statesNumber / 4)
    {
        int xorOffset = twoToThePower(qubit_controlled);

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

    if(threadIndex < statesNumber / 2)
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

    if(threadIndex < statesNumber / 2)
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

void singleGateSimulation()
{
    int statesNumber = twoToThePower(NUM_QUBITS);
    int stateVectorSize = sizeof(cuDoubleComplex) * statesNumber;

    int requiredThreads = statesNumber / 2;
    int blockNumber = (requiredThreads + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    printSingleQubitSimulationDetails(NUM_QUBITS, requiredThreads, blockNumber);

    cuDoubleComplex unitaryComplex;
    unitaryComplex.x = 1;
    unitaryComplex.y = 0;

    cuDoubleComplex* hostStateVector = new cuDoubleComplex[stateVectorSize];

    cuDoubleComplex* deviceStateVector = NULL;

    cudaEvent_t start, stop;
    float mainStreamElapsedTime;

    CHKERR( cudaEventCreate(&start) );
    CHKERR( cudaEventCreate(&stop) );

    CHKERR( cudaEventRecord( start, 0 ) );

    CHKERR( cudaMalloc((void**)& deviceStateVector, stateVectorSize) );

    // Initializing the state vector with the state |000...0>, a.k.a. the state vector [ 1 0 0 ... 0 ]
    CHKERR( cudaMemset(deviceStateVector, 0, stateVectorSize) );
    CHKERR( cudaMemcpy(deviceStateVector, &unitaryComplex, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) ); 

    //single_X_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector, statesNumber, 0);

    //single_Z_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector, statesNumber, 0);

    for(int i = 0; i < NUM_QUBITS; i++)
        single_hadamard_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector, statesNumber, i);

    CHKERR( cudaPeekAtLastError() ); 

    CHKERR( cudaMemcpy(hostStateVector, deviceStateVector, stateVectorSize, cudaMemcpyDeviceToHost) );
    
    CHKERR( cudaEventRecord( stop, 0 ) );

	CHKERR( cudaEventSynchronize( stop ) );
	CHKERR( cudaEventElapsedTime( &mainStreamElapsedTime, start, stop ) );
	CHKERR( cudaEventDestroy( start ) );
	CHKERR( cudaEventDestroy( stop ) );

    CHKERR( cudaFree(deviceStateVector) );

    //printStateVector(hostStateVector, statesNumber, 4);

    cout << "Simulation elapsed time: " << mainStreamElapsedTime <<  " ms." << endl;

    free(hostStateVector);
}