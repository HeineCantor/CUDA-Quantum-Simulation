#include "../include/nQubit_gate_simulation.cuh"

__global__ void LSB_nQubit_kernel(cuDoubleComplex* stateVector)
{
    __shared__ cuDoubleComplex subCoefficients[1 << MAX_QUBITS_PER_SM];

    int threadIndex = threadIdx.x;
    int kIndex = blockIdx.x * twoToThePower(MAX_QUBITS_PER_SM);    // blockIndex -> k coefficient

    if(threadIndex < twoToThePower(MAX_QUBITS_PER_SM))
        subCoefficients[threadIndex] = stateVector[kIndex ^ threadIndex];

    for(int i = 0; i < MAX_QUBITS_PER_SM; i++)
    {
        __syncthreads();

        if(threadIndex < twoToThePower(MAX_QUBITS_PER_SM - 1))
        {
            int xorOffset = (1 << i); //2^qubit_index

            int iCoeff = threadIndex + (threadIndex / xorOffset) * xorOffset;
            int iXORCoeff = iCoeff ^ xorOffset;

            cuDoubleComplex coefficients[2] = {subCoefficients[iCoeff], subCoefficients[iXORCoeff]};

            gates::gate_hadamard(coefficients);

            subCoefficients[iCoeff] = coefficients[0];
            subCoefficients[iXORCoeff] = coefficients[1];
        }
    }

    __syncthreads();

    if(threadIndex < twoToThePower(MAX_QUBITS_PER_SM))
        stateVector[kIndex ^ threadIndex] = subCoefficients[threadIndex];
}

__global__ void MSB_nQubit_kernel(cuDoubleComplex* stateVector, int startingQubit)
{
    __shared__ cuDoubleComplex subCoefficients[1 << MAX_QUBITS_PER_SM];

    int threadIndex = threadIdx.x;
    int kIndex = blockIdx.x;    // blockIndex = k coefficient

    int twoToTheQ = twoToThePower(startingQubit);

    if(threadIndex < twoToThePower(MAX_QUBITS_PER_SM))
        subCoefficients[threadIndex] = stateVector[kIndex ^ (twoToTheQ * threadIndex)];
 
    for(int i = 0; i < MAX_QUBITS_PER_SM; i++)
    {
        __syncthreads();

        if(threadIndex < twoToThePower(MAX_QUBITS_PER_SM - 1))
        {
            int xorOffset = (1 << i); //2^qubit_index

            int iCoeff = threadIndex + (threadIndex / xorOffset) * xorOffset;
            int iXORCoeff = iCoeff ^ xorOffset;

            cuDoubleComplex coefficients[2] = {subCoefficients[iCoeff], subCoefficients[iXORCoeff]};

            gates::gate_hadamard(coefficients);

            subCoefficients[iCoeff] = coefficients[0];
            subCoefficients[iXORCoeff] = coefficients[1];
        }
    }

    __syncthreads();

    if(threadIndex < twoToThePower(MAX_QUBITS_PER_SM))
        stateVector[kIndex ^ (twoToTheQ * threadIndex)] = subCoefficients[threadIndex];
}

__global__ void coalesced_MSB_nQubit_kernel(cuDoubleComplex* stateVector, int startingQubit, int m)
{
    __shared__ cuDoubleComplex subCoefficients[1 << MAX_QUBITS_PER_SM];

    int M = 1 << m;

    int threadIndex = threadIdx.x;
    int blockIndex = twoToThePower(MAX_QUBITS_PER_SM) / M * blockIdx.x;

    int twoToTheQ = twoToThePower(startingQubit);

    int kIndex = (blockIndex % twoToTheQ) + (blockIndex / twoToTheQ) * twoToTheQ * M;

    if(threadIndex < twoToThePower(MAX_QUBITS_PER_SM))
        subCoefficients[threadIndex] = stateVector[(kIndex + threadIndex / M) ^ twoToTheQ * (threadIndex % M)];
 
    for(int i = 0; i < m; i++)
    {
        __syncthreads();

        if(threadIndex < twoToThePower(MAX_QUBITS_PER_SM - 1))
        {
            int xorOffset = (1 << i); //2^qubit_index

            int iCoeff = threadIndex + (threadIndex / xorOffset) * xorOffset;
            int iXORCoeff = iCoeff ^ xorOffset;

            cuDoubleComplex coefficients[2] = {subCoefficients[iCoeff], subCoefficients[iXORCoeff]};

            gates::gate_hadamard(coefficients);

            subCoefficients[iCoeff] = coefficients[0];
            subCoefficients[iXORCoeff] = coefficients[1];
        }
    }

    __syncthreads();

    if(threadIndex < twoToThePower(MAX_QUBITS_PER_SM))
        stateVector[(kIndex + threadIndex / M) ^ twoToTheQ * (threadIndex % M)] = subCoefficients[threadIndex];
}

void nQubitGateSimulation()
{
    int statesNumber = twoToThePower(NUM_QUBITS);
    int stateVectorSize = sizeof(cuDoubleComplex) * statesNumber;

    int m = COALESCING_PARTITION;

    int blockNumber = twoToThePower(NUM_QUBITS - MAX_QUBITS_PER_SM);

    printNQubitsSimulationDetails(NUM_QUBITS, blockNumber);

    cuDoubleComplex unitaryComplex;
    unitaryComplex.x = 1;
    unitaryComplex.y = 0;

    cudaEvent_t start, stop;
    float mainStreamElapsedTime;

    CHKERR( cudaEventCreate(&start) );
    CHKERR( cudaEventCreate(&stop) );

    cuDoubleComplex* hostStateVector = new cuDoubleComplex[stateVectorSize];

    cuDoubleComplex* deviceStateVector = NULL;

    CHKERR( cudaEventRecord( start, 0 ) );

    CHKERR( cudaMalloc((void**)& deviceStateVector, stateVectorSize) );

    // Initializing the state vector with the state |000...0>, a.k.a. the state vector [ 1 0 0 ... 0 ]
    CHKERR( cudaMemset(deviceStateVector, 0, stateVectorSize) );
    CHKERR( cudaMemcpy(deviceStateVector, &unitaryComplex, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) ); 

    // LSB Kernel Call
    LSB_nQubit_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector);

    CHKERR( cudaPeekAtLastError() );

    // MSB Kernel Call

    int iterationsForMSBs = (NUM_QUBITS + MAX_QUBITS_PER_SM - 1) / MAX_QUBITS_PER_SM - 1;

    cout << "How many iterations: " << iterationsForMSBs << endl;

    for(int i = 0; i < iterationsForMSBs; i++)
    {
        int startingQubit = MAX_QUBITS_PER_SM * (i+1);

        //MSB_nQubit_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector, startingQubit);

        coalesced_MSB_nQubit_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector, startingQubit, m);
        coalesced_MSB_nQubit_kernel<<<blockNumber, THREAD_PER_BLOCK>>>(deviceStateVector, startingQubit + m, m);
    }

    CHKERR( cudaPeekAtLastError() );

    CHKERR( cudaMemcpy(hostStateVector, deviceStateVector, stateVectorSize, cudaMemcpyDeviceToHost) );

    CHKERR( cudaEventRecord( stop, 0 ) );
    
	CHKERR( cudaEventSynchronize( stop ) );
	CHKERR( cudaEventElapsedTime( &mainStreamElapsedTime, start, stop ) );
	CHKERR( cudaEventDestroy( start ) );
	CHKERR( cudaEventDestroy( stop ) );

    CHKERR( cudaFree(deviceStateVector) );

    //printStateVector(hostStateVector, statesNumber);
    //printQubitsState(hostStateVector, NUM_QUBITS);

    cout << "Simulation elapsed time: " << mainStreamElapsedTime <<  " ms." << endl;

    free(hostStateVector);
}