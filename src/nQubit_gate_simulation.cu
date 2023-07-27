#include "../include/nQubit_gate_simulation.cuh"

__global__ void LSB_nQubit_kernel(cuDoubleComplex* stateVector, int howManyQubits)
{
    int threadIndex = threadIdx.x;
    int kIndex = blockIdx.x * twoToThePower(howManyQubits);    // blockIndex -> k coefficient

    for(int i = 0; i < howManyQubits; i++)
    {
        __syncthreads();

        if(threadIndex < twoToThePower(howManyQubits - 1))
        {
            int xorOffset = (1 << i); //2^qubit_index

            int iCoeff = kIndex + threadIndex + (threadIndex / xorOffset) * xorOffset;
            int iXORCoeff = iCoeff ^ xorOffset;

            cuDoubleComplex coefficients[2] = {stateVector[iCoeff], stateVector[iXORCoeff]};

            gates::gate_hadamard(coefficients);

            stateVector[iCoeff] = coefficients[0];
            stateVector[iXORCoeff] = coefficients[1];
        }
    }
}

__global__ void MSB_nQubit_kernel(cuDoubleComplex* stateVector, int startingQubit, int howManyQubits)
{
    int threadIndex = threadIdx.x;
    int twoToTheQ = twoToThePower(startingQubit);

    int kIndex = (blockIdx.x % twoToTheQ) + (blockIdx.x / twoToTheQ) * twoToTheQ * twoToThePower(howManyQubits);    // blockIndex = k coefficient

    for(int i = 0; i < howManyQubits; i++)
    {
        __syncthreads();

        if(threadIndex < twoToThePower(howManyQubits - 1))
        {
            int xorOffset = (1 << i); //2^qubit_index

            int iCoeff = threadIndex + (threadIndex / xorOffset) * xorOffset;
            int iXORCoeff = iCoeff ^ xorOffset;

            iCoeff = kIndex ^ (twoToTheQ * iCoeff);
            iXORCoeff = kIndex ^ (twoToTheQ * iXORCoeff);

            cuDoubleComplex coefficients[2] = {stateVector[iCoeff], stateVector[iXORCoeff]};

            gates::gate_hadamard(coefficients);

            stateVector[iCoeff] = coefficients[0];
            stateVector[iXORCoeff] = coefficients[1];
        }
    }
}

__global__ void LSB_nQubit_kernel_shared(cuDoubleComplex* stateVector, int howManyQubits)
{
    extern __shared__ cuDoubleComplex subCoefficients[];

    int threadIndex = threadIdx.x;
    int kIndex = blockIdx.x * twoToThePower(howManyQubits);    // blockIndex -> k coefficient

    if(threadIndex < twoToThePower(howManyQubits))
        subCoefficients[threadIndex] = stateVector[kIndex ^ threadIndex];

    for(int i = 0; i < howManyQubits; i++)
    {
        __syncthreads();

        if(threadIndex < twoToThePower(howManyQubits - 1))
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

    if(threadIndex < twoToThePower(howManyQubits))
        stateVector[kIndex ^ threadIndex] = subCoefficients[threadIndex];
}

__global__ void MSB_nQubit_kernel_shared(cuDoubleComplex* stateVector, int startingQubit, int howManyQubits)
{
    extern __shared__ cuDoubleComplex subCoefficients[];

    int maxCoefficientsPerSM = twoToThePower(howManyQubits);
    int twoToTheQ = twoToThePower(startingQubit);

    int threadIndex = threadIdx.x;
    int kIndex = (blockIdx.x % twoToTheQ) + (blockIdx.x / twoToTheQ) * twoToTheQ * twoToThePower(howManyQubits);

    if(threadIndex < twoToThePower(howManyQubits))
        subCoefficients[threadIndex] = stateVector[kIndex ^ (twoToTheQ * threadIndex)];
 
    for(int i = 0; i < howManyQubits; i++)
    {
        __syncthreads();

        if(threadIndex < twoToThePower(howManyQubits - 1))
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

    if(threadIndex < twoToThePower(howManyQubits))
        stateVector[kIndex ^ (twoToTheQ * threadIndex)] = subCoefficients[threadIndex];
}

__global__ void coalesced_MSB_nQubit_kernel_shared(cuDoubleComplex* stateVector, int startingQubit, int m, int howManyQubits)
{
    extern __shared__ cuDoubleComplex subCoefficients[];

    int M = twoToThePower(m);

    int threadIndex = threadIdx.x;
    int blockIndex = twoToThePower(howManyQubits) / M * blockIdx.x;

    int twoToTheQ = twoToThePower(startingQubit);

    int kIndex = (blockIndex % twoToTheQ) + (blockIndex / twoToTheQ) * twoToTheQ * M;

    if(threadIndex < twoToThePower(howManyQubits))
        subCoefficients[threadIndex] = stateVector[(kIndex + threadIndex / M) ^ twoToTheQ * (threadIndex % M)];
 
    for(int i = 0; i < m; i++)
    {
        __syncthreads();

        if(threadIndex < twoToThePower(howManyQubits - 1))
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

    if(threadIndex < twoToThePower(howManyQubits))
        stateVector[(kIndex + threadIndex / M) ^ twoToTheQ * (threadIndex % M)] = subCoefficients[threadIndex];
}

void nQubitGateSimulation(int numQubits, float &mainStreamElapsedTime, int sharedMemoryOpt, int coalescingOpt)
{
    int statesNumber = twoToThePower(numQubits);
    unsigned long int stateVectorSize = sizeof(cuDoubleComplex) * statesNumber;

    int howManyQubits;

    bool sharedMemoryEnabled = sharedMemoryOpt > 0;
    bool coalescingOptimizationEnabled = coalescingOpt > 0;

    int m = coalescingOpt;
    int iterationsForMSBs;

    int blockNumber = sharedMemoryOpt ? twoToThePower(numQubits - MAX_QUBITS_PER_SM) : twoToThePower(numQubits / 2);
    int threadsPerBlock;

    printNQubitsSimulationDetails(numQubits, blockNumber, sharedMemoryEnabled, coalescingOptimizationEnabled);

    cuDoubleComplex unitaryComplex;
    unitaryComplex.x = 1;
    unitaryComplex.y = 0;

    cudaEvent_t start, stop;

    CHKERR( cudaEventCreate(&start) );
    CHKERR( cudaEventCreate(&stop) );

    cuDoubleComplex* hostStateVector = new cuDoubleComplex[stateVectorSize];

    cuDoubleComplex* deviceStateVector = NULL;

    CHKERR( cudaEventRecord( start, 0 ) );

    CHKERR( cudaMalloc((void**)& deviceStateVector, stateVectorSize) );

    // Initializing the state vector with the state |000...0>, a.k.a. the state vector [ 1 0 0 ... 0 ]
    CHKERR( cudaMemset(deviceStateVector, 0, stateVectorSize) );
    CHKERR( cudaMemcpy(deviceStateVector, &unitaryComplex, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) ); 


    if(!sharedMemoryEnabled)
    {
        int halfQubits = numQubits / 2;
        howManyQubits = MAX_QUBITS_PER_BLOCK;
        threadsPerBlock = twoToThePower(halfQubits);

        if(threadsPerBlock > MAX_THREADS_PER_BLOCK)
            threadsPerBlock = MAX_THREADS_PER_BLOCK;

        iterationsForMSBs = (numQubits + MAX_QUBITS_PER_BLOCK - 1) / MAX_QUBITS_PER_BLOCK - 1;
        howManyQubits = numQubits / (iterationsForMSBs + 1) + 1;

        blockNumber = twoToThePower(numQubits - howManyQubits);

        LSB_nQubit_kernel<<<blockNumber, threadsPerBlock>>>(deviceStateVector, howManyQubits);

        CHKERR( cudaPeekAtLastError() );

        for(int i = 0; i < iterationsForMSBs; i++)
        {
            int startingQubit = howManyQubits * (i+1);

            if(i == iterationsForMSBs - 1) // if last iteration
            {
                blockNumber = twoToThePower(startingQubit);
                howManyQubits = numQubits - startingQubit;
            }

            MSB_nQubit_kernel<<<blockNumber, threadsPerBlock>>>(deviceStateVector, startingQubit, howManyQubits);

            CHKERR( cudaPeekAtLastError() );
        }
    }
    else
    {
        iterationsForMSBs = (numQubits + sharedMemoryOpt - 1) / sharedMemoryOpt - 1;
        howManyQubits = numQubits / (iterationsForMSBs + 1);

        blockNumber = twoToThePower(numQubits - howManyQubits);
        threadsPerBlock = twoToThePower(howManyQubits);

        //cout <<  "HOW MANY ITERATIONS " << iterationsForMSBs << " FOR QUBITS " << howManyQubits << endl;

        // LSB Kernel Call
        LSB_nQubit_kernel_shared<<<blockNumber, threadsPerBlock, sizeof(cuDoubleComplex) * twoToThePower(howManyQubits)>>>(deviceStateVector, howManyQubits);

        CHKERR( cudaPeekAtLastError() );

        // MSB Kernel Call

        for(int i = 0; i < iterationsForMSBs; i++)
        {
            int startingQubit = howManyQubits * (i+1);

            if(!coalescingOptimizationEnabled)
            {
                
                if(i == iterationsForMSBs - 1) // if last iteration
                {
                    blockNumber = twoToThePower(startingQubit);
                    howManyQubits = numQubits - startingQubit;
                    threadsPerBlock = twoToThePower(howManyQubits);
                }

                MSB_nQubit_kernel_shared<<<blockNumber, threadsPerBlock, sizeof(cuDoubleComplex) * twoToThePower(howManyQubits)>>>(deviceStateVector, startingQubit, howManyQubits);
                CHKERR( cudaPeekAtLastError() );
            }
            else
            {
                if(i == iterationsForMSBs - 1) // if last iteration
                {
                    howManyQubits = numQubits - startingQubit;
                    
                    m = howManyQubits / 2;
                    blockNumber = twoToThePower(numQubits - howManyQubits);
                    threadsPerBlock = twoToThePower(howManyQubits);
                }

                coalesced_MSB_nQubit_kernel_shared<<<blockNumber, threadsPerBlock, sizeof(cuDoubleComplex) * twoToThePower(howManyQubits)>>>(deviceStateVector, startingQubit, m, howManyQubits);
                CHKERR( cudaPeekAtLastError() );

                coalesced_MSB_nQubit_kernel_shared<<<blockNumber, threadsPerBlock, sizeof(cuDoubleComplex) * twoToThePower(howManyQubits)>>>(deviceStateVector, startingQubit + m, howManyQubits - m, howManyQubits);
                CHKERR( cudaPeekAtLastError() );
            }
        }
    }

    CHKERR( cudaPeekAtLastError() );

    CHKERR( cudaMemcpy(hostStateVector, deviceStateVector, stateVectorSize, cudaMemcpyDeviceToHost) );

    CHKERR( cudaEventRecord( stop, 0 ) );
    
	CHKERR( cudaEventSynchronize( stop ) );
	CHKERR( cudaEventElapsedTime( &mainStreamElapsedTime, start, stop ) );
	CHKERR( cudaEventDestroy( start ) );
	CHKERR( cudaEventDestroy( stop ) );

    CHKERR( cudaFree(deviceStateVector) );

    printStateVector(hostStateVector, statesNumber, 10);
    //printQubitsState(hostStateVector, NUM_QUBITS);

    cout << "Simulation elapsed time: " << mainStreamElapsedTime <<  " ms." << endl;

    free(hostStateVector);
}