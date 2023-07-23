#include "../include/nQubit_gate_simulation.cuh"

__global__ void LSB_nQubit_kernel(cuDoubleComplex* stateVector, int halfQubits)
{
    int threadIndex = threadIdx.x;
    int kIndex = blockIdx.x * twoToThePower(halfQubits);    // blockIndex -> k coefficient

    for(int i = 0; i < halfQubits; i++)
    {
        __syncthreads();

        if(threadIndex < twoToThePower(halfQubits - 1))
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
    int kIndex = blockIdx.x;    // blockIndex = k coefficient

    int twoToTheQ = twoToThePower(startingQubit);

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

__global__ void LSB_nQubit_kernel_shared(cuDoubleComplex* stateVector)
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

__global__ void MSB_nQubit_kernel_shared(cuDoubleComplex* stateVector, int startingQubit)
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

__global__ void coalesced_MSB_nQubit_kernel_shared(cuDoubleComplex* stateVector, int startingQubit, int m)
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

void nQubitGateSimulation(int numQubits, bool sharedMemoryOpt, bool coalescingOpt)
{
    int statesNumber = twoToThePower(numQubits);
    int stateVectorSize = sizeof(cuDoubleComplex) * statesNumber;

    int m = COALESCING_PARTITION;
    int iterationsForMSBs;

    int blockNumber = sharedMemoryOpt ? twoToThePower(numQubits - MAX_QUBITS_PER_SM) : twoToThePower(numQubits / 2);
    int threadsPerBlock;

    printNQubitsSimulationDetails(numQubits, blockNumber, sharedMemoryOpt, coalescingOpt);

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


    if(!sharedMemoryOpt)
    {
        int halfQubits = numQubits / 2;
        threadsPerBlock = twoToThePower(halfQubits - 1);

        if(threadsPerBlock > MAX_THREADS_PER_BLOCK) // In this case, you'll have to make more MSB kernel calls
        {
            iterationsForMSBs = (numQubits + MAX_QUBITS_PER_BLOCK - 1) / MAX_QUBITS_PER_BLOCK - 1;
            threadsPerBlock = MAX_THREADS_PER_BLOCK;
            blockNumber = numQubits - MAX_QUBITS_PER_BLOCK;

            cout << "THREADS PER BLOCK: " << threadsPerBlock << endl;

            LSB_nQubit_kernel<<<blockNumber, threadsPerBlock>>>(deviceStateVector, MAX_QUBITS_PER_BLOCK);

            CHKERR( cudaPeekAtLastError() );

            for(int i = 0; i < iterationsForMSBs; i++)
            {
                int startingQubit = MAX_QUBITS_PER_BLOCK * (i+1);
                int howManyQubits = MAX_QUBITS_PER_BLOCK;

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
            // LSB Kernel Call
            LSB_nQubit_kernel<<<blockNumber, threadsPerBlock>>>(deviceStateVector, halfQubits);

            CHKERR( cudaPeekAtLastError() );

            // MSB Kernel Call
            MSB_nQubit_kernel<<<blockNumber, threadsPerBlock>>>(deviceStateVector, halfQubits, halfQubits);

            CHKERR( cudaPeekAtLastError() );
        }
    }
    else
    {
        threadsPerBlock = twoToThePower(MAX_QUBITS_PER_SM);

        // LSB Kernel Call
        LSB_nQubit_kernel_shared<<<blockNumber, threadsPerBlock>>>(deviceStateVector);

        CHKERR( cudaPeekAtLastError() );

        // MSB Kernel Call

        iterationsForMSBs = (numQubits + MAX_QUBITS_PER_SM - 1) / MAX_QUBITS_PER_SM - 1;

        for(int i = 0; i < iterationsForMSBs; i++)
        {
            int startingQubit = MAX_QUBITS_PER_SM * (i+1);

            if(!coalescingOpt)
                MSB_nQubit_kernel_shared<<<blockNumber, threadsPerBlock>>>(deviceStateVector, startingQubit);
            else
            {
                coalesced_MSB_nQubit_kernel_shared<<<blockNumber, threadsPerBlock>>>(deviceStateVector, startingQubit, m);
                CHKERR( cudaPeekAtLastError() );
                
                coalesced_MSB_nQubit_kernel_shared<<<blockNumber, threadsPerBlock>>>(deviceStateVector, startingQubit + m, m);
            }

            CHKERR( cudaPeekAtLastError() );
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