#include "../include/single_gate_simulation.cuh"
#include "../include/nQubit_gate_simulation.cuh"

int main()
{
    float totalElapsedTime = 0;

    float elapsedTime;
    int meanIterations = 1000;

    for(int i = 0; i < meanIterations; i++)
    {
        //singleGateSimulation(19, elapsedTime);
        //nQubitGateSimulation(19, elapsedTime, 0, 0);
        //nQubitGateSimulation(26, elapsedTime, 9, 0);
        nQubitGateSimulation(19, elapsedTime, 7, 4);

        totalElapsedTime += elapsedTime;
    }

    totalElapsedTime /= meanIterations;

    cout << endl;

    cout << "MEAN ELAPSED TIME OVER " << meanIterations << " ITERATIONS: " << totalElapsedTime << "ms" << endl;
    
    return 0;
}