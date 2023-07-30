#include "../include/single_gate_simulation.cuh"
#include "../include/nQubit_gate_simulation.cuh"

int main()
{
    float totalElapsedTime = 0;

    float elapsedTime;
    int meanIterations = 100;

    for(int i = 0; i < meanIterations; i++)
    {
        //singleGateSimulation(27, elapsedTime);
        //nQubitGateSimulation(27, elapsedTime, 0, 0);
        //nQubitGateSimulation(27, elapsedTime, 9, 0);
        nQubitGateSimulation(27, elapsedTime, 9, 4);

        totalElapsedTime += elapsedTime;
    }

    totalElapsedTime /= meanIterations;

    cout << endl;

    cout << "MEAN ELAPSED TIME OVER " << meanIterations << " ITERATIONS: " << totalElapsedTime << "ms" << endl;
    
    return 0;
}