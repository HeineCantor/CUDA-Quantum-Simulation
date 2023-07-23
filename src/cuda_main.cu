#include "../include/single_gate_simulation.cuh"
#include "../include/nQubit_gate_simulation.cuh"

int main()
{
    singleGateSimulation(26);
    nQubitGateSimulation(26, 10, 0);
    

    return 0;
}