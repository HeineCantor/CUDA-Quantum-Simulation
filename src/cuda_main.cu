#include "../include/single_gate_simulation.cuh"
#include "../include/nQubit_gate_simulation.cuh"

int main()
{
    //singleGateSimulation(26);
    nQubitGateSimulation(26, 0, 0);
    nQubitGateSimulation(26, 9, 0);
    

    return 0;
}