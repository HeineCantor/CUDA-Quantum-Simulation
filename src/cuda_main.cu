#include "../include/single_gate_simulation.cuh"
#include "../include/nQubit_gate_simulation.cuh"

int main()
{
    singleGateSimulation();
    nQubitGateSimulation(26, false, false);

    return 0;
}