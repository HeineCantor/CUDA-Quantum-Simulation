#include "../include/single_gate_simulation.cuh"
#include "../include/nQubit_gate_simulation.cuh"

int main()
{
    singleGateSimulation(20);
    nQubitGateSimulation(21, true, false);

    return 0;
}