#include "../include/single_gate_simulation.cuh"
#include "../include/nQubit_gate_simulation.cuh"

int main()
{
    singleGateSimulation(24);
    nQubitGateSimulation(24, true, true);

    return 0;
}