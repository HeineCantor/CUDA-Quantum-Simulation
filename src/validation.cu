#include "../include/validation.cuh"
#include <iostream>

bool naiveHadamardValidate(cuDoubleComplex* stateVector, int qubitsNumber)
{
    double tolerance = 2e-18;

    int coeffNumber = 1 << qubitsNumber;
    double hadamardNumber = std::pow(0.7071067811865475, qubitsNumber);

    for(int i = 0; i < coeffNumber; i++)
    {
        if(std::abs(stateVector[i].x - hadamardNumber) > tolerance)
            return false;
    }

    return true;
}