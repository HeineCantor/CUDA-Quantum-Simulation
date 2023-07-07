#ifndef __PRINT_UTIL_H__
#define __PRINT_UTIL_H__

#include <iostream>
#include <cuComplex.h>
#include <bitset>

using namespace std;

void printGenericSimulationDetails(int numQubits)
{
    cout << "Qubit number: " << numQubits << endl;
    cout << "States number: " << (1 << numQubits) << endl;

    cout << endl;
}

void printSingleQubitSimulationDetails(int numQubits, int threadNumber, int blockNumber)
{
    printGenericSimulationDetails(numQubits);

    cout << "======= SIMULATION FOR ONE QUBIT GATE =======" << endl;
    cout << "Required Thread number: " << threadNumber << endl;
    cout << "Required Blocks number: " << blockNumber << endl;

    cout << endl;
}

void printNQubitsSimulationDetails(int numQubits)
{
    printGenericSimulationDetails(numQubits);


}

void printStateVector(cuDoubleComplex* vector, int vectorCount, int maxStatesToPrint = 0)
{
    if(maxStatesToPrint == 0)
        maxStatesToPrint = vectorCount;

    int statesToPrint = vectorCount < maxStatesToPrint ? vectorCount : maxStatesToPrint;

    cout << "Output State Vector: [ ";
    for(int i = 0; i < statesToPrint; i++)
    {
        cout << "(" << vector[i].x << " + ";
        cout << vector[i].y << "i)";

        if(i < vectorCount - 1)
            cout << ", ";
    }

    if(statesToPrint < vectorCount)
        cout << "...";

    cout << " ]" << endl;
}

void printQubitsState(cuDoubleComplex* vector, int qubitCount, int maxStatesToPrint = 0)
{
    int vectorCount = 1 << qubitCount;

    if(maxStatesToPrint == 0)
        maxStatesToPrint = vectorCount;

    int statesToPrint = vectorCount < maxStatesToPrint ? vectorCount : maxStatesToPrint;

    cout << "Output Qubits State:" << endl;

    for(int i = 0; i < statesToPrint; i++)
    {
        cout << "(" << vector[i].x << "+" << vector[i].y << "i) |" << bitset<8>(i) << ">" << endl;
    }

        if(statesToPrint < vectorCount)
        cout << "..." << endl;
}

#endif