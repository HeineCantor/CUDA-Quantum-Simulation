#ifndef __PRINT_UTIL_H__
#define __PRINT_UTIL_H__

#include <iostream>
#include <cuComplex.h>

using namespace std;

void printSimulationDetails(int numQubits, int threadNumber, int blockNumber)
{
    cout << "Qubit number: " << numQubits << endl;
    cout << "States number: " << (1 << numQubits) << endl;

    cout << endl;

    cout << "======= SIMULATION FOR ONE QUBIT GATE =======" << endl;
    cout << "Required Thread number: " << threadNumber << endl;
    cout << "Required Blocks number: " << blockNumber << endl;

    cout << endl;
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

#endif