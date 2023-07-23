#ifndef __PRINT_UTIL_H__
#define __PRINT_UTIL_H__

#include <iostream>
#include <cuComplex.h>
#include <bitset>

using namespace std;

void printGenericSimulationDetails(int numQubits);
void printSingleQubitSimulationDetails(int numQubits, int threadNumber, int blockNumber);
void printNQubitsSimulationDetails(int numQubits, int blockNumber, bool sharedMemoryOptimization, bool coalescingOptimization);

void printStateVector(cuDoubleComplex* vector, int vectorCount, int maxStatesToPrint = 0);
void printQubitsState(cuDoubleComplex* vector, int qubitCount, int maxStatesToPrint = 0);

#endif