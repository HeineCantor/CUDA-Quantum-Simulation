#ifndef __VALIDATION_CUH__
#define __VALIDATION_CUH__

#include <cuComplex.h>
#include <math.h>
#include <quantum.h>

bool naiveHadamardValidate(cuDoubleComplex* stateVector, int qubitsNumber);

#endif