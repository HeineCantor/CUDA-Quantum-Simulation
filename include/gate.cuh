#ifndef __GATE_CUH__
#define __GATE_CUH__

#include <cuComplex.h>
#include "error.cuh"

namespace gates
{
    __device__ inline void gate_x(cuDoubleComplex* inputCoefficients)
    {
        cuDoubleComplex outputCoefficients[2];

        outputCoefficients[0] = inputCoefficients[1];
        outputCoefficients[1] = inputCoefficients[0];

        inputCoefficients[0] = outputCoefficients[0];
        inputCoefficients[1] = outputCoefficients[1];
    }

    __device__ inline void gate_z(cuDoubleComplex* inputCoefficients)
    {
        cuDoubleComplex outputCoefficients[2];

        cuDoubleComplex minusOne;
        minusOne.x = -1;
        minusOne.y = 0;

        outputCoefficients[0] = inputCoefficients[0];
        outputCoefficients[1] = cuCmul(minusOne, inputCoefficients[1]);

        inputCoefficients[0] = outputCoefficients[0];
        inputCoefficients[1] = outputCoefficients[1];
    }

    __device__ inline void gate_hadamard(cuDoubleComplex* inputCoefficients)
    {
        cuDoubleComplex outputCoefficients[2];

        cuDoubleComplex sqrt2;
        sqrt2.x = 0.7071067811865475;
        sqrt2.y = 0;

        cuDoubleComplex minusSqrt2;
        minusSqrt2.x = -0.7071067811865475;
        minusSqrt2.y = 0;

        outputCoefficients[0] = cuCadd(cuCmul(sqrt2, inputCoefficients[0]), cuCmul(sqrt2, inputCoefficients[1]));
        outputCoefficients[1] = cuCadd(cuCmul(sqrt2, inputCoefficients[0]), cuCmul(minusSqrt2, inputCoefficients[1]));

        inputCoefficients[0] = outputCoefficients[0];
        inputCoefficients[1] = outputCoefficients[1];
    }
}

#endif