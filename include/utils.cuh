#ifndef __UTILS_H__
#define __UTILS_H__

__host__ __device__ inline int twoToThePower(int exp)
{
    return (1 << exp);
}

#endif