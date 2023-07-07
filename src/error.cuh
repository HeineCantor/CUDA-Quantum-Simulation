#ifndef __ERROR_CUH__
#define __ERROR_CUH__

#include <iostream>

#define CHKERR(x) { checkCudaError((x), __FILE__, __LINE__, 1); }

inline void checkCudaError(cudaError_t cd, const char *fl, int ln, int ab)
{
	if (cd != cudaSuccess)
    {
		  fprintf(stderr,"Error! - File: %s Line: %d Details: %s.\n", fl, ln, cudaGetErrorString(cd));
		  if (ab) 
            exit(cd);
	}
}

#endif