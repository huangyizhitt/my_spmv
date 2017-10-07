#ifndef _UTILS_CUH_
#define _UTILS_CUH_

#include <cuda_runtime.h>
#include <cstdio>

#define CudaErrorCheck(x)	{															\
		if(x != cudaSuccess) {															\
			printf("\nCUDA Error: %s (err_num = %d)\n", cudaGetErrorString(x), x);		\
			cudaDeviceReset();															\
			exit(1);																	\
		}																				\
}					

__host__ void CudaKernelErrorCheck(const char *prefix, const char *postfix);

#endif
