#include "utils.cuh"

__host__ void CudaKernelErrorCheck(const char *prefix, const char *postfix)
{
	if(cudaPeekAtLastError() != cudaSuccess)
	{
		printf("\n%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
		cudaDeviceReset();
		exit(1);
	}
}
