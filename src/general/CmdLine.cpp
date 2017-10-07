#include "CmdLine.h"
#include "utils.cuh"
#include <cstdio>

using namespace std;

CmdLine::CmdLine(int argc, char **argv) : keys(10), values(10)
{
	for (int i = 1; i < argc; i++)
	{
		string arg = argv[i];
		if ((arg[0] != '-') || (arg[1]) != '-')
		{
			args.push_back(arg);
			continue;
		}

		string::size_type pos;
		string key, val;
		if ((pos = arg.find('=')) == string::npos)
		{
			key = string(arg, 2, arg.length() - 2);
			val = "";
		}
		else
		{
			key = string(arg, 2, pos - 2);
			val = string(arg, pos + 1, arg.length() - 1);
		}

		keys.push_back(key);
		values.push_back(val);
	}
}

bool CmdLine::CheckCmdLineFlag(const char* arg_name)
{
	for (int i = 0; i < static_cast<int>(keys.size()); i++)
	{
		if (keys[i] == string(arg_name))
		{
			return true;
		}
	}
	return false;
}

void CmdLine::DeviceInit(int dev)
{
	int deviceCount;
	CudaErrorCheck(cudaGetDeviceCount(&deviceCount));
	
	if(deviceCount == 0)
	{
		fprintf(stderr, "No devices supporting CUDA.\n");
		exit(1);
	}
	
	if(dev < 0)
	{
		GetCmdLineArgument("device", dev);
	}
	if ((dev > deviceCount - 1) || (dev < 0))
	{
		dev = 0;
	}
	CudaErrorCheck(cudaSetDevice(dev));
	
	CudaErrorCheck(cudaMemGetInfo(&device_free_physmem, &device_total_physmem));
	
	CudaErrorCheck(cudaGetDeviceProperties(&deviceProp, dev));
	
	if (deviceProp.major < 1) {
		fprintf(stderr, "Device does not support CUDA.\n");
		exit(1);
	}
	
	device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;
	
	if (!CheckCmdLineFlag("quiet"))
	{
		printf(
			"Using device %d: %s (SM%d, %d SMs, "
			"%lld free / %lld total MB physmem, "
			"%.3f GB/s @ %d kHz mem clock, ECC %s)\n",
			dev,
			deviceProp.name,
			deviceProp.major * 100 + deviceProp.minor * 10,
			deviceProp.multiProcessorCount,
			(unsigned long long) device_free_physmem / 1024 / 1024,
			(unsigned long long) device_total_physmem / 1024 / 1024,
			device_giga_bandwidth,
			deviceProp.memoryClockRate,
			(deviceProp.ECCEnabled) ? "on" : "off");
		fflush(stdout);
	}
}
