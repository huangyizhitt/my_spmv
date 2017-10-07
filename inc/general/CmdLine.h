#ifndef _CMDLINE_H_
#define _CMDLINE_H_

#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

struct CmdLine {
	float				device_giga_bandwidth;
	size_t				device_free_physmem;
	size_t				device_total_physmem;
	cudaDeviceProp		deviceProp;

	std::vector<std::string>	keys;
	std::vector<std::string>	values;
	std::vector<std::string>	args;

	CmdLine(int argc, char **argv);
	bool CheckCmdLineFlag(const char* arg_name);


	template <typename T>
	void GetCmdLineArgument(const char *arg_name, T& val) const;

	template <typename T>
	void GetCmdLineArgument(const char *arg_name, std::vector<T>& value) const;

	int GetNakedArgsNum() const
	{
		return static_cast<int>(args.size());
	}

	int GetArgc() const
	{
		return static_cast<int>(keys.size());
	}

	void DeviceInit(int dev = -1);
};

template <typename T>
void CmdLine::GetCmdLineArgument(const char *arg_name, T& val) const
{
	for(int i = 0; i < static_cast<int>(keys.size()); i++)
	{
		if(keys[i] == std::string(arg_name))
		{
			std::istringstream str_stream(values[i]);
			str_stream >> val;
		}
	}
}

#endif
