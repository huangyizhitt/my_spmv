#ifndef _ARGS_H_
#define _ARGS_H_

#include <map>
#include <string>
#include <vector>
#include <iostream>

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

	cudaError_t DeviceInit(const int& dev = -1);
};
