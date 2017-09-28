#include "args.h"

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
		if ((pos = arg.find('=')) == string::nops)
		{
			key = string(arg, 2, arg.length() - 2);
			val = "";
		}
		else
		{
			key = string(arg, 2, pos - 2);
			val = string(arg, pos + 1, arg.length() - 1);
		}

		key.push_back(key);
		values.push_back(values);
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