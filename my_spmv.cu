#include "CmdLine.h"
#include <stdio.h>
#include <cuda_runtime.h>


int main(int argc, char **argv)
{
	CmdLine cmd(argc, argv);
	if (cmd.CheckCmdLineFlag("help"))
	{
		printf(
			"%s "
            "[--csrmv | --hybmv | --bsrmv ] "
            "[--device=<device-id>] "
            "[--quiet] "
            "[--v] "
            "[--i=<timing iterations>] "
            "[--fp32] "
            "[--alpha=<alpha scalar (default: 1.0)>] "
            "[--beta=<beta scalar (default: 0.0)>] "
            "\n\t"
                "--mtx=<matrix market file> "
            "\n\t"
                "--dense=<cols>"
            "\n\t"
                "--grid2d=<width>"
            "\n\t"
                "--grid3d=<width>"
            "\n\t"
                "--wheel=<spokes>"
            "\n", argv[0]);
		exit(0);
	}
	
	bool fp32;
	std::string	matrix_filename;
	int                 grid2d              = -1;
    int                 grid3d              = -1;
    int                 wheel               = -1;
    int                 dense               = -1;
    int                 timing_iterations   = -1;
    float               alpha               = 1.0;
    float               beta                = 0.0;
	
	cmd.GetCmdLineArgument("i", timing_iterations);
    cmd.GetCmdLineArgument("mtx", matrix_filename);
    cmd.GetCmdLineArgument("grid2d", grid2d);
    cmd.GetCmdLineArgument("grid3d", grid3d);
    cmd.GetCmdLineArgument("wheel", wheel);
    cmd.GetCmdLineArgument("dense", dense);
    cmd.GetCmdLineArgument("alpha", alpha);
    cmd.GetCmdLineArgument("beta", beta);
	
	cmd.DeviceInit();
}
