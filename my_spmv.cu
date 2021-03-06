#include "CmdLine.h"
#include "utils.cuh"
#include "CooMatrix.hpp"
#include "CsrMatrix.hpp"
#include "test.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

bool                    g_quiet     = false;        // Whether to display stats in CSV format
bool                    g_verbose   = false;        // Whether to display output to console
bool                    g_verbose2  = false;        // Whether to display input to console

template<typename ValueT, typename OffsetT>
void SaveMatrix(const CooMatrix<ValueT, OffsetT>& coo_matrix, const char *filename)
{
	std::ofstream out(filename);
	if(!out)
	{
		std::cout << "error: unable to open output file:"
				  << filename << std::endl;
		exit(1);
	}
	
	for(int i = 0; i != coo_matrix.num_nonzeros; i++)
	{
		out << coo_matrix.coo_tuples[i].row << " " << coo_matrix.coo_tuples[i].col << " " << coo_matrix.coo_tuples[i].val << std::endl;
	}
	
	out.close();
}

template<typename ValueT>
void SaveOutput(const ValueT *output, const int& size, const char *filename)
{
	std::ofstream out(filename);
	if(!out)
	{
		std::cout << "error: unable to open output file:"
				  << filename << std::endl;
		exit(1);
	}
	
	for(int i = 0; i != size; i++)
	{
		out << output[i] << std::endl;
	}
	
	out.close();
}

template <typename ValueT, typename OffsetT>
void RunTest(ValueT aplpha, ValueT beta, CooMatrix<ValueT, OffsetT>& coo_matrix, 
			 int timing_iterations, CmdLine& cmdline)
{
    // Adaptive timing iterations: run 16 billion nonzeros through
    if (timing_iterations == -1)
        timing_iterations = std::min(50000ull, std::max(100ull, ((16ull << 30) / coo_matrix.num_nonzeros)));
	
	if (!g_quiet)
        printf("\t%d timing iterations\n", timing_iterations);
	
	CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
	if(!cmdline.CheckCmdLineFlag("csrmv"))
	{
		coo_matrix.Clear();
	}
	
	csr_matrix.Stats().Display(!g_quiet);
	
	if(!g_quiet)
	{
		printf("\n");
		csr_matrix.DisplayHistogram();
		printf("\n");
		if(g_verbose2)
		{
			csr_matrix.Display();
		}
		printf("\n");	
	}
	fflush(stdout);
	
	ValueT *vector_x = new ValueT[csr_matrix.num_cols];
	ValueT *vector_y_in = new ValueT[csr_matrix.num_rows];
	ValueT *vector_y_out = new ValueT[csr_matrix.num_rows];
	
	for (int col = 0; col < csr_matrix.num_cols; ++col)
        vector_x[col] = 1.0;

    for (int row = 0; row < csr_matrix.num_rows; ++row)
        vector_y_in[row] = 1.0;
	
	SpmvVerification(csr_matrix, vector_x, vector_y_in, vector_y_out, aplpha, beta);
	
	//free memory
	delete [] vector_x;
	vector_x = nullptr;
	delete [] vector_y_in;
	vector_y_in = nullptr;
	delete [] vector_y_out;
	vector_y_out = nullptr;
}

template<typename ValueT, typename OffsetT>
void RunTests(ValueT alpha, ValueT beta, const std::string& mtx_filename, int timing_iterations, CmdLine& cmdline)
{
	CooMatrix<ValueT, OffsetT> coo_matrix;
	
	//Get COO matrix from dataset
	if(!mtx_filename.empty())
	{
		coo_matrix.InitByFile(mtx_filename, 1.0, !g_quiet);
		
		if ((coo_matrix.num_rows == 1) || (coo_matrix.num_cols == 1) || (coo_matrix.num_nonzeros == 1))
        {
            if (!g_quiet) std::cout << "Trivial dataset" << std::endl;
            exit(0);
        }
        std::cout << mtx_filename << std::endl;
	}
	else 
	{
		fprintf(stderr, "No graph type specified.\n");
        exit(1);
	}
	
	//Run tests
	RunTest(alpha, beta, coo_matrix, timing_iterations, cmdline);
}

int main(int argc, char **argv)
{
	CmdLine cmd(argc, argv);
	if (cmd.CheckCmdLineFlag("help"))
	{
		std::cout << "usage: " << argv[0] << " [option] [option] ..."
				  << "\nOption:"
				  << "\n[--csrmv | --hybmv | --bsrmv ]"
				  << "\n[--device=<device-id>]"
				  << "\n[--quiet]"
				  << "\n[--v]"
				  << "\n[--i=<timing iterations>]"
				  << "\n[--fp32]"
				  << "\n[--alpha=<alpha scalar (default: 1.0)>]"
				  << "\n[--beta=<beta scalar (default: 0.0)>]"
				  << "\n[--mtx=<matrix market file>]"
				  << "\n[--help]"
				  << std::endl;
		exit(0);
	}
	
	bool fp32;
	std::string	matrix_filename;
/*	int                 grid2d              = -1;RunTest
    int                 grid3d              = -1;
    int                 wheel               = -1;
    int                 dense               = -1;*/
    int                 timing_iterations   = -1;
    float               alpha               = 1.0;
    float               beta                = 0.0;
	
	g_verbose = cmd.CheckCmdLineFlag("v");
    g_verbose2 = cmd.CheckCmdLineFlag("v2");
    g_quiet = cmd.CheckCmdLineFlag("quiet");
    fp32 = cmd.CheckCmdLineFlag("fp32");
	cmd.GetCmdLineArgument("i", timing_iterations);
    cmd.GetCmdLineArgument("mtx", matrix_filename);
    cmd.GetCmdLineArgument("alpha", alpha);
    cmd.GetCmdLineArgument("beta", beta);
	
	cmd.DeviceInit();
	
	if(fp32)
	{
		RunTests<float, int>(alpha, beta, matrix_filename, timing_iterations, cmd);
	}
	else
	{
		RunTests<double, int>(alpha, beta, matrix_filename, timing_iterations, cmd);
	}
	
	CudaErrorCheck(cudaDeviceSynchronize())
	printf("\n");
	
	return 0;
}
