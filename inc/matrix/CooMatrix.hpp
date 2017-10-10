#ifndef _COOMATRIX_HPP_
#define _COOMATRIX_HPP_

#include <iostream>
#include <cstdio>
#include <fstream>

template <typename ValueT, typename OffsetT>
struct CooTuple{
	OffsetT		row;
	OffsetT		col;
	ValueT		val;

	CooTuple() {}
	CooTuple(const OffsetT& row, const OffsetT& col) : row(row), col(col) {}
	CooTuple(const OffsetT& row, const OffsetT& col, const ValueT& val) : row(row), col(col), val(val) {}
	
	bool operator<(const CooTuple<ValueT, OffsetT>& other) const
	{
		return ((row < other.row) || (row == other.row && col < other.col));
	}
};

/*template <typename ValueT, typename OffsetT>
bool operator<(const CooTuple<ValueT, OffsetT>& a, const CooTuple<ValueT, OffsetT>& b)
{
	return ((a.row < b.row) || (a.row == b.row && a.col < b.col));
}*/

template <typename ValueT, typename OffsetT>
struct CooMatrix{
	OffsetT		num_rows;
	OffsetT		num_cols;
	OffsetT		num_nonzeros;
	CooTuple<ValueT, OffsetT>	*coo_tuples;

	CooMatrix() : num_rows(0), num_cols(0), num_nonzeros(0), coo_tuples(NULL) {}
	~CooMatrix()
	{
		Clear();
	}
	
	void Clear();
	void Display();
	void InitByFile(const std::string& file_name, const ValueT& default_value = 1.0, bool verbose = false);
};

template <typename ValueT, typename OffsetT>
void CooMatrix<ValueT, OffsetT>::Display()
{
	std::cout << "COO Matrix (" << num_rows << " rows, " << num_cols << " columns, " << num_nonzeros << " non-zeros);" << std::endl;
	std::cout << "Ordinal, Row, Column, Value" << std::endl;
	for(OffsetT i = 0; i < num_nonzeros; i++)
	{
		std::cout << '\t' << i << ',' << coo_tuples[i].row << ',' << coo_tuples[i].col << ',' << coo_tuples[i].val << std::endl;
	}
}

template <typename ValueT, typename OffsetT>
void CooMatrix<ValueT, OffsetT>::Clear()
{
	if(coo_tuples)
	{
		delete [] coo_tuples;
		coo_tuples = nullptr;
	}
}

template <typename ValueT, typename OffsetT>
void CooMatrix<ValueT, OffsetT>::InitByFile(const std::string& file_name, const ValueT& default_value, bool verbose)
{
	if(verbose) 
	{
		printf("Reading... ");
		fflush(stdout);
	}
	
	if(coo_tuples)
	{
		fprintf(stderr, "Matrix already constructed\n");
		exit(1);
	}
	
	std::ifstream ifs;
	ifs.open(file_name.c_str(), std::ifstream::in);
	if(!ifs.good())
	{
		fprintf(stderr, "Error opening file\n");
		exit(1);
	}
	
	bool    array = false;
	bool    symmetric = false;
	bool    skew = false;
	OffsetT     current_nz = -1;
	std::string line;
	
	if (verbose) 
	{
		printf("Parsing... "); fflush(stdout);
	}
	
	while(getline(ifs, line))
	{
		if(line[0] == '%')
		{
			if(line[1] == '%')
			{
				symmetric = (line.find("symmetric") != std::string::npos);
				skew = (line.find("skew") != std::string::npos);
				array = (line.find("array") != std::string::npos);
				
				if(verbose) 
				{
					printf("(symmetric: %d, skew: %d, array: %d)", symmetric, skew, array);
					fflush(stdout);
				}
			}
		}
		else if(current_nz == -1)
		{
			OffsetT nparsed = sscanf(line.c_str(), "%d %d %d", &num_rows, &num_cols, &num_nonzeros);
			if(!array && nparsed == 3)
			{
				if(symmetric)
					num_nonzeros *= 2;
				
				coo_tuples = new CooTuple<ValueT, OffsetT>[num_nonzeros];
				current_nz = 0;
			}
			else if(array && nparsed == 2)
			{
				num_nonzeros = num_rows * num_cols;
				coo_tuples = new CooTuple<ValueT, OffsetT>[num_nonzeros];
				current_nz = 0;
			}
			else
			{
				fprintf(stderr, "Error parsing MARKET matrix: invalid problem description: %s\n", line.c_str());
				exit(1);
			}
		}
		else
		{
			if(current_nz >= num_nonzeros)
			{
				fprintf(stderr, "Error parsing MARKET matrix: encountered more than %d num_nonzeros\n", num_nonzeros);
				exit(1);
			}
			
			OffsetT row, col;
			double val;
			if(array)
			{
				if(sscanf(line.c_str(), "%lf", &val) != 1)
				{
					fprintf(stderr, "Error parsing MARKET matrix: badly formed current_nz: '%s' at edge %d\n", line.c_str(), current_nz);
					exit(1);
				}
				col = (current_nz / num_rows);
				row = (current_nz - (num_rows * col));
				
				coo_tuples[current_nz] = CooTuple<ValueT, OffsetT>(row, col, val);
			}
			else
			{
				const char *l = line.c_str();
				char *t = NULL;
				row = strtol(l, &t, 0);
				if(t == l)
				{
					fprintf(stderr, "Error parsing MARKET matrix: badly formed row at edge %d\n", current_nz);
					exit(1);
				}
				l = t;
				
				col = strtol(l, &t, 0);
				if(t == l)
				{
					fprintf(stderr, "Error parsing MARKET matrix: badly formed row at edge %d\n", current_nz);
					exit(1);
				}
				l = t;
				
				val = strtod(l, &t);
				if(t == l)
				{
					val = default_value;
				}
				
				coo_tuples[current_nz] = CooTuple<ValueT, OffsetT>(row - 1, col - 1, val);
			}
			
			current_nz++;
			
			if(symmetric && (row != col))
			{
				coo_tuples[current_nz].row = coo_tuples[current_nz - 1].col;
				coo_tuples[current_nz].col = coo_tuples[current_nz - 1].row;
				coo_tuples[current_nz].val = coo_tuples[current_nz - 1].val * (skew ? -1 : 1);
				current_nz++;
			}
		}
	}
	
	num_nonzeros = current_nz;
	
	if (verbose) 
	{
		printf("done. "); fflush(stdout);
	}
	
	ifs.close();
}
#endif

