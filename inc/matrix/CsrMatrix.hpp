#ifndef _CSRMATRIX_HPP_
#define _CSRMATRIX_HPP_

#include "CooMatrix.hpp"
#include "GraphStats.h"
#include <iostream>
#include <algorithm>

template <typename ValueT, typename OffsetT>
struct CsrMatrix{
	OffsetT	num_rows;
	OffsetT num_cols;
	OffsetT num_nonzeros;
	OffsetT *row_offsets;
	OffsetT *column_indices;
	ValueT *values;
	
	CsrMatrix(CooMatrix<ValueT, OffsetT>& coo_matrix, bool verbose = false)
	{
		Init(coo_matrix, verbose);
	}
	~CsrMatrix()
	{
		Clear();
	}
	
	void Init(CooMatrix<ValueT, OffsetT> &coo_matrix, bool verbose = false);
	void Clear();
	void DisplayHistogram();
	void Display();
	GraphStats Stats();
};

template <typename ValueT, typename OffsetT>
void CsrMatrix<ValueT, OffsetT>::Init(CooMatrix<ValueT, OffsetT> &coo_matrix, bool verbose)
{
	num_rows = coo_matrix.num_rows;
	num_cols = coo_matrix.num_cols;
	num_nonzeros = coo_matrix.num_nonzeros;
	
	if(verbose)
	{
		std::cout << "Ordering..." << std::endl;
	}
	std::stable_sort(coo_matrix.coo_tuples, coo_matrix.coo_tuples + num_nonzeros);
	if(verbose)
	{
		std::cout << "done." << std::endl;
	}
	
	row_offsets = new OffsetT[num_rows + 1];
	column_indices = new OffsetT[num_nonzeros];
	values = new ValueT[num_nonzeros];
	
	OffsetT prev_row = -1;
	for(OffsetT current_nz = 0; current_nz < num_nonzeros; current_nz++)
	{
		OffsetT current_row = coo_matrix.coo_tuples[current_nz].row;

		// Fill in rows up to and including the current row
		for (OffsetT row = prev_row + 1; row <= current_row; row++)
		{
			row_offsets[row] = current_nz;
		}
		prev_row = current_row;

		column_indices[current_nz]    = coo_matrix.coo_tuples[current_nz].col;
		values[current_nz]            = coo_matrix.coo_tuples[current_nz].val;
	}
	
	for (OffsetT row = prev_row + 1; row <= num_rows; row++)
	{
		row_offsets[row] = num_nonzeros;
	}
}

template <typename ValueT, typename OffsetT>
void CsrMatrix<ValueT, OffsetT>::Clear()
{
	if (row_offsets)    delete[] row_offsets;
	row_offsets = nullptr;
	if (column_indices) delete[] column_indices;
	column_indices = nullptr;
	if (values)         delete[] values;
	values = nullptr;
}

template <typename ValueT, typename OffsetT>
GraphStats CsrMatrix<ValueT, OffsetT>::Stats()
{
	GraphStats stats;
	stats.num_rows = num_rows;
	stats.num_cols = num_cols;
	stats.num_nonzeros = num_nonzeros;
	
	//
	// Compute diag-distance statistics
	//
	
	OffsetT samples     = 0;
	double  mean        = 0.0;
	double  ss_tot      = 0.0;

	for (OffsetT row = 0; row < num_rows; ++row)
	{
		OffsetT nz_idx_start    = row_offsets[row];
		OffsetT nz_idx_end      = row_offsets[row + 1];

		for (int nz_idx = nz_idx_start; nz_idx < nz_idx_end; ++nz_idx)
		{
			OffsetT col             = column_indices[nz_idx];
			double x                = (col > row) ? col - row : row - col;

			samples++;
			double delta            = x - mean;
			mean                    = mean + (delta / samples);
			ss_tot                  += delta * (x - mean);
		}
	}
	
	//
	// Compute deming statistics
	//

	samples         = 0;
	double mean_x   = 0.0;
	double mean_y   = 0.0;
	double ss_x     = 0.0;
	double ss_y     = 0.0;

	for (OffsetT row = 0; row < num_rows; ++row)
	{
		OffsetT nz_idx_start    = row_offsets[row];
		OffsetT nz_idx_end      = row_offsets[row + 1];

		for (int nz_idx = nz_idx_start; nz_idx < nz_idx_end; ++nz_idx)
		{
			OffsetT col             = column_indices[nz_idx];

			samples++;
			double x                = col;
			double y                = row;
			double delta;

			delta                   = x - mean_x;
			mean_x                  = mean_x + (delta / samples);
			ss_x                    += delta * (x - mean_x);

			delta                   = y - mean_y;
			mean_y                  = mean_y + (delta / samples);
			ss_y                    += delta * (y - mean_y);
		}
	}

	samples         = 0;
	double s_xy     = 0.0;
	double s_xxy    = 0.0;
	double s_xyy    = 0.0;
	for (OffsetT row = 0; row < num_rows; ++row)
	{
		OffsetT nz_idx_start    = row_offsets[row];
		OffsetT nz_idx_end      = row_offsets[row + 1];

		for (int nz_idx = nz_idx_start; nz_idx < nz_idx_end; ++nz_idx)
		{
			OffsetT col             = column_indices[nz_idx];

			samples++;
			double x                = col;
			double y                = row;

			double xy =             (x - mean_x) * (y - mean_y);
			double xxy =            (x - mean_x) * (x - mean_x) * (y - mean_y);
			double xyy =            (x - mean_x) * (y - mean_y) * (y - mean_y);
			double delta;

			delta                   = xy - s_xy;
			s_xy                    = s_xy + (delta / samples);

			delta                   = xxy - s_xxy;
			s_xxy                   = s_xxy + (delta / samples);

			delta                   = xyy - s_xyy;
			s_xyy                   = s_xyy + (delta / samples);
		}
	}

	double s_xx     = ss_x / num_nonzeros;
	double s_yy     = ss_y / num_nonzeros;

	double deming_slope = (s_yy - s_xx + sqrt(((s_yy - s_xx) * (s_yy - s_xx)) + (4 * s_xy * s_xy))) / (2 * s_xy);

	stats.pearson_r = (num_nonzeros * s_xy) / (sqrt(ss_x) * sqrt(ss_y));
	
	//
	// Compute row-length statistics
	//

	// Sample mean
	stats.row_length_mean       = double(num_nonzeros) / num_rows;
	double variance             = 0.0;
	stats.row_length_skewness   = 0.0;
	for (OffsetT row = 0; row < num_rows; ++row)
	{
		OffsetT length              = row_offsets[row + 1] - row_offsets[row];
		double delta                = double(length) - stats.row_length_mean;
		variance   += (delta * delta);
		stats.row_length_skewness   += (delta * delta * delta);
	}
	variance                    /= num_rows;
	stats.row_length_std_dev    = sqrt(variance);
	stats.row_length_skewness   = (stats.row_length_skewness / num_rows) / pow(stats.row_length_std_dev, 3.0);
	stats.row_length_variation  = stats.row_length_std_dev / stats.row_length_mean;

	return stats;
}


/**
 * Display log-histogram to stdout
 */
template <typename ValueT, typename OffsetT>
void CsrMatrix<ValueT, OffsetT>::DisplayHistogram()
{
	OffsetT log_counts[9];
	for(OffsetT i = 0; i != 9; i++)
	{
		log_counts[i] = 0;
	}
	
	//Scan
	OffsetT max_log_length = -1;
	OffsetT max_length = -1;
	for(OffsetT row = 0; row < num_rows; row++)
	{
		OffsetT length = row_offsets[row + 1] - row_offsets[row];
		if(length > max_length)
		{
			max_length = length;
		}
		
		OffsetT log_length = -1;
		while(length > 0)
		{
			length /= 10;
			log_length++;
		}
		
		if(log_length > max_log_length)
		{
			max_log_length = log_length;
		}
		
		log_counts[log_length + 1]++;
	}
	printf("CSR matrix (%d rows, %d columns, %d non-zeros, max-length %d):\n", (int) num_rows, (int) num_cols, (int) num_nonzeros, (int) max_length);
	for (OffsetT i = -1; i < max_log_length + 1; i++)
	{
		printf("\tDegree 1e%d: \t%d (%.2f%%)\n", i, log_counts[i + 1], (float) log_counts[i + 1] * 100.0 / num_cols);
	}
	fflush(stdout);
}

template <typename ValueT, typename OffsetT>
void CsrMatrix<ValueT, OffsetT>::Display()
{
	printf("Input Matrix (%d vertices, %d nonzeros):\n", (int) num_rows, (int) num_nonzeros);
	for (OffsetT row = 0; row < num_rows; row++)
	{
		printf("%d [@%d, #%d]: ", row, row_offsets[row], row_offsets[row + 1] - row_offsets[row]);
		for (OffsetT col_offset = row_offsets[row]; col_offset < row_offsets[row + 1]; col_offset++)
		{
			printf("%d (%f), ", column_indices[col_offset], values[col_offset]);
		}
		printf("\n");
	}
	fflush(stdout);
}

#endif
