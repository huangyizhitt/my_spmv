#include "GraphStats.h"
#include <cstdio>

void GraphStats::Display(bool show_labels)
{
	if (show_labels)
		printf("\n"
		"\t num_rows: %d\n"
		"\t num_cols: %d\n"
		"\t num_nonzeros: %d\n"
		"\t row_length_mean: %.5f\n"
		"\t row_length_std_dev: %.5f\n"
		"\t row_length_variation: %.5f\n"
		"\t row_length_skewness: %.5f\n",
		num_rows,
		num_cols,
		num_nonzeros,
		row_length_mean,
		row_length_std_dev,
		row_length_variation,
		row_length_skewness);
	else
		printf(
			"%d, "
			"%d, "
			"%d, "
			"%.5f, "
			"%.5f, "
			"%.5f, "
			"%.5f, ",
			num_rows,
			num_cols,
			num_nonzeros,
			row_length_mean,
			row_length_std_dev,
			row_length_variation,
			row_length_skewness);
}
