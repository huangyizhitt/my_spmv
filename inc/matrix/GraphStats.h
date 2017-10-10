struct GraphStats
{
    int         num_rows;
    int         num_cols;
    int         num_nonzeros;

    double      pearson_r;              // coefficient of variation x vs y (how linear the sparsity plot is)

    double      row_length_mean;        // mean
    double      row_length_std_dev;     // sample std_dev
    double      row_length_variation;   // coefficient of variation
    double      row_length_skewness;    // skewness

	void Display(bool show_labels = true);
};
