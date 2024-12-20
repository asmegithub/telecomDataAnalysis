import seaborn as sns
import matplotlib.pyplot as plt

def univariate_analysis(data, column, plot_type='hist', bins=30):
    """
    Perform univariate analysis for a specific column in the dataset.

    Args:
        data (DataFrame): The dataset to analyze.
        column (str): The column name for analysis.
        plot_type (str): Type of plot ('hist', 'box', 'bar').
        bins (int): Number of bins for histogram (if applicable).
    """
    plt.figure(figsize=(8, 6))
    if plot_type == 'hist':
        sns.histplot(data[column], bins=bins, kde=True)
        plt.title(f'Histogram of {column}')
    elif plot_type == 'box':
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
    elif plot_type == 'bar':
        data[column].value_counts().plot(kind='bar')
        plt.title(f'Barplot of {column}')
    else:
        print(f"Plot type {plot_type} not supported.")
        return
    plt.xlabel(column)
    plt.show()

# Example usage:
# univariate_analysis(df, 'Dur. (s)', plot_type='hist')
# univariate_analysis(df, 'UserID', plot_type='bar')

def bivariate_analysis(data, column_x, column_y, plot_type='scatter'):
    """
    Perform bivariate analysis for two columns in the dataset.

    Args:
        data (DataFrame): The dataset to analyze.
        column_x (str): The first column name (x-axis).
        column_y (str): The second column name (y-axis).
        plot_type (str): Type of plot ('scatter', 'correlation').
    """
    plt.figure(figsize=(8, 6))
    if plot_type == 'scatter':
        sns.scatterplot(x=column_x, y=column_y, data=data)
        plt.title(f'{column_x} vs {column_y}')
    elif plot_type == 'correlation':
        corr_matrix = data[[column_x, column_y]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True)
        plt.title('Correlation Matrix')
    else:
        print(f"Plot type {plot_type} not supported.")
        return
    plt.show()

# Example usage:
# bivariate_analysis(df, 'Dur. (s)', 'Total Data (Bytes)', plot_type='scatter')
# bivariate_analysis(df, 'Total UL (Bytes)', 'Total DL (Bytes)', plot_type='correlation')

def summary_statistics(data, columns):
    """
    Generate summary statistics for selected columns.

    Args:
        data (DataFrame): The dataset to analyze.
        columns (list): List of column names for which to generate statistics.
    """
    stats = data[columns].describe()
    print("Summary Statistics:")
    print(stats)
    return stats

# Example usage:
# stats = summary_statistics(df, ['Dur. (s)', 'Total UL (Bytes)', 'Total DL (Bytes)'])
