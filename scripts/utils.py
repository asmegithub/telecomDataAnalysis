import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# User Engagement Analysis
# User Engagement Analysis

# Bearer Id', 'Start', 'Start ms', 'End', 'End ms', 'Dur. (ms)',
    #    'IMSI', 'MSISDN/Number', 'IMEI', 'Last Location Name',
    #    'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)',
    #    'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)',
    #    'TCP UL Retrans. Vol (Bytes)', 'DL TP < 50 Kbps (%)',
    #    '50 Kbps < DL TP < 250 Kbps (%)', '250 Kbps < DL TP < 1 Mbps (%)',
    #    'DL TP > 1 Mbps (%)', 'UL TP < 10 Kbps (%)',
    #    '10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)',
    #    'UL TP > 300 Kbps (%)', 'HTTP DL (Bytes)', 'HTTP UL (Bytes)',
    #    'Activity Duration DL (ms)', 'Activity Duration UL (ms)',
    #    'Dur. (ms).1', 'Handset Manufacturer', 'Handset Type',
    #    'Nb of sec with 125000B < Vol DL',
    #    'Nb of sec with 1250B < Vol UL < 6250B',
    #    'Nb of sec with 31250B < Vol DL < 125000B',
    #    'Nb of sec with 37500B < Vol UL',
    #    'Nb of sec with 6250B < Vol DL < 31250B',
    #    'Nb of sec with 6250B < Vol UL < 37500B',
    #    'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B',
    #    'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
    #    'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
    #    'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
    #    'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
    #    'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)',
    #    'Total UL (Bytes)', 'Total DL (Bytes)'], dtype=object)





def segment_users(df):
    """
    Segment users into deciles and compute total data usage.
    """
    # Updated column names for total duration
    duration_columns = [
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)',
        'Other DL (Bytes)'
    ]
    if not all(col in df.columns for col in duration_columns):
        raise KeyError("One or more duration columns are missing from the dataset.")

    # Compute total duration
    df['Total Duration'] = df[duration_columns].sum(axis=1)
    
    # Segment into deciles
    df['Decile Class'] = pd.qcut(df['Total Duration'], 10, labels=False) + 1
    
    # Compute total data (Download + Upload)
    df['Total Data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    
    # Aggregate total data per decile
    decile_data = df.groupby('Decile Class')['Total Data'].sum().reset_index()
    
    return df, decile_data

def compute_basic_metrics(df):
    """
    Compute and display basic metrics like mean, median, and standard deviation.
    """
    mean = df.mean(numeric_only=True)
    median = df.median(numeric_only=True)
    std_dev = df.std(numeric_only=True)
    
    print("\nMean:\n", mean)
    print("\nMedian:\n", median)
    print("\nStandard Deviation:\n", std_dev)
    
    return mean, median, std_dev

def non_graphical_univariate_analysis(df):
    """
    Compute dispersion parameters for quantitative variables.
    """
    dispersion = df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    print("\nDispersion Parameters:\n", dispersion)
    return dispersion

def plot_univariate_graphs(df):
    """
    Create histograms for numeric variables with improved stability.
    """
    # Sample data to avoid memory issues
    sample_df = df.sample(n=10000) if len(df) > 10000 else df
    
    # Select numeric columns
    numeric_columns = sample_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        try:
            # Handle NaN values
            data = sample_df[col].dropna()
            if len(data) == 0:  # Skip empty columns
                continue
            
            # Plot distribution
            plt.figure(figsize=(8, 4))
            sns.histplot(data, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
        except Exception as e:
            print(f"Could not plot {col}: {e}")


def bivariate_analysis(df):
    """
    Explore relationships between application data and total data usage.
    """
    duration_columns = [
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)',
        'Other DL (Bytes)'
    ]
    if 'Total Data' not in df.columns:
        raise KeyError("The 'Total Data' column is missing. Ensure it is computed before running this function.")

    for app in duration_columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=app, y='Total Data', data=df)
        plt.title(f"{app} vs Total Data")
        plt.xlabel(app)
        plt.ylabel("Total Data")
        plt.show()

def correlation_analysis(df):
    """
    Compute and visualize correlation matrix for application data.
    """
    correlation_vars = [
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)',
        'Other DL (Bytes)'
    ]
    if not all(col in df.columns for col in correlation_vars):
        raise KeyError("One or more correlation variables are missing from the dataset.")

    correlation_matrix = df[correlation_vars].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
    return correlation_matrix

def perform_pca(df, n_components=3):
    """
    Perform PCA on selected variables and return results.

    Args:
        df (DataFrame): The input dataset.
        n_components (int): Number of PCA components.

    Returns:
        Tuple: PCA results and explained variance ratio.
    """
    # Correct column names
    pca_vars = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                'Google DL (Bytes)', 'Google UL (Bytes)',
                'Email DL (Bytes)', 'Email UL (Bytes)',
                'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
                'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
                'Other DL (Bytes)', 'Other UL (Bytes)']
    
    # Check if all required columns are in the dataset
    missing_columns = [col for col in pca_vars if col not in df.columns]
    if missing_columns:
        raise KeyError(f"The following required columns are missing: {missing_columns}")
    
    # Handle missing values by filling them with column means
    df_cleaned = df[pca_vars].fillna(df[pca_vars].mean())
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_cleaned)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_
    
    print("\nExplained Variance by PCA Components:\n", explained_variance)
    return pca_results, explained_variance


def visualize_pca(df, pca_results):
    """
    Add PCA results to dataset and visualize PCA scatterplot.

    Args:
        df (DataFrame): The input dataset.
        pca_results (ndarray): The PCA-transformed data.

    Returns:
        None
    """
    if pca_results.shape[1] < 2:
        raise ValueError("PCA results must have at least two components to visualize.")
    
    # Add PCA results to the dataset
    df['PCA1'] = pca_results[:, 0]
    df['PCA2'] = pca_results[:, 1]
    
    # Plot PCA scatterplot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', data=df)
    plt.title("PCA1 vs PCA2")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.show()

class EngagementAnalyzer:
    def __init__(self, df):
        self.df = df

    def user_engagement(self,df):
       # Calculate session frequency for each user
        session_frequency = df.groupby('MSISDN/Number').size().reset_index(name='Session Frequency')

        # Calculate duration of the session (already provided as 'Dur. (ms)')
        # If we need total session duration per user, you can sum it up
        session_duration = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='Total Session Duration (ms)')

        # Calculate session total traffic
        session_traffic = df.groupby('MSISDN/Number').agg({
            'Total UL (Bytes)': 'sum',
            'Total DL (Bytes)': 'sum'
        }).reset_index()
        session_traffic.columns = ['MSISDN/Number', 'Total UL (Bytes)', 'Total DL (Bytes)']

        # Merge all metrics into a single DataFrame
        user_engagement = session_frequency.merge(session_duration, on='MSISDN/Number')
        user_engagement = user_engagement.merge(session_traffic, on='MSISDN/Number')

        # Display the final DataFrame with user engagement metrics
        return user_engagement
    
    def high_engagement_users(self,df):
        user_engagement=self.user_engagement(df)
        # Define high engagement threshold (e.g., top 10% of each metric)
        freq_threshold = user_engagement['Session Frequency'].quantile(0.9)
        duration_threshold = user_engagement['Total Session Duration (ms)'].quantile(0.9)
        traffic_threshold = user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1).quantile(0.9)

        # Filter high engagement users
        high_engagement_users = user_engagement[
            (user_engagement['Session Frequency'] >= freq_threshold) &
            (user_engagement['Total Session Duration (ms)'] >= duration_threshold) &
            ((user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)) >= traffic_threshold)
        ]
        return high_engagement_users
    
    def plot_user_engagement(self,df):
        user_engagement=self.user_engagement(df)
        # Define high engagement threshold (e.g., top 10% of each metric)
        freq_threshold = user_engagement['Session Frequency'].quantile(0.9)
        duration_threshold = user_engagement['Total Session Duration (ms)'].quantile(0.9)
        traffic_threshold = user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1).quantile(0.9)

        # Filter high engagement users
        high_engagement_users = user_engagement[
            (user_engagement['Session Frequency'] >= freq_threshold) &
            (user_engagement['Total Session Duration (ms)'] >= duration_threshold) &
            ((user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)) >= traffic_threshold)
        ]

        # Plot High Engagement Users

        # Set up the figure and axes
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # Plot High Engagement Users by Session Frequency
        sns.histplot(high_engagement_users['Session Frequency'], bins=50, kde=True, ax=axs[0], color='blue')
        axs[0].set_title('High Engagement Users - Session Frequency')
        axs[0].set_xlabel('Session Frequency')
        axs[0].set_ylabel('Number of High Engagement Users')

        # Plot High Engagement Users by Session Duration
        sns.histplot(high_engagement_users['Total Session Duration (ms)'], bins=50, kde=True, ax=axs[1], color='green')
        axs[1].set_title('High Engagement Users - Total Session Duration')
        axs[1].set_xlabel('Total Session Duration (ms)')
        axs[1].set_ylabel('Number of High Engagement Users')

        # Plot High Engagement Users by Total Traffic
        high_engagement_users['Total Traffic (Bytes)'] = high_engagement_users[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)
        sns.histplot(high_engagement_users['Total Traffic (Bytes)'], bins=50, kde=True, ax=axs[2], color='red')
        axs[2].set_title('High Engagement Users - Total Traffic')
        axs[2].set_xlabel('Total Traffic (Bytes)')
        axs[2].set_ylabel('Number of High Engagement Users')

        # Adjust layout.
        plt.tight_layout()
        plt.show()
    def top_10_users_per_metric(self, df):
        # Calculate the top 10 users based on the specified metric
        high_engagement_users=self.high_engagement_users(df)
        # Calculate Total Traffic using sum of UL and DL Bytes
        high_engagement_users['Total Traffic (Bytes)'] = high_engagement_users[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)
        top_10_users_freq = high_engagement_users.nlargest(10, 'Session Frequency')
        top_10_users_duration = high_engagement_users.nlargest(10, 'Total Session Duration (ms)')
        top_10_users_traffic = high_engagement_users.nlargest(10, 'Total Traffic (Bytes)')
        print("Top 10 Users by Session Frequency:\n", top_10_users_freq, "\n")
        print("Top 10 Users by Total Session Duration:\n", top_10_users_duration, "\n")
        print("Top 10 Users by Total Traffic:\n", top_10_users_traffic, "\n")
    
    
    def top_10_users(self,df):
        # Calculate the top 10 users based on the specified metric
        high_engagement_users=self.high_engagement_users(df)
        # Calculate Total Traffic using sum of UL and DL Bytes
        high_engagement_users['Total Traffic (Bytes)'] = high_engagement_users[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)
        top_10_users_freq = high_engagement_users.nlargest(10, 'Session Frequency')
        top_10_users_duration = high_engagement_users.nlargest(10, 'Total Session Duration (ms)')
        top_10_users_traffic = high_engagement_users.nlargest(10, 'Total Traffic (Bytes)')
        return top_10_users_freq, top_10_users_duration, top_10_users_traffic
    def aggregate_traffic_per_user(self, df, applications):
        """
        Aggregates the traffic per user for the specified applications.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the traffic data.
        applications (list of str): List of application columns to aggregate.

        Returns:
        pd.DataFrame: A DataFrame with aggregated traffic per user.
        """
        return df.groupby('MSISDN/Number')[applications].sum().reset_index()

    def calculate_total_traffic(self, app_engagement, applications):
        """
        Calculates the total traffic for each application (DL + UL) and adds it to the DataFrame.

        Parameters:
        app_engagement (pd.DataFrame): DataFrame with aggregated traffic per user.
        applications (list of str): List of application columns to calculate total traffic for.

        Returns:
        pd.DataFrame: Updated DataFrame with total traffic columns added.
        """
        for app in applications:
            total_col_name = app.replace(' (Bytes)', ' Total (Bytes)')
            app_engagement[total_col_name] = app_engagement[app]
        
        # Calculate total traffic for Social Media as an example
        app_engagement['Social Media Total (Bytes)'] = (
            app_engagement['Social Media DL Total (Bytes)'] + 
            app_engagement['Social Media UL Total (Bytes)']
        )
        
        return app_engagement
    def get_top_users(self, app_engagement, application, n=10):
        """
        Retrieves the top N users based on total traffic for a given application.

        Parameters:
        app_engagement (pd.DataFrame): DataFrame with total traffic per user.
        application (str): Application name for which to retrieve top users.
        n (int): Number of top users to retrieve (default is 10).

        Returns:
        pd.DataFrame: A DataFrame containing the top N users for the specified application.
        """
        column_name = f'{application} Total (Bytes)'
        return app_engagement.nlargest(n, column_name)


    def prepare_user_engagement_data(self,df):
        """
        Prepare user engagement data by aggregating and calculating necessary metrics.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing telecom data.
            
        Returns:
            pd.DataFrame: A DataFrame with aggregated user engagement metrics.
            pd.DataFrame: Top 10 users by session frequency.
            pd.DataFrame: Top 10 users by total duration.
            pd.DataFrame: Top 10 users by total traffic.
        """
        # Calculate total data volume per session
        df['Total Duration'] = df['Total UL (Bytes)'] + df['Total DL (Bytes)']
        
        # Aggregate data by MSISDN/Number
        user_engagement_df = df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',  # Number of sessions per user
            'Total Duration': 'sum',  # Total data volume of all sessions
            'Total UL (Bytes)': 'sum',  # Total upload bytes
            'Total DL (Bytes)': 'sum'  # Total download bytes
        }).reset_index()
        
        # Calculate the total traffic per user
        user_engagement_df['Total Traffic (Bytes)'] = user_engagement_df['Total UL (Bytes)'] + user_engagement_df['Total DL (Bytes)']
        
        # Rename columns for better understanding
        user_engagement_df.rename(columns={'Bearer Id': 'Session Frequency'}, inplace=True)
        
        # Find the top 10 customers per engagement metric
        top10_sessions = user_engagement_df.nlargest(10, 'Session Frequency')
        top10_duration = user_engagement_df.nlargest(10, 'Total Duration')
        top10_traffic = user_engagement_df.nlargest(10, 'Total Traffic (Bytes)')
        
        return user_engagement_df, top10_sessions, top10_duration, top10_traffic
    
    def apply_clustering(self, user_engagement_df, n_clusters=3):
        """
        Apply K-Means clustering to the user engagement data.
        
        Args:
            user_engagement_df (pd.DataFrame): The DataFrame containing user engagement metrics.
            n_clusters (int): The number of clusters for K-Means. Default is 3.
            
        Returns:
            pd.DataFrame: The DataFrame with an added 'Engagement Cluster' column.
        """
        # Selecting only the relevant columns for normalization
        metrics = ['Session Frequency', 'Total Duration', 'Total Traffic (Bytes)']
        
        # Normalize the selected metrics for clustering
        scaler = MinMaxScaler()
        user_engagement_df[metrics] = scaler.fit_transform(user_engagement_df[metrics])
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        user_engagement_df['Engagement Cluster'] = kmeans.fit_predict(user_engagement_df[metrics])
        
        return user_engagement_df
    def plot_top_applications(self, df, applications, top_n=3):
        """
        Summarize the total traffic for each application and plot the top N most used applications.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing traffic data for applications.
            applications (list): A list of column names representing different applications in the DataFrame.
            top_n (int): The number of top applications to plot. Default is 3.
        
        Returns:
            pd.Series: A Series containing the total traffic for each application.
        """
        # Sum up total traffic for each application
        app_usage = df[applications].sum().sort_values(ascending=False)
        
        # Plot the top N applications
        top_apps = app_usage.head(top_n)
        plt.figure(figsize=(12, 6))
        top_apps.plot(kind='bar', title=f'Top {top_n} Most Used Applications')
        plt.ylabel('Total Traffic (Bytes)')
        plt.xlabel('Application')
        plt.xticks(rotation=45, ha='right')
        plt.show()
        
        return app_usage
    
    def plot_elbow_curve(self, df, metrics, max_k=10):
        """
        Determine the optimal number of clusters using the Elbow Method and plot the elbow curve.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the data to cluster.
            metrics (list): A list of column names to use for clustering.
            max_k (int): The maximum number of clusters to test. Default is 10.
        
        Returns:
            list: A list of distortions (inertia) for each number of clusters.
        """
        distortions = []
        
        # Iterate over the range of cluster numbers
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df[metrics])
            distortions.append(kmeans.inertia_)
        
        # Plot the elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), distortions, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Distortion (Inertia)')
        plt.grid(True)
        plt.show()
        
        return distortions