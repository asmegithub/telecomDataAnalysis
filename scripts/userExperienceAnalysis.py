import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Utility Functions
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df, columns_to_fill):
    for column in columns_to_fill:
        df[column].fillna(df[column].mean(), inplace=True)
    df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)
    return df

def replace_outliers_with_mean(column, df):
    mean = df[column].mean()
    std_dev = df[column].std()
    outlier_threshold = 3 * std_dev
    df[column] = np.where(
        np.abs(df[column] - mean) > outlier_threshold, mean, df[column]
    )
    return df

def handle_outliers(df, columns):
    for column in columns:
        replace_outliers_with_mean(column, df)
    return df

def aggregate_per_customer(df):
    aggregated_data = df.groupby('IMSI').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'Handset Type': lambda x: x.mode()[0]  # Most frequent handset type
    }).reset_index()

    aggregated_data.rename(columns={
        'TCP DL Retrans. Vol (Bytes)': 'Avg_TCP_DL_Retransmissions',
        'TCP UL Retrans. Vol (Bytes)': 'Avg_TCP_UL_Retransmissions',
        'Avg RTT DL (ms)': 'Avg_RTT_DL',
        'Avg RTT UL (ms)': 'Avg_RTT_UL',
        'Avg Bearer TP DL (kbps)': 'Avg_Throughput_DL',
        'Avg Bearer TP UL (kbps)': 'Avg_Throughput_UL',
        'Handset Type': 'Most_Frequent_Handset_Type'
    }, inplace=True)

    return aggregated_data

def compute_top_bottom_frequent(df, column):
    return {
        "top_10": df[column].nlargest(10),
        "bottom_10": df[column].nsmallest(10),
        "most_frequent": df[column].value_counts().head(10)
    }

def analyze_distribution(aggregated_data):
    throughput_distribution = aggregated_data.groupby('Most_Frequent_Handset_Type')[['Avg_Throughput_DL', 'Avg_Throughput_UL']].mean()
    tcp_retransmission_distribution = aggregated_data.groupby('Most_Frequent_Handset_Type')[['Avg_TCP_DL_Retransmissions', 'Avg_TCP_UL_Retransmissions']].mean()
    return throughput_distribution, tcp_retransmission_distribution

def plot_distribution(distribution, title, xlabel, ylabel):
    plt.figure(figsize=(14, 7))
    distribution.plot(kind='bar', figsize=(14, 7), title=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def perform_kmeans_clustering(aggregated_data, n_clusters):
    features = aggregated_data[['Avg_TCP_DL_Retransmissions', 'Avg_TCP_UL_Retransmissions', 'Avg_RTT_DL', 'Avg_RTT_UL', 'Avg_Throughput_DL', 'Avg_Throughput_UL']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    aggregated_data['Cluster'] = kmeans.fit_predict(scaled_features)

    cluster_analysis = aggregated_data.groupby('Cluster')[['Avg_TCP_DL_Retransmissions', 'Avg_TCP_UL_Retransmissions', 'Avg_RTT_DL', 'Avg_RTT_UL', 'Avg_Throughput_DL', 'Avg_Throughput_UL']].mean()
    return aggregated_data, cluster_analysis

def plot_clusters(aggregated_data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=aggregated_data['Avg_Throughput_DL'],
        y=aggregated_data['Avg_RTT_DL'],
        hue=aggregated_data['Cluster'],
        palette="viridis",
        style=aggregated_data['Cluster'],
        s=100
    )
    plt.title("Clusters Based on Throughput and RTT")
    plt.xlabel("Average Throughput DL (kbps)")
    plt.ylabel("Average RTT DL (ms)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()
