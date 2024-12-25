import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean

# Function to load the dataset and handle missing values
def load_and_clean_data_from_db(db_connection, query):
    # Execute the query and fetch data into a pandas DataFrame
    df = db_connection.fetch_data(query)
    
    # Handle missing values in the dataset
    columns_to_fill = [
        'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
        'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 
        'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'
    ]
    for column in columns_to_fill:
        df[column].fillna(df[column].mean(), inplace=True)
    df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)
    return df

# Function to aggregate the data by customer IMSI
def aggregate_data(df):
    return df.groupby('IMSI').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean'
    }).reset_index()

# Function to perform K-Means clustering and assign clusters
def perform_kmeans_clustering(data, n_clusters):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    return clusters, kmeans

# Function to calculate engagement and experience scores
def calculate_scores(aggregated_data, experience_cluster_means, engagement_cluster_centers, less_engaged_cluster, worst_experience_cluster):
    # Make sure to match the columns
    columns_of_interest = [
        'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
        'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 
        'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'
    ]
    
    # Calculate Engagement_Score
    aggregated_data['Engagement_Score'] = aggregated_data.apply(
        lambda row: euclidean(row[columns_of_interest], engagement_cluster_centers[less_engaged_cluster]), axis=1
    )

    # Calculate Experience_Score
    aggregated_data['Experience_Score'] = aggregated_data.apply(
        lambda row: euclidean(row[columns_of_interest], experience_cluster_means.loc[worst_experience_cluster, columns_of_interest].values), axis=1
    )
    
    return aggregated_data


# Function to compute the satisfaction score
def compute_satisfaction_score(aggregated_data):
    aggregated_data['Satisfaction_Score'] = aggregated_data[['Engagement_Score', 'Experience_Score']].mean(axis=1)
    return aggregated_data

# Function to build a regression model
def build_regression_model(aggregated_data):
    X = aggregated_data[['Engagement_Score', 'Experience_Score']]
    y = aggregated_data['Satisfaction_Score']
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    return reg_model

# Function to perform K-Means clustering on satisfaction scores
def cluster_satisfaction_scores(aggregated_data):
    kmeans = KMeans(n_clusters=2, random_state=42)
    aggregated_data['Satisfaction_Cluster'] = kmeans.fit_predict(aggregated_data[['Engagement_Score', 'Experience_Score']])
    return aggregated_data, kmeans

# Function to aggregate average scores per satisfaction cluster
def aggregate_cluster_summary(aggregated_data):
    cluster_summary = aggregated_data.groupby('Satisfaction_Cluster')[['Engagement_Score', 'Experience_Score', 'Satisfaction_Score']].mean()
    return cluster_summary



def export_to_postgresql(merged_df, table_name, db_config):
        """
        Exports the final table containing user ID, engagement, experience, and satisfaction scores to PostgreSQL database.

        Parameters:
        - merged_df (pd.DataFrame): DataFrame containing the final table data.
        - table_name (str): Name of the table in the PostgreSQL database.
        - db_config (dict): Dictionary containing database configuration details like user, password, host, port, database.

        Returns:
        - None
        """
        try:
            # Define the SQLAlchemy engine for PostgreSQL
            engine = create_engine(
                f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            # Export the DataFrame to the PostgreSQL table
            merged_df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
            print(f"Data exported successfully to the table '{table_name}' in the PostgreSQL database.")
            
        except Exception as e:
            print("Error occurred while exporting data to PostgreSQL:", e)

# Function to verify data export
def verify_export(engine):
    with engine.connect() as connection:
        result = connection.execute("SELECT * FROM user_satisfaction LIMIT 10;")
        print("Exported data preview:")
        for row in result:
            print(row)

#