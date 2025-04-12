import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# Create directories for outputs
os.makedirs('visualizations', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

print("E-commerce Customer Behavior Analysis")
print("====================================")

# Load the dataset
df = pd.read_csv('ecommerce_customer_behavior.csv')
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# Display basic information
print("\nDataset Overview:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert boolean column to proper boolean type if needed
if df['Discount Applied'].dtype == 'object':
    df['Discount Applied'] = df['Discount Applied'].map({'TRUE': True, 'FALSE': False})

# Save cleaned data
df.to_csv('data/cleaned_ecommerce_data.csv', index=False)
print("\nCleaned data saved to data/cleaned_ecommerce_data.csv")

# Basic Analysis
print("\n1. Analyzing customer spending patterns...")

# Gender-based analysis
gender_stats = df.groupby('Gender').agg({
    'Total Spend': ['mean', 'sum', 'count'],
    'Items Purchased': ['mean', 'sum'],
    'Average Rating': 'mean'
}).round(2)

print("\nSpending by Gender:")
print(gender_stats)

# Visualize gender-based spending
plt.figure(figsize=(12, 6))
sns.barplot(x='Gender', y='Total Spend', data=df)
plt.title('Average Spending by Gender')
plt.savefig('visualizations/spending_by_gender.png', dpi=300, bbox_inches='tight')
plt.close()

# Membership type analysis
membership_stats = df.groupby('Membership Type').agg({
    'Total Spend': ['mean', 'sum', 'count'],
    'Items Purchased': ['mean', 'sum'],
    'Average Rating': 'mean',
    'Days Since Last Purchase': 'mean'
}).round(2)

print("\nAnalysis by Membership Type:")
print(membership_stats)

# Visualize membership-based spending
plt.figure(figsize=(12, 6))
sns.barplot(x='Membership Type', y='Total Spend', data=df, order=['Gold', 'Silver', 'Bronze'])
plt.title('Average Spending by Membership Type')
plt.savefig('visualizations/spending_by_membership.png', dpi=300, bbox_inches='tight')
plt.close()

# City-based analysis
city_stats = df.groupby('City').agg({
    'Total Spend': ['mean', 'sum', 'count'],
    'Items Purchased': ['mean', 'sum'],
    'Average Rating': 'mean'
}).round(2)

print("\nAnalysis by City:")
print(city_stats)

# Visualize top cities by total spending
city_total_spend = df.groupby('City')['Total Spend'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(14, 7))
sns.barplot(x='City', y='Total Spend', data=city_total_spend.head(10))
plt.title('Top 10 Cities by Total Spending')
plt.xticks(rotation=45)
plt.savefig('visualizations/top_cities_by_spending.png', dpi=300, bbox_inches='tight')
plt.close()

# Age distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution of Customers')
plt.savefig('visualizations/age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Correlation between age and spending
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Total Spend', hue='Gender', data=df)
plt.title('Relationship Between Age and Total Spend')
plt.savefig('visualizations/age_vs_spending.png', dpi=300, bbox_inches='tight')
plt.close()

# Satisfaction level analysis
satisfaction_stats = df.groupby('Satisfaction Level').agg({
    'Total Spend': ['mean', 'sum', 'count'],
    'Items Purchased': 'mean',
    'Average Rating': 'mean',
    'Days Since Last Purchase': 'mean'
}).round(2)

print("\nAnalysis by Satisfaction Level:")
print(satisfaction_stats)

# Visualize satisfaction level distribution
plt.figure(figsize=(12, 6))
satisfaction_order = ['Satisfied', 'Neutral', 'Unsatisfied']
sns.countplot(x='Satisfaction Level', data=df, order=satisfaction_order)
plt.title('Distribution of Satisfaction Levels')
plt.savefig('visualizations/satisfaction_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualize average rating by satisfaction level
plt.figure(figsize=(12, 6))
sns.barplot(x='Satisfaction Level', y='Average Rating', data=df, order=satisfaction_order)
plt.title('Average Rating by Satisfaction Level')
plt.savefig('visualizations/rating_by_satisfaction.png', dpi=300, bbox_inches='tight')
plt.close()

# Discount analysis
discount_stats = df.groupby('Discount Applied').agg({
    'Total Spend': ['mean', 'sum', 'count'],
    'Items Purchased': 'mean',
    'Average Rating': 'mean'
}).round(2)

print("\nAnalysis by Discount Applied:")
print(discount_stats)

# Visualize impact of discount on spending
plt.figure(figsize=(12, 6))
sns.barplot(x='Discount Applied', y='Total Spend', data=df)
plt.title('Average Spending by Discount Status')
plt.savefig('visualizations/spending_by_discount.png', dpi=300, bbox_inches='tight')
plt.close()

# Correlation matrix
correlation_columns = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
correlation_matrix = df[correlation_columns].corr().round(2)

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Days since last purchase analysis
plt.figure(figsize=(12, 6))
sns.histplot(df['Days Since Last Purchase'], bins=20, kde=True)
plt.title('Distribution of Days Since Last Purchase')
plt.savefig('visualizations/days_since_purchase_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Relationship between days since last purchase and satisfaction
plt.figure(figsize=(12, 6))
sns.boxplot(x='Satisfaction Level', y='Days Since Last Purchase', data=df, order=satisfaction_order)
plt.title('Days Since Last Purchase by Satisfaction Level')
plt.savefig('visualizations/days_since_purchase_by_satisfaction.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n2. Performing customer segmentation...")

# RFM Analysis (Recency, Frequency, Monetary)
# For this dataset:
# Recency = Days Since Last Purchase
# Frequency = Items Purchased (as a proxy)
# Monetary = Total Spend

# Create RFM dataframe
rfm_df = df[['Customer ID', 'Days Since Last Purchase', 'Items Purchased', 'Total Spend']].copy()
rfm_df.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']

# Convert recency to a score (lower days = higher score)
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

# Convert scores to numeric
rfm_df['R_Score'] = rfm_df['R_Score'].astype(int)
rfm_df['F_Score'] = rfm_df['F_Score'].astype(int)
rfm_df['M_Score'] = rfm_df['M_Score'].astype(int)

# Calculate RFM Score
rfm_df['RFM_Score'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']

# Define RFM segments
def rfm_segment(row):
    if row['RFM_Score'] >= 13:
        return 'Champions'
    elif (row['R_Score'] >= 4) and (row['F_Score'] + row['M_Score'] >= 6):
        return 'Loyal Customers'
    elif (row['R_Score'] >= 3) and (row['F_Score'] + row['M_Score'] >= 5):
        return 'Potential Loyalists'
    elif (row['R_Score'] >= 4) and (row['F_Score'] + row['M_Score'] < 4):
        return 'New Customers'
    elif (row['R_Score'] >= 3) and (row['F_Score'] + row['M_Score'] <= 4):
        return 'Promising'
    elif (row['R_Score'] <= 2) and (row['F_Score'] + row['M_Score'] >= 8):
        return 'At Risk'
    elif (row['R_Score'] <= 2) and (row['F_Score'] + row['M_Score'] >= 5):
        return 'Needs Attention'
    elif (row['R_Score'] <= 1) and (row['F_Score'] + row['M_Score'] < 5):
        return 'Hibernating'
    else:
        return 'About to Sleep'

rfm_df['Segment'] = rfm_df.apply(rfm_segment, axis=1)

# Save RFM analysis
rfm_df.to_csv('data/rfm_segments.csv', index=False)
print("RFM segmentation completed and saved to data/rfm_segments.csv")

# Visualize RFM segments
segment_counts = rfm_df['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']

plt.figure(figsize=(14, 7))
sns.barplot(x='Segment', y='Count', data=segment_counts)
plt.title('Customer Segments Distribution')
plt.xticks(rotation=45)
plt.savefig('visualizations/rfm_segments_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Merge RFM segments back to original data
df_with_segments = df.merge(rfm_df[['Customer_ID', 'Segment']], left_on='Customer ID', right_on='Customer_ID')
df_with_segments.drop('Customer_ID', axis=1, inplace=True)

# Analyze spending by segment
segment_spending = df_with_segments.groupby('Segment').agg({
    'Total Spend': ['mean', 'sum', 'count'],
    'Items Purchased': 'mean',
    'Average Rating': 'mean',
    'Days Since Last Purchase': 'mean'
}).round(2)

print("\nAnalysis by RFM Segment:")
print(segment_spending)

# Visualize average spending by segment
segment_avg_spend = df_with_segments.groupby('Segment')['Total Spend'].mean().reset_index()
plt.figure(figsize=(14, 7))
sns.barplot(x='Segment', y='Total Spend', data=segment_avg_spend)
plt.title('Average Spending by Customer Segment')
plt.xticks(rotation=45)
plt.savefig('visualizations/spending_by_segment.png', dpi=300, bbox_inches='tight')
plt.close()

# K-means clustering
print("\n3. Performing K-means clustering...")

# Select features for clustering
features = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
X = df[features].copy()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using silhouette score
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"Silhouette score for k={k}: {score:.4f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k Values')
plt.grid(True)
plt.savefig('visualizations/kmeans_silhouette_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# Get optimal k
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")

# Apply K-means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_stats = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Total Spend': 'mean',
    'Items Purchased': 'mean',
    'Average Rating': 'mean',
    'Days Since Last Purchase': 'mean',
    'Customer ID': 'count'
}).round(2)
cluster_stats.columns = ['Avg_Age', 'Avg_Spend', 'Avg_Items', 'Avg_Rating', 'Avg_Days_Since_Purchase', 'Customer_Count']

print("\nCluster Statistics:")
print(cluster_stats)

# Save cluster data
df.to_csv('data/customer_clusters.csv', index=False)
print("K-means clustering completed and saved to data/customer_clusters.csv")

# Visualize clusters (2D projections)
# Age vs Total Spend
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Age', y='Total Spend', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Clusters: Age vs Total Spend')
plt.savefig('visualizations/clusters_age_vs_spend.png', dpi=300, bbox_inches='tight')
plt.close()

# Items Purchased vs Average Rating
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Items Purchased', y='Average Rating', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Clusters: Items Purchased vs Average Rating')
plt.savefig('visualizations/clusters_items_vs_rating.png', dpi=300, bbox_inches='tight')
plt.close()

# Total Spend vs Days Since Last Purchase
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Total Spend', y='Days Since Last Purchase', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Clusters: Total Spend vs Days Since Last Purchase')
plt.savefig('visualizations/clusters_spend_vs_recency.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete! All visualizations saved to the 'visualizations' directory.")
