# E-commerce Customer Behavior Analysis

## Project Overview

This project analyzes customer behavior patterns in an e-commerce platform using a comprehensive dataset of customer interactions and transactions. The analysis provides insights into customer spending habits, segmentation, satisfaction levels, and demographic patterns to help businesses make data-driven decisions.

## Dataset Description

The dataset contains detailed information about customer behavior within an e-commerce platform, with each entry corresponding to a unique customer. It includes the following features:

- **Customer ID**: Unique identifier for each customer
- **Gender**: Customer gender (Male/Female)
- **Age**: Customer age
- **City**: City of residence
- **Membership Type**: Type of membership (Gold, Silver, Bronze)
- **Total Spend**: Total monetary expenditure on the platform
- **Items Purchased**: Number of items purchased
- **Average Rating**: Average rating given by the customer (0-5)
- **Discount Applied**: Whether a discount was applied (True/False)
- **Days Since Last Purchase**: Number of days since the customer's most recent purchase
- **Satisfaction Level**: Overall satisfaction level (Satisfied, Neutral, Unsatisfied)

## Analysis Methods

This project employs several analytical techniques to extract insights from the e-commerce customer data:

### 1. Exploratory Data Analysis (EDA)
- Demographic analysis (age, gender, city)
- Spending patterns across different customer segments
- Correlation analysis between numerical variables
- Impact of discounts on purchasing behavior

### 2. Customer Segmentation
- **RFM Analysis**: Segments customers based on Recency, Frequency, and Monetary value
  - Recency: Days since last purchase
  - Frequency: Number of items purchased
  - Monetary: Total spend
- **K-means Clustering**: Identifies natural groupings of customers based on multiple features
  - Determines optimal number of clusters using silhouette scores
  - Creates customer segments based on behavioral patterns

### 3. Satisfaction Analysis
- Relationship between satisfaction levels and spending
- Correlation between ratings and satisfaction
- Impact of membership type on satisfaction

## Key Visualizations

The analysis generates multiple visualizations to help understand customer behavior:

1. **Spending Patterns**:
   - Average spending by gender
   - Average spending by membership type
   - Top cities by total spending
   - Relationship between age and spending

2. **Customer Segmentation**:
   - RFM segment distribution
   - Average spending by customer segment
   - K-means cluster visualizations
   - Silhouette scores for optimal clustering

3. **Satisfaction Analysis**:
   - Distribution of satisfaction levels
   - Average rating by satisfaction level
   - Days since purchase by satisfaction level

4. **Correlation Analysis**:
   - Correlation matrix of numerical variables
   - Relationship between key metrics

## Project Structure

```
├── ecommerce_customer_behavior.csv   # Original dataset
├── analysis.py                       # Main analysis script
├── data/                             # Processed data files
│   ├── cleaned_ecommerce_data.csv    # Cleaned dataset
│   ├── rfm_segments.csv              # RFM segmentation results
│   └── customer_clusters.csv         # K-means clustering results
├── visualizations/                   # Generated visualizations
│   ├── spending_by_gender.png
│   ├── spending_by_membership.png
│   ├── top_cities_by_spending.png
│   ├── age_distribution.png
│   ├── age_vs_spending.png
│   ├── satisfaction_distribution.png
│   ├── rating_by_satisfaction.png
│   ├── spending_by_discount.png
│   ├── correlation_matrix.png
│   ├── days_since_purchase_distribution.png
│   ├── days_since_purchase_by_satisfaction.png
│   ├── rfm_segments_distribution.png
│   ├── spending_by_segment.png
│   ├── kmeans_silhouette_scores.png
│   ├── clusters_age_vs_spend.png
│   ├── clusters_items_vs_rating.png
│   └── clusters_spend_vs_recency.png
└── README.md                         # Project documentation
```

## How to Run the Analysis

### Prerequisites
- Python 3.6 or higher
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn

### Installation

1. Clone this repository:
```bash
git clone https://github.com/keerthika-art/E-commerce-Analysis.git
cd E-commerce-Analysis
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Run the analysis script:
```bash
python analysis.py
```

This will generate all visualizations in the `visualizations` directory and processed data files in the `data` directory.

## Key Insights

The analysis reveals several important insights about customer behavior:

1. **Membership Tiers**: Gold members spend significantly more than Silver and Bronze members, suggesting that loyalty programs effectively encourage higher spending.

2. **Customer Segments**: The RFM analysis identifies key customer segments including:
   - Champions: High-value, frequent, recent customers
   - Loyal Customers: Regular spenders with good purchase frequency
   - At Risk: Previously valuable customers who haven't purchased recently
   - Hibernating: Customers who made few purchases and haven't returned

3. **Satisfaction Correlation**: Customer satisfaction strongly correlates with average rating and inversely correlates with days since last purchase.

4. **Discount Impact**: Customers who received discounts show different spending patterns compared to those who didn't, providing insights into promotion effectiveness.

5. **Geographic Patterns**: Certain cities show significantly higher spending, suggesting opportunities for targeted marketing.

## Business Recommendations

Based on the analysis, several recommendations can be made:

1. **Targeted Marketing**: Develop personalized marketing strategies for different customer segments:
   - Reward and retain Champions with exclusive offers
   - Re-engage At Risk customers with special promotions
   - Convert Loyal Customers to Champions with loyalty incentives

2. **Membership Program Enhancement**: Optimize the membership program to encourage upgrades from Bronze to Silver and Silver to Gold tiers.

3. **Geographic Focus**: Allocate more marketing resources to high-performing cities while developing strategies to improve performance in underperforming regions.

4. **Discount Strategy Optimization**: Refine discount strategies based on their impact on different customer segments.

5. **Satisfaction Improvement**: Address factors contributing to low satisfaction levels, particularly focusing on customers with longer periods since their last purchase.

## Future Work

Potential extensions to this analysis include:

1. Time series analysis to identify seasonal patterns and trends
2. Predictive modeling to forecast customer lifetime value
3. Churn prediction to identify customers at risk of leaving
4. Product recommendation system based on purchase patterns
5. A/B testing framework for evaluating marketing strategies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided for educational purposes
- Analysis inspired by real-world e-commerce business challenges
