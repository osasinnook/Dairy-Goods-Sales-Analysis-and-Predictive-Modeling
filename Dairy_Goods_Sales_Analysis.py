# -*- coding: utf-8 -*-
"""
Dairy Goods Sales Analysis and Predictive Modeling

@author: Okosun Innocent Osas
"""

import kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

try:
    kaggle.api.dataset_download_files('suraj520/dairy-goods-sales-dataset', unzip=True)
    print("Dataset download successful.")
except Exception as e:
    print(f"Dataset download failed: {str(e)}")

df = pd.read_csv('dairy_dataset.csv')

# Convert date columns to datetime objects for proper time-based analysis
df['Date'] = pd.to_datetime(df['Date'])
df['Production Date'] = pd.to_datetime(df['Production Date'])
df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])

# Set seaborn style and context for better visualizations
sns.set(style='whitegrid', context='notebook')

# Function to plot bar plots with standardized formatting
def plot_barplot(data, x, y, title, xlabel, ylabel, top_n=10, palette='viridis'):
    """
    Generate a bar plot for top 'top_n' values of 'y' against 'x' using seaborn.
    """
    plt.figure(figsize=(14, 7))
    sns.barplot(x=x, y=y, data=data.nlargest(top_n, y), palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to analyze farm performance
def analyze_farm_performance(df):
    """
    Analyze farm performance by plotting average land area and cow population by location.
    """
    location_performance = df.groupby('Location')[['Total Land Area (acres)', 'Number of Cows']].mean().reset_index()

    # Plotting top 10 locations with largest average land area
    plot_barplot(location_performance, 'Location', 'Total Land Area (acres)',
                 'Top 10 Locations by Average Land Area',
                 'Location', 'Average Land Area (acres)')

    # Plotting top 10 locations with highest average number of cows
    plot_barplot(location_performance, 'Location', 'Number of Cows',
                 'Top 10 Locations by Average Number of Cows',
                 'Location', 'Average Number of Cows',
                 palette='coolwarm')

# Function to analyze sales distribution
def analyze_sales_distribution(df):
    """
    Analyze sales distribution by brand and customer location.
    """
    sales_distribution = df.groupby(['Brand', 'Customer Location'])['Quantity Sold (liters/kg)'].sum().unstack()

    # Get top 5 sales distributions by brand and region
    top_5_sales_distribution = sales_distribution.sum(axis=1).nlargest(5).index
    sales_distribution_top_5 = sales_distribution.loc[top_5_sales_distribution]

    # Plotting top 5 sales distribution by brand and region
    plt.figure(figsize=(14, 7))
    sales_distribution_top_5.plot(kind='bar', stacked=True, colormap='tab20c')
    plt.title('Top 5 Sales Distribution by Brand and Region')
    plt.ylabel('Quantity Sold (liters/kg)')
    plt.xlabel('Brand')
    plt.xticks(rotation=45)
    plt.legend(title='Customer Location', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Function to analyze sales distribution by product name and customer location
def analyze_sales_distribution_by_product(df):
    """
    Analyze sales distribution of dairy products by product name and customer location.
    """
    # Group by Product Name and Customer Location to calculate total quantity sold
    sales_distribution = df.groupby(['Product Name', 'Customer Location'])['Quantity Sold (liters/kg)'].sum().unstack()

    # Get top 5 sales distributions by product and region
    top_5_sales_distribution = sales_distribution.sum(axis=1).nlargest(5).index
    sales_distribution_top_5 = sales_distribution.loc[top_5_sales_distribution]

    # Plotting top 5 sales distribution by product and region
    plt.figure(figsize=(14, 7))
    sales_distribution_top_5.plot(kind='bar', stacked=True, colormap='tab20c')
    plt.title('Sales Distribution of Dairy Products by Product Name and Customer Location')
    plt.ylabel('Quantity Sold (liters/kg)')
    plt.xlabel('Product Name')
    plt.xticks(rotation=45)
    plt.legend(title='Customer Location', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Function to analyze impact of storage conditions on shelf life
def analyze_storage_impact(df):
    """
    Analyze the impact of storage conditions on shelf life using a boxplot.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Storage Condition', y='Shelf Life (days)', data=df)
    plt.title('Impact of Storage Condition on Shelf Life')
    plt.xlabel('Storage Condition')
    plt.ylabel('Shelf Life (days)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to analyze customer preferences
def analyze_customer_preferences(df):
    """
    Analyze customer preferences by location and sales channel using stacked bar plots.
    """
    customer_preferences = df.groupby(['Customer Location', 'Sales Channel'])['Approx. Total Revenue(INR)'].sum().unstack()

    # Plotting customer preferences by location and sales channel
    plt.figure(figsize=(14, 7))
    customer_preferences.plot(kind='bar', stacked=True, colormap='Paired')
    plt.title('Customer Preferences by Location and Sales Channel')
    plt.ylabel('Approx. Total Revenue (INR)')
    plt.xlabel('Customer Location')
    plt.xticks(rotation=45)
    plt.legend(title='Sales Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Function to analyze inventory management
def analyze_inventory_management(df):
    """
    Analyze inventory management by product name using bar plots.
    """
    inventory_management = df.groupby('Product Name')[['Quantity in Stock (liters/kg)', 'Minimum Stock Threshold (liters/kg)', 'Reorder Quantity (liters/kg)']].sum().reset_index()

    # Plotting inventory management by product name
    plt.figure(figsize=(14, 7))
    inventory_management.set_index('Product Name').plot(kind='bar', colormap='Dark2')
    plt.title('Inventory Management by Product Name')
    plt.ylabel('Quantity (liters/kg)')
    plt.xlabel('Product Name')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to analyze sales trends
def analyze_sales_trends(df):
    """
    Analyze sales trends over time using line plots.
    """
    sales_trends = df.groupby(df['Date'].dt.to_period('M'))['Approx. Total Revenue(INR)'].sum().reset_index()
    sales_trends['Date'] = sales_trends['Date'].dt.to_timestamp()

    # Plotting sales trends over time
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='Date', y='Approx. Total Revenue(INR)', data=sales_trends, marker='o')
    plt.title('Sales Trends Over Time')
    plt.ylabel('Approx. Total Revenue (INR)')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Perform all analyses
analyze_farm_performance(df)
analyze_sales_distribution(df)
analyze_sales_distribution_by_product(df)
analyze_storage_impact(df)
analyze_customer_preferences(df)
analyze_inventory_management(df)
analyze_sales_trends(df)


# Feature Engineering: Extracting useful features from date columns
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

def demand_forecasting_model(df):
    # Selecting relevant features for demand forecasting
    features_demand = ['Month', 'Year', 'Product ID', 'Quantity Sold (liters/kg)', 'Price per Unit', 'Total Value']
    X_demand = df[features_demand]
    y_demand = df['Quantity Sold (liters/kg)']

    # Splitting data into training and testing sets
    X_train_demand, X_test_demand, y_train_demand, y_test_demand = train_test_split(X_demand, y_demand, test_size=0.2, random_state=42)

    # Scaling features for better model performance
    scaler_demand = StandardScaler()
    X_train_demand_scaled = scaler_demand.fit_transform(X_train_demand)
    X_test_demand_scaled = scaler_demand.transform(X_test_demand)

    # Initializing and training a Linear Regression model for demand forecasting
    model_demand = LinearRegression()
    model_demand.fit(X_train_demand_scaled, y_train_demand)

    # Making predictions on the test data
    y_pred_demand = model_demand.predict(X_test_demand_scaled)

    # Evaluating the model using Root Mean Squared Error (RMSE)
    rmse_demand = np.sqrt(mean_squared_error(y_test_demand, y_pred_demand))
    print(f'Demand Forecasting - Root Mean Squared Error: {rmse_demand:.2f}')

     # Evaluating the model the test score
    model_demand_score = model_demand.score(X_test_demand_scaled, y_test_demand )
    print(f'Demand Forecasting - Test Score: {model_demand_score:.2f}')

    # Visualizing Actual vs Predicted Quantity Sold
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_demand, y_pred_demand, alpha=0.3)
    plt.xlabel('Actual Quantity Sold')
    plt.ylabel('Predicted Quantity Sold')
    plt.title('Demand Forecasting: Actual vs Predicted Quantity Sold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model_demand, scaler_demand

# Pricing Strategy Model
def pricing_strategy_model(df):
    # Selecting relevant features for pricing strategy
    features_pricing = ['Month', 'Year', 'Product ID', 'Approx. Total Revenue(INR)', 'Quantity Sold (liters/kg)', 'Total Value', 'Quantity in Stock (liters/kg)', 'Minimum Stock Threshold (liters/kg)',
       'Reorder Quantity (liters/kg)']
    X_pricing = df[features_pricing]
    y_pricing = df['Price per Unit (sold)']

    # Splitting data into training and testing sets
    X_train_pricing, X_test_pricing, y_train_pricing, y_test_pricing = train_test_split(X_pricing, y_pricing, test_size=0.2, random_state=42)

    # Scaling features for better model performance
    scaler_pricing = StandardScaler()
    X_train_pricing_scaled = scaler_pricing.fit_transform(X_train_pricing)
    X_test_pricing_scaled = scaler_pricing.transform(X_test_pricing)

    # Initializing and training a Random Forest Regressor model for pricing strategy
    model_pricing = RandomForestRegressor(n_estimators=100, random_state=42)
    model_pricing.fit(X_train_pricing_scaled, y_train_pricing)

    # Making predictions on the test data
    y_pred_pricing = model_pricing.predict(X_test_pricing_scaled)

    # Evaluating the model using Root Mean Squared Error (RMSE)
    rmse_pricing = np.sqrt(mean_squared_error(y_test_pricing, y_pred_pricing))
    print(f'Pricing Strategy - Root Mean Squared Error: {rmse_pricing:.2f}')

    # Evaluating the model the test score
    pred_pricing_score = model_pricing.score(X_test_pricing_scaled, y_test_pricing )
    print(f'Pricing Strategy - Test Score: {pred_pricing_score:.2f}')

    # Visualizing Actual vs Predicted Price per Unit Sold
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_pricing, y_pred_pricing, alpha=0.3)
    plt.xlabel('Actual Price per Unit (sold)')
    plt.ylabel('Predicted Price per Unit (sold)')
    plt.title('Pricing Strategy: Actual vs Predicted Price per Unit (sold)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model_pricing, scaler_pricing

# Example usage:
demand_model, demand_scaler = demand_forecasting_model(df)
pricing_model, pricing_scaler = pricing_strategy_model(df)

