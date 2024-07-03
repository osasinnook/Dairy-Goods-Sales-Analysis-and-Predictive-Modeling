# Dairy Goods Sales Analysis and Predictive Modeling

## Introduction
This project involves analyzing the Dairy Goods Sales Dataset, which provides a detailed and comprehensive collection of data related to dairy farms, dairy products, sales, and inventory management. The primary goals are to understand the factors affecting dairy farm performance, sales distribution, storage impact, customer preferences, inventory management, and sales trends. Additionally, we aim to develop predictive models for demand forecasting and pricing strategies.

## Dataset Description
The dataset encompasses various features including:
- **Location:** Geographical location of the dairy farm.
- **Total Land Area (acres):** Total land area occupied by the dairy farm.
- **Number of Cows:** Number of cows present in the dairy farm.
- **Farm Size:** Size of the dairy farm (in sq.km).
- **Date:** Date of data recording.
- **Product ID:** Unique identifier for each dairy product.
- **Product Name:** Name of the dairy product.
- **Brand:** Brand associated with the dairy product.
- **Quantity (liters/kg):** Quantity of the dairy product available.
- **Price per Unit:** Price per unit of the dairy product.
- **Total Value:** Total value of the available quantity of the dairy product.
- **Shelf Life (days):** Shelf life of the dairy product in days.
- **Storage Condition:** Recommended storage condition for the dairy product.
- **Production Date:** Date of production for the dairy product.
- **Expiration Date:** Date of expiration for the dairy product.
- **Quantity Sold (liters/kg):** Quantity of the dairy product sold.
- **Price per Unit (sold):** Price per unit at which the dairy product was sold.
- **Approx. Total Revenue (INR):** Approximate total revenue generated from the sale of the dairy product.
- **Customer Location:** Location of the customer who purchased the dairy product.
- **Sales Channel:** Channel through which the dairy product was sold (Retail, Wholesale, Online).
- **Quantity in Stock (liters/kg):** Quantity of the dairy product remaining in stock.
- **Minimum Stock Threshold (liters/kg):** Minimum stock threshold for the dairy product.
- **Reorder Quantity (liters/kg):** Recommended quantity to reorder for the dairy product.

The dataset includes data from the period between 2019 and 2022, focusing on selected dairy brands operating in specific states and union territories of India.

## Exploratory Data Analysis (EDA)
We performed several analyses to understand different aspects of the dairy goods sales data:

### Farm Performance Analysis
- Analyzed average land area and cow population by location.
- Plotted top 10 locations with the largest average land area and the highest average number of cows.

### Sales Distribution Analysis
- Analyzed sales distribution by brand and customer location.
- Analyzed sales distribution of dairy products by product name and customer location.

### Storage Impact Analysis
- Analyzed the impact of storage conditions on shelf life using a boxplot.

### Customer Preferences Analysis
- Analyzed customer preferences by location and sales channel using stacked bar plots.

### Inventory Management Analysis
- Analyzed inventory management by product name using bar plots.

### Sales Trends Analysis
- Analyzed sales trends over time using line plots.

## Predictive Modeling
We developed two predictive models:

### Demand Forecasting Model
- **Features:** Month, Year, Product ID, Quantity Sold (liters/kg), Price per Unit, Total Value.
- **Model:** Linear Regression.
- **Performance:** Evaluated using Root Mean Squared Error (RMSE) and test score.

### Pricing Strategy Model
- **Features:** Month, Year, Product ID, Approx. Total Revenue (INR), Quantity Sold (liters/kg), Total Value, Quantity in Stock (liters/kg), Minimum Stock Threshold (liters/kg), Reorder Quantity (liters/kg).
- **Model:** Random Forest Regressor.
- **Performance:** Evaluated using RMSE and test score.
