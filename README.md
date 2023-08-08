# ITEM-DEMAND-FORECASTING-WITH-TIME-SERIES-AND-ML-MODEL


This repository contains the code and resources for the Item Demand Forecasting project. The goal of this project is to develop accurate sales forecasting models for 50 items using both Time Series Analysis (SARIMAX) and Machine Learning (XGBoost) techniques. The forecasts are made for the next 90 days and include seasonality considerations.

#### Table of Contents

Project Overview
Dataset
Models
Feature Engineering
Setup
Usage
Results
Deployment
License

### Project Overview

The primary objective of this project is to predict item sales for the next 90 days based on historical sales data. It involves implementing two distinct models: a SARIMAX Time Series model and an XGBoost Machine Learning model.

### Dataset

The dataset provided for this project contains four columns: date, store, item, and sales. This dataset serves as the foundation for building the forecasting models.

### Models

#### SARIMAX Time Series Model: 
Utilized the SARIMAX (Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors) technique to capture temporal patterns and seasonality. 
Order Selection: Determined the optimal order parameters for the SARIMAX model by analyzing the Partial Auto Correlation Function (PACF) plot, iterating through different parameter combinations, and selecting the combination that provided the best fit.
Model Training: Split the dataset into training and testing sets. Fit the SARIMAX model to the training data using the selected order parameters.
Forecasting: Utilized the trained model to make sales predictions for the next 90 days.
Evaluation: Evaluated the model's performance using relevant metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
#### XGBoost Machine Learning Model: 
Employed the XGBoost algorithm to harness the power of machine learning. Feature engineering was performed to enhance model performance, and hyperparameters were tuned for optimal results.
Exploratory Data Analysis (EDA): Conducted initial data exploration to understand patterns, trends, and seasonality in the sales data.

### Feature Engineering

To enhance the performance of the XGBoost model, feature engineering techniques were applied to create additional input variables, such as lagged sales values and temporal features and fine tuned the model.

### Setup

Clone the repository: git clone https://github.com/your-username/item-demand-forecasting.git
Install the necessary dependencies using pip install

### Usage

Navigate to the project directory.
Run the Jupyter Notebooks for the SARIMAX and XGBoost models to train and evaluate the models.

### Results

Both the SARIMAX Time Series model and the XGBoost Machine Learning model demonstrated remarkable accuracy in forecasting item sales for the next 90 days. The SARIMAX model effectively captured seasonal trends, while the XGBoost model leveraged machine learning to achieve high accuracy(99%)

#### Deployment

The models were deployed using Streamlit, providing an interactive interface to input item and start date for customized demand predictions.





