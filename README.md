# Product Backorder Risk Prediction

## Project Overview
This machine learning project predicts whether a product is likely to go on backorder using inventory, sales, forecast, and supply-chain data.

A backorder occurs when a customer orders a product that is currently unavailable in inventory and the order must be fulfilled later.

## Business Problem
Product backorders can lead to delayed deliveries, customer dissatisfaction, and supply-chain inefficiencies. Identifying products at risk of backorder can help businesses prioritize inventory planning and replenishment decisions.

## Project Objectives
- Analyze inventory and supply-chain data.
- Clean missing values and prepare features for machine learning.
- Perform exploratory data analysis.
- Build and compare classification models.
- Evaluate model performance, especially its ability to identify backorders.
- Deploy the selected model using Streamlit.

## Machine Learning Models
The following models were evaluated:
- Logistic Regression
- Decision Tree
- Random Forest

The tuned Random Forest model was selected as the final model.

## Model Performance
- Accuracy: approximately 93%
- ROC-AUC: 0.950
- Average Precision: 0.772
- Backorder recall at a 0.50 threshold: approximately 67%

These results are based on the project's test evaluation. Performance may vary on new data.

## Application
The selected model was integrated into a Streamlit application to demonstrate backorder-risk predictions for input data.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook
- Streamlit
- Joblib

## Project Notebook
See `Product_Backorder_Risk_Prediction.ipynb` for the data preparation, exploratory analysis, feature engineering, model training, evaluation, and validation workflow.

## Dataset
Dataset source: [Product Backorder Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/gowthammiryala/back-order-prediction-dataset)

The training data was used for this project.

## Disclaimer
This project is for educational and demonstration purposes. Predictions are not a guarantee that a product will or will not go on backorder.
