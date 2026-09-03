# SCT_ML_1

# House Price Prediction using Linear Regression

## SkillCraft Technology - Machine Learning Internship

### Task 01

## Objective

The objective of this project is to implement a Linear Regression
model to predict house prices based on square footage, number of
bedrooms, and number of bathrooms.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

House Prices dataset containing residential property information.

## Features

- Square Footage
- Number of Bedrooms
- Number of Bathrooms

## Target

- House Price

## Machine Learning Algorithm

Linear Regression

## Workflow

1. Load the dataset
2. Explore the data
3. Handle missing values
4. Select relevant features
5. Split data into training and testing sets
6. Train Linear Regression model
7. Make predictions
8. Evaluate the model
9. Visualize actual vs predicted prices
10. Predict price for a new house

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Conclusion

The Linear Regression model provides a baseline solution for predicting
house prices using property features such as square footage, bedrooms,
and bathrooms. The project demonstrates the complete machine learning
workflow from data preprocessing to prediction and evaluation.

#python code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("train.csv")

print("First 5 rows:")
print(df.head())

# Select required columns
data = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].copy()

# Rename columns
data.rename(columns={
    'GrLivArea': 'SquareFootage',
    'BedroomAbvGr': 'Bedrooms',
    'FullBath': 'Bathrooms',
    'SalePrice': 'Price'
}, inplace=True)

# Remove missing values
data.dropna(inplace=True)

print("\nDataset shape:", data.shape)

# Features and target
X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n----- Model Performance -----")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Coefficients
print("\n----- Model Coefficients -----")
print("Intercept:", model.intercept_)

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

# Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()

# User prediction
print("\n----- New House Prediction -----")

sqft = float(input("Enter square footage: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

new_house = pd.DataFrame({
    'SquareFootage': [sqft],
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms]
})

prediction = model.predict(new_house)

print("Estimated House Price:", prediction[0])
