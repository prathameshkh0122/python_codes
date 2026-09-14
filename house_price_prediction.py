import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 1. Load dataset
df = pd.read_csv("house_price_dataset_600.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# 2. Select input features and target
X = df[
    [
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "Age_years",
        "Parking",
        "Location_Score"
    ]
]

y = df["Price_Lakh"]

# 3. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 4. Create Linear Regression model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict house prices
y_pred = model.predict(X_test)

# 7. Evaluate model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n========== MODEL EVALUATION ==========")

print(f"R2 Score: {r2:.4f}")
print(f"R2 Score (%): {r2 * 100:.2f}%")

print(f"MAE: {mae:.2f} lakh")
print(f"RMSE: {rmse:.2f} lakh")

if r2 >= 0.85:
    print("\nTarget Achieved: R2 Score is 85% or above.")
else:
    print("\nTarget Not Achieved: R2 Score is below 85%.")

# 8. Display coefficients
print("\n========== FEATURE COEFFICIENTS ==========")

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print("\nIntercept:", model.intercept_)

# 9. Predict price of a new house

new_house = pd.DataFrame({
    "Area_sqft": [1500],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age_years": [5],
    "Parking": [1],
    "Location_Score": [8]
})

predicted_price = model.predict(new_house)[0]

print("\n========== NEW HOUSE PREDICTION ==========")
print(f"Predicted House Price: ₹{predicted_price:.2f} lakh")

# 10. Actual vs Predicted graph

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price (Lakh)")
plt.ylabel("Predicted Price (Lakh)")

plt.title("Actual vs Predicted House Prices")

plt.grid(True)

plt.show()

# 11. Residual plot

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Price (Lakh)")
plt.ylabel("Residual")

plt.title("Residual Plot")

plt.grid(True)

plt.show()