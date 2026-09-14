import pandas as pd
import numpy as np

np.random.seed(42)

# Number of rows
n = 600

# Generate house features
area = np.random.randint(500, 4001, n)

bedrooms = np.random.randint(1, 6, n)

bathrooms = np.clip(
    bedrooms
    - np.random.binomial(1, 0.35, n)
    + np.random.binomial(1, 0.35, n),
    1,
    5
)

age = np.random.randint(0, 31, n)

parking = np.random.randint(0, 3, n)

location_score = np.random.randint(1, 11, n)

# Generate house prices
price = (
    18
    + 0.085 * area
    + 8.0 * bedrooms
    + 5.0 * bathrooms
    + 3.0 * parking
    + 7.0 * location_score
    - 0.65 * age
    + np.random.normal(0, 12, n)
)

price = np.round(price, 2)

# Create dataframe
df = pd.DataFrame({
    "Area_sqft": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Age_years": age,
    "Parking": parking,
    "Location_Score": location_score,
    "Price_Lakh": price
})

# Save CSV
df.to_csv("house_price_dataset_600.csv", index=False)

print("Dataset created successfully!")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("\nFile created: house_price_dataset_600.csv")
