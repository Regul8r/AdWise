import pandas as pd

df = pd.read_csv('product_advertising.csv')  

# Basic checks
print("✓ File loaded successfully!")
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())