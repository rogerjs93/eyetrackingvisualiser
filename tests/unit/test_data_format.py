"""Quick test to check autism data format"""
from autism_data_loader import AutismDataLoader
import pandas as pd

loader = AutismDataLoader()
data = loader.load_participant_data(1)

print("Columns:", data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())
print("\nData types:")
print(data.dtypes)
print("\nData shape:", data.shape)
print("\nNull values:")
print(data.isnull().sum())

# Check if data can be converted to dict (for Dash)
data_dict = data.to_dict('records')
print("\nFirst record as dict:")
print(data_dict[0])
