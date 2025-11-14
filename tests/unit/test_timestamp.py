"""Test timestamp normalization"""
from autism_data_loader import AutismDataLoader

loader = AutismDataLoader()
data = loader.load_participant_data(1)

print("Timestamp range:")
print(f"  Min: {data['timestamp'].min()}")
print(f"  Max: {data['timestamp'].max()}")
print(f"  Duration: {data['timestamp'].max() / 1000:.1f} seconds")

print("\nFirst 10 timestamps:")
print(data['timestamp'].head(10).tolist())

print("\nLast 10 timestamps:")
print(data['timestamp'].tail(10).tolist())
