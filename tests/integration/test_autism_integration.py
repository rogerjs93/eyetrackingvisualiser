"""
Simple test to verify autism data loading works with the dashboard
"""

from autism_data_loader import AutismDataLoader
import pandas as pd

print("=" * 70)
print("Testing Autism Data Integration")
print("=" * 70)

# Initialize loader
loader = AutismDataLoader()

# Get available participants
participants = loader.get_available_participants()
print(f"\n✅ Found {len(participants)} participants")

# Test loading a few participants
for pid in participants[:3]:
    print(f"\n📊 Participant {pid}:")
    
    # Get info
    info = loader.get_participant_info(pid)
    if info:
        print(f"   Age: {info['Age']}, Gender: {info['Gender']}, CARS: {info['CARS Score']}")
    
    # Load data
    data = loader.load_participant_data(pid)
    print(f"   Points: {len(data)}, Duration: {data['timestamp'].max()/1000:.1f}s")
    
    # Verify data format (same as synthetic data)
    required_cols = ['x', 'y', 'timestamp', 'duration']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        print(f"   ⚠️  Missing columns: {missing}")
    else:
        print(f"   ✅ Data format compatible with dashboard")

print("\n" + "=" * 70)
print("✅ Integration test complete!")
print("=" * 70)
