"""
Convert scaler.pkl to scaler.json for web use
"""

import pickle
import json
from pathlib import Path

def convert_scaler():
    """Convert StandardScaler pickle to JSON"""
    
    scaler_path = Path('models/baseline/scaler.pkl')
    output_path = Path('models/baseline/scaler.json')
    
    print("🔄 Converting scaler.pkl to scaler.json...")
    
    # Load pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Extract parameters
    scaler_data = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist(),
        'n_features': int(scaler.n_features_in_),
        'n_samples_seen': int(scaler.n_samples_seen_)
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(scaler_data, f, indent=2)
    
    print(f"✅ Created scaler.json")
    print(f"   Features: {scaler_data['n_features']}")
    print(f"   Samples seen: {scaler_data['n_samples_seen']}")
    
    # Also verify baseline_statistics.json exists
    stats_path = Path('models/baseline/baseline_statistics.json')
    if stats_path.exists():
        print(f"✅ baseline_statistics.json exists")
    else:
        print(f"⚠️  baseline_statistics.json not found")

if __name__ == '__main__':
    convert_scaler()
