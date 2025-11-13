"""
Data Preparation Script for Eye-Tracking Training
Converts your CSV format to training-ready feature matrices

Your CSV Format (detected):
- Columns: Pupil Size Right X [px], Pupil Size Right Y [px]
- Multiple participants in single file
- Trials: Trial001, Trial002, etc.
- Participant: Unidentified(Neg), etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Import feature extractor from training script
from train_enhanced_model import EnhancedFeatureExtractor


def load_pupil_size_csv(csv_path):
    """
    Load CSV with Pupil Size format
    
    Expected columns:
    - Pupil Size Right X [px]
    - Pupil Size Right Y [px]
    - recordingtime [ms]
    - participant
    - trial
    """
    print(f"\n{'='*60}")
    print(f"Loading: {csv_path}")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"📋 Columns: {list(df.columns)[:10]}...")
    
    # Rename columns for consistency
    x_col = 'Pupil Size Right X [px]'
    y_col = 'Pupil Size Right Y [px]'
    time_col = 'recordingtime [ms]'
    
    if x_col not in df.columns or y_col not in df.columns:
        print(f"❌ Error: Expected columns '{x_col}' and '{y_col}'")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Filter out invalid points (zeros, dashes, missing)
    df = df[df[x_col] != '-']
    df = df[df[y_col] != '-']
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    
    # Remove NaN and zeros
    df = df.dropna(subset=[x_col, y_col])
    df = df[(df[x_col] != 0) | (df[y_col] != 0)]  # Keep if at least one is non-zero
    
    print(f"✅ After filtering: {len(df)} valid points")
    
    return df, x_col, y_col, time_col


def normalize_coordinates(df, x_col, y_col):
    """
    Normalize coordinates to 0-100 range
    """
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    
    df['x'] = (df[x_col] - x_min) / (x_max - x_min) * 100
    df['y'] = (df[y_col] - y_min) / (y_max - y_min) * 100
    
    print(f"📊 Normalized: X[{x_min:.2f}, {x_max:.2f}] → [0, 100]")
    print(f"📊 Normalized: Y[{y_min:.2f}, {y_max:.2f}] → [0, 100]")
    
    return df


def split_by_participant_and_trial(df):
    """
    Split data by participant and trial
    Each (participant, trial) combination = 1 training sample
    """
    print(f"\n{'='*60}")
    print("Splitting by participant and trial")
    print(f"{'='*60}")
    
    if 'participant' in df.columns and 'trial' in df.columns:
        grouped = df.groupby(['participant', 'trial'])
        print(f"✅ Found {len(grouped)} unique (participant, trial) combinations")
    elif 'participant' in df.columns:
        grouped = df.groupby('participant')
        print(f"✅ Found {len(grouped)} unique participants")
    else:
        print("⚠️ No participant/trial columns found. Treating entire dataset as 1 sample.")
        grouped = [('sample', df)]
    
    samples = []
    for name, group in grouped:
        if len(group) < 10:  # Skip very short sequences
            continue
        
        # Prepare data for feature extraction
        sample_data = pd.DataFrame({
            'x': group['x'].values,
            'y': group['y'].values,
            'timestamp': group[group.columns[1]].values if len(group.columns) > 1 else range(len(group)),
            'fixation_duration': 200,  # Default if not available
            'pupil_size': 3.5  # Default if not available
        })
        
        samples.append({
            'name': str(name),
            'data': sample_data,
            'n_points': len(sample_data)
        })
    
    print(f"✅ Created {len(samples)} training samples")
    print(f"📊 Points per sample: min={min(s['n_points'] for s in samples)}, "
          f"max={max(s['n_points'] for s in samples)}, "
          f"mean={np.mean([s['n_points'] for s in samples]):.0f}")
    
    return samples


def extract_features_from_samples(samples):
    """
    Extract 43 features from each sample
    """
    print(f"\n{'='*60}")
    print("Extracting 43 features from each sample")
    print(f"{'='*60}")
    
    extractor = EnhancedFeatureExtractor()
    
    feature_matrix = []
    sample_names = []
    
    for i, sample in enumerate(samples):
        try:
            features = extractor.extract_features(sample['data'])
            feature_matrix.append(features)
            sample_names.append(sample['name'])
            
            if (i + 1) % 10 == 0:
                print(f"✅ Processed {i+1}/{len(samples)} samples...")
        
        except Exception as e:
            print(f"❌ Error processing sample {sample['name']}: {e}")
            continue
    
    feature_matrix = np.array(feature_matrix)
    
    print(f"\n✅ Extracted features: {feature_matrix.shape}")
    print(f"📊 Feature matrix: {len(sample_names)} samples × {feature_matrix.shape[1]} features")
    
    # Check for NaN/Inf
    nan_count = np.isnan(feature_matrix).sum()
    inf_count = np.isinf(feature_matrix).sum()
    
    if nan_count > 0:
        print(f"⚠️ Found {nan_count} NaN values - will be replaced with 0")
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    
    if inf_count > 0:
        print(f"⚠️ Found {inf_count} Inf values - will be clipped")
        feature_matrix = np.nan_to_num(feature_matrix, posinf=1e6, neginf=-1e6)
    
    return feature_matrix, sample_names, extractor.feature_names


def save_processed_data(feature_matrix, sample_names, feature_names, output_path):
    """
    Save processed features for training
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True)
    
    # Save as numpy
    np.save(output_path, feature_matrix)
    print(f"✅ Saved feature matrix: {output_path}")
    
    # Save metadata
    metadata = {
        'n_samples': len(sample_names),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'sample_names': sample_names,
        'feature_statistics': {
            name: {
                'mean': float(feature_matrix[:, i].mean()),
                'std': float(feature_matrix[:, i].std()),
                'min': float(feature_matrix[:, i].min()),
                'max': float(feature_matrix[:, i].max())
            }
            for i, name in enumerate(feature_names)
        }
    }
    
    metadata_path = output_path.replace('.npy', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata: {metadata_path}")


def main():
    """
    Main data preparation pipeline
    
    Usage:
        python prepare_training_data.py <csv_path> <output_name> <group_label>
    
    Example:
        python prepare_training_data.py data/children_asd.csv data/features_children_asd.npy children_asd
    """
    if len(sys.argv) < 4:
        print("Usage: python prepare_training_data.py <csv_path> <output_path> <group_label>")
        print("\nExample:")
        print("  python prepare_training_data.py data/children_asd.csv data/features_children.npy children_asd")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2]
    group_label = sys.argv[3]
    
    print(f"\n{'='*60}")
    print(f"EYE-TRACKING DATA PREPARATION")
    print(f"{'='*60}")
    print(f"Input: {csv_path}")
    print(f"Output: {output_path}")
    print(f"Group: {group_label}")
    
    # Step 1: Load CSV
    df, x_col, y_col, time_col = load_pupil_size_csv(csv_path)
    
    # Step 2: Normalize coordinates
    df = normalize_coordinates(df, x_col, y_col)
    
    # Step 3: Split by participant/trial
    samples = split_by_participant_and_trial(df)
    
    if len(samples) < 5:
        print(f"\n⚠️ WARNING: Only {len(samples)} samples found!")
        print("For robust training, you need at least 15-20 samples per group.")
        print("Consider:")
        print("  - Combining multiple CSV files")
        print("  - Using different trials as separate samples")
        print("  - Recruiting more participants")
    
    # Step 4: Extract features
    feature_matrix, sample_names, feature_names = extract_features_from_samples(samples)
    
    # Step 5: Save processed data
    save_processed_data(feature_matrix, sample_names, feature_names, output_path)
    
    print(f"\n{'='*60}")
    print(f"✅ DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Samples: {len(sample_names)}")
    print(f"Features: {len(feature_names)}")
    print(f"\nNext step: Train model using this data")
    print(f"  python train_enhanced_model.py --data {output_path}")


if __name__ == '__main__':
    main()
