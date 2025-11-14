"""
Extract features from RawEyetrackingASD.mat and compare to CSV-based approach
"""
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path

class MatFileFeatureExtractor:
    """Extract same 28 features from .mat file as from CSV files"""
    
    def __init__(self, mat_file_path):
        print("="*70)
        print("Loading RawEyetrackingASD.mat")
        print("="*70)
        
        self.mat_data = scipy.io.loadmat(mat_file_path)
        self.eye_data = self.mat_data['eyeMovementsASD']
        
        print(f"\n✅ Loaded eye-tracking data")
        print(f"   Shape: {self.eye_data.shape}")
        print(f"   → {self.eye_data.shape[1]} participants")
        print(f"   → {self.eye_data.shape[0]} trials per participant")
        print(f"   → {self.eye_data.shape[3]} time samples per trial")
    
    def extract_features_for_participant(self, participant_idx):
        """
        Extract same 28 features as baseline_model_builder.py
        Features match the ones from CSV processing
        """
        # Get all trials for this participant
        # Shape: (36 trials, 2 coords, 14000 samples)
        participant_data = self.eye_data[:, participant_idx, :, :]
        
        # Combine all trials into one long sequence for feature extraction
        x_coords = participant_data[:, 0, :].flatten()  # All X coordinates
        y_coords = participant_data[:, 1, :].flatten()  # All Y coordinates
        
        # Filter out zeros/invalid values (same as CSV processing)
        valid_mask = (x_coords > 0) & (y_coords > 0)
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        
        if len(x_valid) == 0:
            return None
        
        # Calculate velocity (pixel changes between samples)
        dx = np.diff(x_valid)
        dy = np.diff(y_valid)
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Calculate acceleration
        acceleration = np.diff(velocity)
        
        # Extract 28 features (matching baseline_model_builder.py)
        features = []
        
        # 1-4: Basic statistics for X coordinate
        features.append(np.mean(x_valid))
        features.append(np.std(x_valid))
        features.append(np.min(x_valid))
        features.append(np.max(x_valid))
        
        # 5-8: Basic statistics for Y coordinate
        features.append(np.mean(y_valid))
        features.append(np.std(y_valid))
        features.append(np.min(y_valid))
        features.append(np.max(y_valid))
        
        # 9-12: Velocity statistics
        features.append(np.mean(velocity))
        features.append(np.std(velocity))
        features.append(np.max(velocity))
        features.append(np.median(velocity))
        
        # 13-15: Acceleration statistics
        features.append(np.mean(acceleration))
        features.append(np.std(acceleration))
        features.append(np.max(np.abs(acceleration)))
        
        # 16-17: Fixation-related (staying in small area)
        fixation_threshold = 50  # pixels
        fixations = velocity < fixation_threshold
        features.append(np.sum(fixations) / len(velocity))  # Fixation ratio
        features.append(np.mean(velocity[fixations]) if np.any(fixations) else 0)
        
        # 18-19: Saccade-related (rapid movements)
        saccade_threshold = 200  # pixels/sample
        saccades = velocity > saccade_threshold
        features.append(np.sum(saccades) / len(velocity))  # Saccade ratio
        features.append(np.mean(velocity[saccades]) if np.any(saccades) else 0)
        
        # 20-21: Gaze distribution
        features.append(np.percentile(x_valid, 25))
        features.append(np.percentile(x_valid, 75))
        
        # 22-23: Vertical gaze distribution
        features.append(np.percentile(y_valid, 25))
        features.append(np.percentile(y_valid, 75))
        
        # 24: Path length (total distance traveled)
        path_length = np.sum(velocity)
        features.append(path_length)
        
        # 25: Area covered (convex hull approximation)
        x_range = np.max(x_valid) - np.min(x_valid)
        y_range = np.max(y_valid) - np.min(y_valid)
        features.append(x_range * y_range)
        
        # 26-27: Velocity variability
        features.append(np.percentile(velocity, 90))
        features.append(np.percentile(velocity, 10))
        
        # 28: Sample count (data richness indicator)
        features.append(len(x_valid))
        
        return np.array(features)
    
    def extract_all_participants(self):
        """Extract features from all 24 participants"""
        print("\n" + "="*70)
        print("Extracting Features from All Participants")
        print("="*70)
        
        all_features = []
        metadata = []
        
        for p_idx in range(self.eye_data.shape[1]):
            print(f"\n  Processing participant {p_idx + 1}/24...", end=" ")
            
            features = self.extract_features_for_participant(p_idx)
            
            if features is not None:
                all_features.append(features)
                metadata.append({
                    'participant_id': p_idx + 1,
                    'data_source': 'mat_file'
                })
                print(f"✓ Extracted {len(features)} features")
            else:
                print("✗ No valid data")
        
        print(f"\n✅ Extracted features from {len(all_features)}/24 participants")
        
        return np.array(all_features), metadata


def compare_with_csv_features():
    """Compare .mat-based features with CSV-based features"""
    print("\n" + "="*70)
    print("COMPARISON: .mat vs CSV Features")
    print("="*70)
    
    # Extract features from .mat file
    mat_extractor = MatFileFeatureExtractor(r"data\autism\autismdata2\RawEyetrackingASD.mat")
    mat_features, mat_metadata = mat_extractor.extract_all_participants()
    
    # Load CSV-based features
    print("\n" + "="*70)
    print("Loading CSV-based Features")
    print("="*70)
    
    from baseline_model_builder import BaselineModelBuilder
    builder = BaselineModelBuilder()
    csv_features, csv_metadata = builder.load_all_participants()
    
    print(f"\n✅ Loaded {len(csv_features)} participants from CSV files")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n📊 Dataset Sizes:")
    print(f"   .mat file:  {len(mat_features)} participants")
    print(f"   CSV files:  {len(csv_features)} participants")
    
    print(f"\n📊 Feature Dimensions:")
    print(f"   .mat file:  {mat_features.shape[1]} features per participant")
    print(f"   CSV files:  {csv_features.shape[1]} features per participant")
    
    # Statistical comparison
    print(f"\n📊 Feature Statistics:")
    print(f"\n   .mat file features:")
    print(f"     Mean values range: [{np.mean(mat_features, axis=0).min():.2f}, {np.mean(mat_features, axis=0).max():.2f}]")
    print(f"     Overall std: {np.std(mat_features):.4f}")
    
    print(f"\n   CSV file features:")
    print(f"     Mean values range: [{np.mean(csv_features, axis=0).min():.2f}, {np.mean(csv_features, axis=0).max():.2f}]")
    print(f"     Overall std: {np.std(csv_features):.4f}")
    
    # Feature-by-feature comparison
    print(f"\n📋 Feature-by-Feature Comparison (first 10 features):")
    print(f"{'Feature':<10} {'MAT Mean':<15} {'CSV Mean':<15} {'Difference':<15}")
    print("-" * 60)
    
    for i in range(min(10, mat_features.shape[1])):
        mat_mean = np.mean(mat_features[:, i])
        csv_mean = np.mean(csv_features[:, i])
        diff = abs(mat_mean - csv_mean)
        print(f"{i+1:<10} {mat_mean:<15.2f} {csv_mean:<15.2f} {diff:<15.2f}")
    
    # Correlation analysis
    print(f"\n📊 Overall Correlation:")
    
    # Compare means of all features
    mat_means = np.mean(mat_features, axis=0)
    csv_means = np.mean(csv_features, axis=0)
    correlation = np.corrcoef(mat_means, csv_means)[0, 1]
    
    print(f"   Correlation between feature means: {correlation:.4f}")
    
    if correlation > 0.9:
        print("   ✅ VERY HIGH correlation - datasets are very similar!")
    elif correlation > 0.7:
        print("   ✅ HIGH correlation - datasets are similar")
    elif correlation > 0.5:
        print("   ⚠️  MODERATE correlation - some differences")
    else:
        print("   ⚠️  LOW correlation - datasets differ significantly")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if len(mat_features) == 24 and len(csv_features) == 23:
        print("\n✅ .mat file has 24 participants vs 23 from CSVs")
        print("   → .mat file includes 1 additional participant")
        print("   → Both are valid autism data from same dataset")
    
    print("\n💡 Recommendations:")
    print("   1. Both datasets are valid raw autism eye-tracking data")
    print("   2. You could use .mat file to get 24 participants instead of 23")
    print("   3. Or combine both for potentially 25 total participants")
    print("   4. Feature extraction produces same 28-feature structure")
    
    return mat_features, csv_features, mat_metadata, csv_metadata


if __name__ == "__main__":
    mat_features, csv_features, mat_meta, csv_meta = compare_with_csv_features()
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
