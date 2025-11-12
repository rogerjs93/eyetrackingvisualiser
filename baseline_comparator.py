"""
Baseline Comparator
Compares new eye-tracking data against the autism baseline model.
Provides similarity scores and deviation analysis.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
import json
from pathlib import Path
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import zscore

class BaselineComparator:
    """Compare eye-tracking data against autism baseline model."""
    
    def __init__(self, model_dir='models/baseline'):
        """Load baseline model and metadata."""
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.baseline_stats = None
        self.metadata = None
        self.feature_names = []
        
        self.load_baseline_model()
    
    def load_baseline_model(self):
        """Load the trained baseline model and associated files."""
        print(f"Loading baseline model from {self.model_dir}...")
        
        # Load TensorFlow model
        model_path = self.model_dir / 'autism_baseline_model.keras'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = keras.models.load_model(model_path)
        print(f"  ✓ Model loaded")
        
        # Load scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"  ✓ Scaler loaded")
        
        # Load baseline statistics
        stats_path = self.model_dir / 'baseline_statistics.json'
        with open(stats_path, 'r') as f:
            self.baseline_stats = json.load(f)
        self.feature_names = self.baseline_stats['feature_names']
        print(f"  ✓ Baseline statistics loaded ({self.baseline_stats['n_participants']} participants)")
        
        # Load metadata
        metadata_path = self.model_dir / 'model_metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"  ✓ Metadata loaded")
        print(f"\n📊 Baseline Model Info:")
        print(f"  Age range: {self.metadata['age_range']}")
        print(f"  CARS range: {self.metadata['cars_range']}")
        print(f"  Features: {self.metadata['n_features']}")
    
    def extract_features(self, data):
        """
        Extract features from new data (same as baseline builder).
        Must match the feature extraction in baseline_model_builder.py
        """
        if len(data) < 2:
            return None
            
        x = data['x'].values
        y = data['y'].values
        
        features = []
        
        # Spatial features
        features.extend([
            np.mean(x), np.std(x), np.min(x), np.max(x),
            np.mean(y), np.std(y), np.min(y), np.max(y),
            np.ptp(x), np.ptp(y),
        ])
        
        # Duration features
        if 'duration' in data.columns:
            dur = data['duration'].values
            features.extend([
                np.mean(dur), np.std(dur), np.median(dur),
                np.percentile(dur, 25), np.percentile(dur, 75)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Timestamp features
        if 'timestamp' in data.columns:
            time = data['timestamp'].values
            time_span = time[-1] - time[0] if len(time) > 1 else 0
            features.extend([
                time_span,
                len(data) / (time_span / 1000) if time_span > 0 else 0
            ])
        else:
            features.extend([0, 0])
        
        # Movement features
        if len(x) > 1:
            dx = np.diff(x)
            dy = np.diff(y)
            distances = np.sqrt(dx**2 + dy**2)
            
            features.extend([
                np.sum(distances),
                np.mean(distances),
                np.std(distances),
            ])
            
            straight_line = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
            efficiency = straight_line / np.sum(distances) if np.sum(distances) > 0 else 0
            features.append(efficiency)
            
            if 'timestamp' in data.columns:
                time = data['timestamp'].values
                dt = np.diff(time)
                dt[dt == 0] = 1
                velocities = distances / dt
                features.extend([
                    np.mean(velocities),
                    np.std(velocities),
                    np.max(velocities)
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # Spatial entropy
        hist_2d, _, _ = np.histogram2d(x, y, bins=10)
        hist_2d = hist_2d.flatten()
        hist_2d = hist_2d[hist_2d > 0]
        if len(hist_2d) > 0:
            probabilities = hist_2d / np.sum(hist_2d)
            entropy = -np.sum(probabilities * np.log2(probabilities))
        else:
            entropy = 0
        features.append(entropy)
        
        # Concentration metrics
        center_x, center_y = np.mean(x), np.mean(y)
        distances_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        features.extend([
            np.mean(distances_from_center),
            np.std(distances_from_center),
            np.percentile(distances_from_center, 50)
        ])
        
        return np.array(features)
    
    def compare_to_baseline(self, data):
        """
        Compare new data against baseline model.
        Returns comprehensive comparison results.
        """
        print("\n🔍 Comparing data to autism baseline...")
        
        # Extract features
        features = self.extract_features(data)
        if features is None:
            return {'error': 'Insufficient data points'}
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Model reconstruction error
        reconstruction = self.model.predict(features_scaled, verbose=0)
        reconstruction_error = np.mean((features_scaled - reconstruction)**2)
        mae = np.mean(np.abs(features_scaled - reconstruction))
        
        print(f"  Reconstruction error (MSE): {reconstruction_error:.4f}")
        print(f"  Mean absolute error (MAE): {mae:.4f}")
        
        # Calculate similarity score (0-100, higher = more similar to baseline)
        # Lower reconstruction error = higher similarity
        max_error = 1.0  # Typical max error for very different patterns
        similarity_score = max(0, min(100, (1 - reconstruction_error / max_error) * 100))
        
        # Z-scores for each feature (how many std deviations from baseline)
        baseline_means = np.array(self.baseline_stats['feature_means'])
        baseline_stds = np.array(self.baseline_stats['feature_stds'])
        z_scores = (features - baseline_means) / (baseline_stds + 1e-10)
        
        # Identify most deviant features
        abs_z_scores = np.abs(z_scores)
        deviant_indices = np.argsort(abs_z_scores)[-5:][::-1]  # Top 5 most deviant
        
        deviant_features = []
        for idx in deviant_indices:
            deviant_features.append({
                'feature': self.feature_names[idx],
                'value': float(features[idx]),
                'baseline_mean': float(baseline_means[idx]),
                'baseline_std': float(baseline_stds[idx]),
                'z_score': float(z_scores[idx]),
                'deviation': 'higher' if z_scores[idx] > 0 else 'lower'
            })
        
        # Overall deviation level
        mean_abs_z = np.mean(abs_z_scores)
        if mean_abs_z < 1.0:
            deviation_level = 'low'
            deviation_interpretation = 'Very similar to baseline autism patterns'
        elif mean_abs_z < 2.0:
            deviation_level = 'moderate'
            deviation_interpretation = 'Moderately different from baseline autism patterns'
        else:
            deviation_level = 'high'
            deviation_interpretation = 'Significantly different from baseline autism patterns'
        
        # Distance metrics
        baseline_center = baseline_means
        euclidean_dist = euclidean(features, baseline_center)
        cosine_sim = 1 - cosine(features, baseline_center)
        
        results = {
            'similarity_score': float(similarity_score),
            'reconstruction_error': float(reconstruction_error),
            'mae': float(mae),
            'deviation_level': deviation_level,
            'deviation_interpretation': deviation_interpretation,
            'mean_absolute_z_score': float(mean_abs_z),
            'euclidean_distance': float(euclidean_dist),
            'cosine_similarity': float(cosine_sim),
            'most_deviant_features': deviant_features,
            'baseline_info': {
                'n_participants': self.baseline_stats['n_participants'],
                'age_range': self.metadata['age_range'],
                'cars_range': self.metadata['cars_range']
            }
        }
        
        print(f"\n📊 Comparison Results:")
        print(f"  Similarity Score: {similarity_score:.1f}/100")
        print(f"  Deviation Level: {deviation_level.upper()}")
        print(f"  {deviation_interpretation}")
        
        return results
    
    def generate_comparison_report(self, data, output_path=None):
        """Generate detailed comparison report."""
        results = self.compare_to_baseline(data)
        
        report = f"""
# 🔬 Eye-Tracking Baseline Comparison Report

## Summary
- **Similarity Score**: {results['similarity_score']:.1f}/100
- **Deviation Level**: {results['deviation_level'].upper()}
- **Interpretation**: {results['deviation_interpretation']}

## Detailed Metrics
- **Reconstruction Error (MSE)**: {results['reconstruction_error']:.4f}
- **Mean Absolute Error**: {results['mae']:.4f}
- **Mean Z-Score**: {results['mean_absolute_z_score']:.2f}
- **Euclidean Distance**: {results['euclidean_distance']:.2f}
- **Cosine Similarity**: {results['cosine_similarity']:.4f}

## Most Deviant Features
The following features show the largest deviations from the baseline:

"""
        
        for i, feat in enumerate(results['most_deviant_features'], 1):
            report += f"""
### {i}. {feat['feature'].replace('_', ' ').title()}
- **Your Value**: {feat['value']:.2f}
- **Baseline Mean**: {feat['baseline_mean']:.2f} ± {feat['baseline_std']:.2f}
- **Z-Score**: {feat['z_score']:.2f} ({feat['deviation']} than baseline)
"""
        
        report += f"""

## Baseline Reference
- **Participants**: {results['baseline_info']['n_participants']} children with ASD
- **Age Range**: {results['baseline_info']['age_range']}
- **CARS Range**: {results['baseline_info']['cars_range']}

## Interpretation Guide
- **Similarity Score** (0-100): Higher scores indicate patterns more similar to typical autism gaze behavior
- **Z-Scores**: Values beyond ±2 indicate significant deviation (95% confidence)
- **Reconstruction Error**: Lower values indicate better match to baseline patterns

---
*Generated by Autism Baseline Comparator*
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(report, encoding='utf-8')
            print(f"\n✅ Report saved to: {output_path}")
        
        return report


def demo_comparison():
    """Demo: Compare a participant against the baseline."""
    print("=" * 70)
    print("🔬 AUTISM BASELINE COMPARISON DEMO")
    print("=" * 70)
    
    from autism_data_loader import AutismDataLoader
    
    # Load a test participant
    loader = AutismDataLoader()
    participants = loader.get_available_participants()
    
    if not participants:
        print("\n❌ No participants found. Please ensure data is in data/autism/")
        return
    
    test_pid = participants[0]
    print(f"\nTesting with participant: {test_pid}")
    
    data = loader.load_participant_data(test_pid, 1920, 1080)
    print(f"Loaded {len(data)} data points")
    
    # Compare
    comparator = BaselineComparator()
    results = comparator.compare_to_baseline(data)
    
    # Generate report
    report = comparator.generate_comparison_report(
        data, 
        output_path=f'comparison_report_{test_pid}.md'
    )
    
    print("\n" + "=" * 70)
    print("✅ Comparison complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo_comparison()
