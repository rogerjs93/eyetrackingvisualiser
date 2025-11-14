"""
Baseline Model Builder for Autism Eye-Tracking Data
Creates a reference model from all 25 participants to use as comparison baseline.
Uses TensorFlow to learn typical autism gaze patterns.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json
from pathlib import Path
from autism_data_loader import AutismDataLoader

class BaselineModelBuilder:
    """Build and train baseline model from all autism participants."""
    
    def __init__(self):
        self.loader = AutismDataLoader()
        self.model = None
        self.scaler = None
        self.baseline_statistics = {}
        self.feature_names = []
        
    def extract_features(self, data):
        """
        Extract comprehensive features from eye-tracking data.
        
        Features include:
        - Spatial statistics (mean, std, range of x, y)
        - Temporal patterns (fixation durations, saccade velocities)
        - Movement characteristics (path length, efficiency)
        - Distribution metrics (entropy, spread)
        """
        if len(data) < 2:
            return None
            
        x = data['x'].values
        y = data['y'].values
        
        features = []
        feature_names = []
        
        # Spatial features
        features.extend([
            np.mean(x), np.std(x), np.min(x), np.max(x),
            np.mean(y), np.std(y), np.min(y), np.max(y),
            np.ptp(x),  # x range
            np.ptp(y),  # y range
        ])
        feature_names.extend([
            'x_mean', 'x_std', 'x_min', 'x_max',
            'y_mean', 'y_std', 'y_min', 'y_max',
            'x_range', 'y_range'
        ])
        
        # Duration features (if available)
        if 'duration' in data.columns:
            dur = data['duration'].values
            features.extend([
                np.mean(dur), np.std(dur), np.median(dur),
                np.percentile(dur, 25), np.percentile(dur, 75)
            ])
            feature_names.extend([
                'duration_mean', 'duration_std', 'duration_median',
                'duration_q25', 'duration_q75'
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
            feature_names.extend([
                'duration_mean', 'duration_std', 'duration_median',
                'duration_q25', 'duration_q75'
            ])
        
        # Timestamp features
        if 'timestamp' in data.columns:
            time = data['timestamp'].values
            time_span = time[-1] - time[0] if len(time) > 1 else 0
            features.extend([
                time_span,
                len(data) / (time_span / 1000) if time_span > 0 else 0  # points per second
            ])
            feature_names.extend(['time_span', 'points_per_second'])
        else:
            features.extend([0, 0])
            feature_names.extend(['time_span', 'points_per_second'])
        
        # Movement features
        if len(x) > 1:
            dx = np.diff(x)
            dy = np.diff(y)
            distances = np.sqrt(dx**2 + dy**2)
            
            features.extend([
                np.sum(distances),  # Total path length
                np.mean(distances),  # Mean step distance
                np.std(distances),   # Step distance variability
            ])
            feature_names.extend(['path_length', 'mean_distance', 'std_distance'])
            
            # Path efficiency (straight line / actual path)
            straight_line = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
            efficiency = straight_line / np.sum(distances) if np.sum(distances) > 0 else 0
            features.append(efficiency)
            feature_names.append('path_efficiency')
            
            # Velocity features
            if 'timestamp' in data.columns:
                dt = np.diff(time)
                dt[dt == 0] = 1
                velocities = distances / dt
                features.extend([
                    np.mean(velocities),
                    np.std(velocities),
                    np.max(velocities)
                ])
                feature_names.extend(['velocity_mean', 'velocity_std', 'velocity_max'])
            else:
                features.extend([0, 0, 0])
                feature_names.extend(['velocity_mean', 'velocity_std', 'velocity_max'])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])
            feature_names.extend([
                'path_length', 'mean_distance', 'std_distance', 'path_efficiency',
                'velocity_mean', 'velocity_std', 'velocity_max'
            ])
        
        # Distribution features (spatial entropy)
        hist_2d, _, _ = np.histogram2d(x, y, bins=10)
        hist_2d = hist_2d.flatten()
        hist_2d = hist_2d[hist_2d > 0]
        if len(hist_2d) > 0:
            probabilities = hist_2d / np.sum(hist_2d)
            entropy = -np.sum(probabilities * np.log2(probabilities))
        else:
            entropy = 0
        features.append(entropy)
        feature_names.append('spatial_entropy')
        
        # Concentration metrics (how focused vs scattered)
        center_x, center_y = np.mean(x), np.mean(y)
        distances_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        features.extend([
            np.mean(distances_from_center),
            np.std(distances_from_center),
            np.percentile(distances_from_center, 50)  # Median distance from center
        ])
        feature_names.extend(['center_distance_mean', 'center_distance_std', 'center_distance_median'])
        
        self.feature_names = feature_names
        return np.array(features)
    
    def load_all_participants(self, screen_width=1920, screen_height=1080):
        """Load data from all 25 participants."""
        print("Loading all participant data...")
        all_features = []
        all_metadata = []
        participants = self.loader.get_available_participants()
        
        for i, pid in enumerate(participants, 1):
            try:
                data = self.loader.load_participant_data(pid, screen_width, screen_height)
                info = self.loader.get_participant_info(pid)
                
                features = self.extract_features(data)
                if features is not None:
                    all_features.append(features)
                    all_metadata.append({
                        'participant_id': pid,
                        'age': info.get('Age', 0),
                        'gender': info.get('Gender', 'Unknown'),
                        'cars_score': info.get('CARS Score', 0)
                    })
                    print(f"  ✓ Loaded participant {pid} ({i}/{len(participants)})")
            except Exception as e:
                print(f"  ✗ Error loading participant {pid}: {e}")
        
        print(f"\n✅ Successfully loaded {len(all_features)} participants")
        return np.array(all_features), all_metadata
    
    def calculate_baseline_statistics(self, features, metadata):
        """Calculate statistical baseline from all participants."""
        print("\nCalculating baseline statistics...")
        
        self.baseline_statistics = {
            'n_participants': len(features),
            'feature_means': np.mean(features, axis=0).tolist(),
            'feature_stds': np.std(features, axis=0).tolist(),
            'feature_mins': np.min(features, axis=0).tolist(),
            'feature_maxs': np.max(features, axis=0).tolist(),
            'feature_medians': np.median(features, axis=0).tolist(),
            'feature_names': self.feature_names,
            'age_stats': {
                'mean': np.mean([m['age'] for m in metadata]),
                'std': np.std([m['age'] for m in metadata]),
                'min': np.min([m['age'] for m in metadata]),
                'max': np.max([m['age'] for m in metadata])
            },
            'cars_stats': {
                'mean': np.mean([m['cars_score'] for m in metadata]),
                'std': np.std([m['cars_score'] for m in metadata]),
                'min': np.min([m['cars_score'] for m in metadata]),
                'max': np.max([m['cars_score'] for m in metadata])
            }
        }
        
        print(f"  Baseline age: {self.baseline_statistics['age_stats']['mean']:.1f} ± {self.baseline_statistics['age_stats']['std']:.1f} years")
        print(f"  Baseline CARS: {self.baseline_statistics['cars_stats']['mean']:.1f} ± {self.baseline_statistics['cars_stats']['std']:.1f}")
        
        return self.baseline_statistics
    
    def build_autoencoder_model(self, input_dim):
        """
        Build autoencoder model to learn normal autism gaze patterns.
        The model learns to reconstruct typical patterns and can detect anomalies.
        """
        print("\nBuilding autoencoder model...")
        
        # Encoder
        encoder_input = keras.Input(shape=(input_dim,), name='encoder_input')
        x = keras.layers.Dense(64, activation='relu', name='encoder_1')(encoder_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(32, activation='relu', name='encoder_2')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        latent = keras.layers.Dense(16, activation='relu', name='latent')(x)
        
        # Decoder
        x = keras.layers.Dense(32, activation='relu', name='decoder_1')(latent)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation='relu', name='decoder_2')(x)
        x = keras.layers.BatchNormalization()(x)
        decoder_output = keras.layers.Dense(input_dim, activation='linear', name='decoder_output')(x)
        
        # Full autoencoder
        autoencoder = keras.Model(encoder_input, decoder_output, name='baseline_autoencoder')
        
        # Compile
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"  Model architecture:")
        print(f"    Input: {input_dim} features")
        print(f"    Encoder: 64 → 32 → 16 (latent)")
        print(f"    Decoder: 32 → 64 → {input_dim} (output)")
        print(f"    Total parameters: {autoencoder.count_params():,}")
        
        return autoencoder
    
    def train_model(self, features, epochs=100, validation_split=0.2):
        """Train the baseline model."""
        print("\nTraining baseline model...")
        
        # Normalize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_val = train_test_split(features_scaled, test_size=validation_split, random_state=42)
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        # Build model
        self.model = self.build_autoencoder_model(features.shape[1])
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            X_train, X_train,  # Autoencoder reconstructs input
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=4,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Final evaluation
        final_loss = self.model.evaluate(X_val, X_val, verbose=0)
        print(f"\n✅ Training complete!")
        print(f"  Final validation loss: {final_loss[0]:.4f}")
        print(f"  Final validation MAE: {final_loss[1]:.4f}")
        
        return history
    
    def save_baseline_model(self, output_dir='models/baseline'):
        """Save the trained model and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving baseline model to {output_path}...")
        
        # Save TensorFlow model
        model_path = output_path / 'autism_baseline_model.keras'
        self.model.save(model_path)
        print(f"  ✓ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = output_path / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  ✓ Scaler saved: {scaler_path}")
        
        # Save baseline statistics
        stats_path = output_path / 'baseline_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(self.baseline_statistics, f, indent=2)
        print(f"  ✓ Statistics saved: {stats_path}")
        
        # Save metadata
        metadata_path = output_path / 'model_metadata.json'
        metadata = {
            'model_type': 'autoencoder',
            'n_participants': self.baseline_statistics['n_participants'],
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_date': pd.Timestamp.now().isoformat(),
            'description': 'Baseline autism gaze pattern model trained on 25 ASD participants',
            'usage': 'Compare new eye-tracking data against this baseline to detect deviations',
            'age_range': f"{self.baseline_statistics['age_stats']['min']:.1f}-{self.baseline_statistics['age_stats']['max']:.1f} years",
            'cars_range': f"{self.baseline_statistics['cars_stats']['min']:.1f}-{self.baseline_statistics['cars_stats']['max']:.1f}"
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Metadata saved: {metadata_path}")
        
        print(f"\n✅ Baseline model successfully saved!")
        print(f"\nModel files:")
        print(f"  • {model_path.name} - TensorFlow/Keras model")
        print(f"  • {scaler_path.name} - Feature scaler")
        print(f"  • {stats_path.name} - Statistical baseline")
        print(f"  • {metadata_path.name} - Model information")
        
        return output_path


def main():
    """Build and save baseline model."""
    print("=" * 70)
    print("🧠 AUTISM BASELINE MODEL BUILDER")
    print("=" * 70)
    print("\nThis script builds a reference model from all 25 autism participants")
    print("that can be used to compare new eye-tracking data against.")
    print()
    
    # Initialize builder
    builder = BaselineModelBuilder()
    
    # Load all participant data
    features, metadata = builder.load_all_participants()
    
    if len(features) < 5:
        print("\n❌ Error: Not enough participants loaded. Need at least 5.")
        return
    
    # Calculate baseline statistics
    baseline_stats = builder.calculate_baseline_statistics(features, metadata)
    
    # Train model
    history = builder.train_model(features, epochs=200, validation_split=0.2)
    
    # Save everything
    output_path = builder.save_baseline_model()
    
    print("\n" + "=" * 70)
    print("🎉 BASELINE MODEL READY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Use 'baseline_comparator.py' to compare new data against this baseline")
    print("  2. The model will calculate similarity scores and detect anomalies")
    print("  3. Push the model to GitHub for sharing with researchers")
    print()


if __name__ == '__main__':
    main()
