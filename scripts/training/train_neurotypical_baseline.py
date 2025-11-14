"""
Train neurotypical baseline model from the ETDB v1.0 dataset
This creates a baseline for healthy individuals (no ASD)
Dataset: https://datadryad.org/dataset/doi:10.5061/dryad.9pf75
"""
import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class NeurotypicalBaselineTrainer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.hdf5_file = self.data_path / 'etdb_v1.0.hdf5'
        
    def extract_features_from_trial(self, x_coords, y_coords):
        """
        Extract 28 features from a single trial
        Same features as ASD models for consistency
        """
        if len(x_coords) < 2:
            return None
            
        # Spatial statistics
        x_mean = np.mean(x_coords)
        x_std = np.std(x_coords)
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        
        y_mean = np.mean(y_coords)
        y_std = np.std(y_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        
        # Fixation duration (simulated - dataset doesn't have explicit fixations)
        fixation_duration_mean = 200.0  # Default
        fixation_duration_std = 50.0
        fixation_count = len(x_coords)
        
        # Velocity calculations
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        velocities = np.sqrt(dx**2 + dy**2)
        saccade_velocity_mean = np.mean(velocities) if len(velocities) > 0 else 0
        saccade_velocity_std = np.std(velocities) if len(velocities) > 0 else 0
        
        # Amplitude calculations
        amplitudes = np.sqrt(dx**2 + dy**2)
        saccade_amplitude_mean = np.mean(amplitudes) if len(amplitudes) > 0 else 0
        saccade_amplitude_std = np.std(amplitudes) if len(amplitudes) > 0 else 0
        
        # Pupil size (not available, use default)
        pupil_size_mean = 3.5
        pupil_size_std = 0.5
        
        # Spatial coverage
        grid_size = 5
        grid_coords = np.floor(np.array([x_coords, y_coords]).T / (100 / grid_size))
        unique_cells = len(np.unique(grid_coords, axis=0))
        spatial_coverage = unique_cells / (grid_size * grid_size)
        
        # Fixation dispersion
        fixation_dispersion = np.sqrt(x_std**2 + y_std**2)
        
        # Scan path length
        scan_path_length = np.sum(np.sqrt(dx**2 + dy**2))
        
        # Gaze entropy
        grid_counts = {}
        for x, y in zip(x_coords, y_coords):
            key = f"{int(x//(100/grid_size))},{int(y//(100/grid_size))}"
            grid_counts[key] = grid_counts.get(key, 0) + 1
        total = len(x_coords)
        gaze_entropy = -sum((count/total) * np.log2(count/total) for count in grid_counts.values())
        
        # Center bias
        center_x, center_y = 50, 50
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        center_bias = 1 / (1 + np.mean(distances))
        
        # Edge bias
        edge_count = np.sum((x_coords < 10) | (x_coords > 90) | (y_coords < 10) | (y_coords > 90))
        edge_bias = edge_count / len(x_coords)
        
        # ROI focus (center region)
        roi_count = np.sum((x_coords >= 30) & (x_coords <= 70) & (y_coords >= 30) & (y_coords <= 70))
        roi_focus = roi_count / len(x_coords)
        
        # Vertical/Horizontal ratio
        v_movement = np.sum(np.abs(dy))
        h_movement = np.sum(np.abs(dx))
        vh_ratio = v_movement / h_movement if h_movement > 0 else 1
        
        # Temporal consistency
        temporal_consistency = 1 / (1 + np.std(np.sqrt(dx**2 + dy**2))) if len(dx) > 0 else 0
        
        # Attention switches
        prev_grid = None
        switches = 0
        for x, y in zip(x_coords, y_coords):
            curr_grid = f"{int(x//(100/3))},{int(y//(100/3))}"
            if prev_grid and curr_grid != prev_grid:
                switches += 1
            prev_grid = curr_grid
        attention_switches = switches
        
        # Revisit rate
        revisits = sum(1 for count in grid_counts.values() if count > 1)
        revisit_rate = revisits / len(grid_counts) if len(grid_counts) > 0 else 0
        
        return np.array([
            x_mean, x_std, x_min, x_max,
            y_mean, y_std, y_min, y_max,
            fixation_duration_mean, fixation_duration_std, fixation_count,
            saccade_velocity_mean, saccade_velocity_std,
            saccade_amplitude_mean, saccade_amplitude_std,
            pupil_size_mean, pupil_size_std,
            spatial_coverage, fixation_dispersion, scan_path_length,
            gaze_entropy, center_bias, edge_bias, roi_focus,
            vh_ratio, temporal_consistency, attention_switches, revisit_rate
        ])
    
    def load_and_extract_features(self, max_trials=1000):
        """
        Load eye-tracking data and extract features
        """
        print(f"📂 Loading data from {self.hdf5_file}")
        
        all_features = []
        
        with h5py.File(self.hdf5_file, 'r') as f:
            # Iterate through datasets
            dataset_names = ['Baseline', 'Age study', 'Memory I', 'Memory II', 'Patch']
            
            for dataset_name in dataset_names:
                if dataset_name not in f:
                    continue
                    
                print(f"\n📊 Processing dataset: {dataset_name}")
                dataset = f[dataset_name]
                
                # Get x and y coordinates
                x_data = dataset['x'][:]
                y_data = dataset['y'][:]
                trial_data = dataset['trial'][:]
                
                # Process unique trials
                unique_trials = np.unique(trial_data)
                print(f"   Found {len(unique_trials)} trials")
                
                for trial_id in unique_trials[:max_trials//len(dataset_names)]:
                    mask = trial_data == trial_id
                    x_coords = x_data[mask]
                    y_coords = y_data[mask]
                    
                    if len(x_coords) < 10:  # Skip very short trials
                        continue
                    
                    # Normalize coordinates to 0-100 range
                    x_coords = (x_coords - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords) + 1e-8) * 100
                    y_coords = (y_coords - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords) + 1e-8) * 100
                    
                    features = self.extract_features_from_trial(x_coords, y_coords)
                    if features is not None:
                        all_features.append(features)
                
                if len(all_features) >= max_trials:
                    break
        
        features_array = np.array(all_features)
        print(f"\n✅ Extracted {len(features_array)} feature vectors")
        print(f"   Feature shape: {features_array.shape}")
        
        return features_array
    
    def train_autoencoder(self, features, output_dir='models/baseline_neurotypical'):
        """
        Train autoencoder model
        Same architecture as ASD models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Split data
        split_idx = int(0.8 * len(features_scaled))
        train_data = features_scaled[:split_idx]
        val_data = features_scaled[split_idx:]
        
        print(f"\n🧠 Training autoencoder...")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        
        # Build autoencoder (same architecture as ASD models)
        input_dim = 28
        
        encoder_input = tf.keras.Input(shape=(input_dim,), name='encoder_input')
        
        # Encoder
        x = tf.keras.layers.Dense(32, activation='relu')(encoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Dense(48, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Latent space
        latent = tf.keras.layers.Dense(24, activation='relu', name='latent')(x)
        
        # Decoder
        x = tf.keras.layers.Dense(48, activation='relu')(latent)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        decoder_output = tf.keras.layers.Dense(input_dim, activation='linear')(x)
        
        model = tf.keras.Model(encoder_input, decoder_output, name='neurotypical_autoencoder')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(model.summary())
        
        # Train
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        history = model.fit(
            train_data, train_data,
            validation_data=(val_data, val_data),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        val_predictions = model.predict(val_data)
        val_mae = np.mean(np.abs(val_data - val_predictions))
        print(f"\n✅ Validation MAE: {val_mae:.4f}")
        
        # Save model
        model_path = output_path / 'neurotypical_baseline.keras'
        model.save(model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Save scaler
        scaler_path = output_path / 'scaler.json'
        with open(scaler_path, 'w') as f:
            json.dump({
                'mean': scaler.mean_.tolist(),
                'std': scaler.scale_.tolist()
            }, f, indent=2)
        print(f"💾 Scaler saved to {scaler_path}")
        
        # Save metadata
        metadata = {
            'dataset': 'ETDB v1.0 - Neurotypical baseline',
            'source': 'https://datadryad.org/dataset/doi:10.5061/dryad.9pf75',
            'n_samples': len(features),
            'n_features': input_dim,
            'architecture': '28-32-48-24-48-32-28',
            'validation_mae': float(val_mae),
            'training_samples': len(train_data),
            'validation_samples': len(val_data)
        }
        
        metadata_path = output_path / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Metadata saved to {metadata_path}")
        
        return model, history, val_mae

def main():
    print("="*60)
    print("NEUROTYPICAL BASELINE MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    data_path = 'data/standard/doi_10_5061_dryad_9pf75__v20171209'
    trainer = NeurotypicalBaselineTrainer(data_path)
    
    # Extract features
    features = trainer.load_and_extract_features(max_trials=1000)
    
    # Train model
    model, history, mae = trainer.train_autoencoder(features)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📊 Neurotypical baseline MAE: {mae:.4f}")
    print(f"📊 Children ASD MAE: 0.4069 (for comparison)")
    print(f"📊 Adult ASD MAE: 0.6065 (for comparison)")

if __name__ == "__main__":
    main()
