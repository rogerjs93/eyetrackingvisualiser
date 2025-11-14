"""
Enhanced Eye-Tracking Baseline Model Training Script
Trains autoencoder with 43 features (28 original + 15 advanced)

Research-focused approach for clinical ASD detection support
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import json
import os

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class EnhancedFeatureExtractor:
    """
    Extract 43 clinically-validated eye-tracking features
    Matches JavaScript implementation in baseline_model_web.js
    """
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, data):
        """
        Extract features from eye-tracking data
        
        Args:
            data: DataFrame or dict with columns: x, y, timestamp, 
                  optional: fixation_duration, pupil_size
        
        Returns:
            numpy array of 43 features
        """
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        features = []
        self.feature_names = []
        
        # Ensure required columns
        if 'fixation_duration' not in data.columns:
            data['fixation_duration'] = 200  # Default 200ms
        if 'pupil_size' not in data.columns:
            data['pupil_size'] = 3.5  # Default size
        
        x = data['x'].values
        y = data['y'].values
        
        # ===== ORIGINAL 28 FEATURES =====
        
        # Spatial statistics (8)
        features.extend([
            np.mean(x), np.std(x), np.min(x), np.max(x),
            np.mean(y), np.std(y), np.min(y), np.max(y)
        ])
        self.feature_names.extend([
            'x_mean', 'x_std', 'x_min', 'x_max',
            'y_mean', 'y_std', 'y_min', 'y_max'
        ])
        
        # Fixation statistics (3)
        fix_dur = data['fixation_duration'].values
        features.extend([
            np.mean(fix_dur), np.std(fix_dur), len(data)
        ])
        self.feature_names.extend([
            'fixation_duration_mean', 'fixation_duration_std', 'fixation_count'
        ])
        
        # Saccade statistics (4)
        velocities = self._calculate_velocities(data)
        amplitudes = self._calculate_amplitudes(data)
        features.extend([
            np.mean(velocities), np.std(velocities),
            np.mean(amplitudes), np.std(amplitudes)
        ])
        self.feature_names.extend([
            'saccade_velocity_mean', 'saccade_velocity_std',
            'saccade_amplitude_mean', 'saccade_amplitude_std'
        ])
        
        # Pupil statistics (2)
        pupil = data['pupil_size'].values
        features.extend([np.mean(pupil), np.std(pupil)])
        self.feature_names.extend(['pupil_size_mean', 'pupil_size_std'])
        
        # Spatial metrics (11)
        features.append(self._spatial_coverage(data))
        self.feature_names.append('spatial_coverage')
        
        features.append(self._fixation_dispersion(data))
        self.feature_names.append('fixation_dispersion')
        
        features.append(self._scan_path_length(data))
        self.feature_names.append('scan_path_length')
        
        features.append(self._gaze_entropy(data))
        self.feature_names.append('gaze_entropy')
        
        features.append(self._center_bias(data))
        self.feature_names.append('center_bias')
        
        features.append(self._edge_bias(data))
        self.feature_names.append('edge_bias')
        
        features.append(self._roi_focus(data))
        self.feature_names.append('roi_focus')
        
        features.append(self._vh_ratio(data))
        self.feature_names.append('vertical_horizontal_ratio')
        
        features.append(self._temporal_consistency(data))
        self.feature_names.append('temporal_consistency')
        
        features.append(self._attention_switches(data))
        self.feature_names.append('attention_switches')
        
        features.append(self._revisit_rate(data))
        self.feature_names.append('revisit_rate')
        
        # ===== NEW 15 ADVANCED FEATURES =====
        
        # Directional entropy
        features.append(self._directional_entropy(data))
        self.feature_names.append('saccade_direction_entropy')
        
        # Spatial autocorrelation
        features.append(self._autocorrelation(x))
        features.append(self._autocorrelation(y))
        self.feature_names.extend(['spatial_autocorr_x', 'spatial_autocorr_y'])
        
        # Cluster density
        features.append(self._cluster_density(data))
        self.feature_names.append('fixation_cluster_density')
        
        # First fixation bias
        features.append(self._first_fixation_bias(data))
        self.feature_names.append('first_fixation_center_dist')
        
        # Spatial revisitation
        features.append(self._spatial_revisitation(data))
        self.feature_names.append('spatial_revisitation_rate')
        
        # Velocity distribution shape
        features.append(stats.skew(velocities))
        features.append(stats.kurtosis(velocities))
        self.feature_names.extend(['velocity_skewness', 'velocity_kurtosis'])
        
        # Inter-saccadic interval variability
        features.append(self._isi_variability(data))
        self.feature_names.append('isi_coefficient_variation')
        
        # Ambient vs focal ratio
        features.append(self._ambient_focal_ratio(data))
        self.feature_names.append('ambient_focal_ratio')
        
        # Amplitude entropy
        features.append(self._amplitude_entropy(amplitudes))
        self.feature_names.append('saccade_amplitude_entropy')
        
        # Scanpath efficiency
        features.append(self._scanpath_efficiency(data))
        self.feature_names.append('scanpath_efficiency')
        
        # Fixation duration entropy
        features.append(self._fixation_duration_entropy(fix_dur))
        self.feature_names.append('fixation_duration_entropy')
        
        # Cross-correlation XY
        features.append(np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0)
        self.feature_names.append('cross_correlation_xy')
        
        # Peak velocity ratio
        features.append(np.max(velocities) / np.mean(velocities) if np.mean(velocities) > 0 else 1)
        self.feature_names.append('peak_velocity_ratio')
        
        return np.array(features)
    
    # ===== HELPER METHODS =====
    
    def _calculate_velocities(self, data):
        velocities = []
        for i in range(1, len(data)):
            dx = data['x'].iloc[i] - data['x'].iloc[i-1]
            dy = data['y'].iloc[i] - data['y'].iloc[i-1]
            dt = (data['timestamp'].iloc[i] - data['timestamp'].iloc[i-1]) / 1000
            dist = np.sqrt(dx**2 + dy**2)
            velocities.append(dist / dt if dt > 0 else 0)
        return np.array(velocities) if velocities else np.array([0])
    
    def _calculate_amplitudes(self, data):
        amplitudes = []
        for i in range(1, len(data)):
            dx = data['x'].iloc[i] - data['x'].iloc[i-1]
            dy = data['y'].iloc[i] - data['y'].iloc[i-1]
            amplitudes.append(np.sqrt(dx**2 + dy**2))
        return np.array(amplitudes) if amplitudes else np.array([0])
    
    def _spatial_coverage(self, data):
        grid_size = 5
        visited = set()
        for _, row in data.iterrows():
            grid_x = int(row['x'] / (100 / grid_size))
            grid_y = int(row['y'] / (100 / grid_size))
            visited.add((grid_x, grid_y))
        return len(visited) / (grid_size * grid_size)
    
    def _fixation_dispersion(self, data):
        return np.sqrt(np.std(data['x'])**2 + np.std(data['y'])**2)
    
    def _scan_path_length(self, data):
        length = 0
        for i in range(1, len(data)):
            dx = data['x'].iloc[i] - data['x'].iloc[i-1]
            dy = data['y'].iloc[i] - data['y'].iloc[i-1]
            length += np.sqrt(dx**2 + dy**2)
        return length
    
    def _gaze_entropy(self, data):
        grid_size = 5
        grid = {}
        for _, row in data.iterrows():
            key = (int(row['x'] / (100/grid_size)), int(row['y'] / (100/grid_size)))
            grid[key] = grid.get(key, 0) + 1
        
        probs = np.array(list(grid.values())) / len(data)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def _center_bias(self, data):
        center_x, center_y = 50, 50
        distances = np.sqrt((data['x'] - center_x)**2 + (data['y'] - center_y)**2)
        return 1 / (1 + np.mean(distances))
    
    def _edge_bias(self, data):
        edge_count = len(data[(data['x'] < 10) | (data['x'] > 90) | 
                              (data['y'] < 10) | (data['y'] > 90)])
        return edge_count / len(data)
    
    def _roi_focus(self, data):
        roi_count = len(data[(data['x'] >= 30) & (data['x'] <= 70) & 
                             (data['y'] >= 30) & (data['y'] <= 70)])
        return roi_count / len(data)
    
    def _vh_ratio(self, data):
        v_movement = np.sum(np.abs(np.diff(data['y'])))
        h_movement = np.sum(np.abs(np.diff(data['x'])))
        return v_movement / h_movement if h_movement > 0 else 1
    
    def _temporal_consistency(self, data):
        distances = []
        for i in range(1, len(data)):
            dx = data['x'].iloc[i] - data['x'].iloc[i-1]
            dy = data['y'].iloc[i] - data['y'].iloc[i-1]
            distances.append(np.sqrt(dx**2 + dy**2))
        return 1 / (1 + np.std(distances)) if distances else 0
    
    def _attention_switches(self, data):
        grid_size = 3
        switches = 0
        for i in range(1, len(data)):
            prev_grid = (int(data['x'].iloc[i-1] / (100/grid_size)), 
                        int(data['y'].iloc[i-1] / (100/grid_size)))
            curr_grid = (int(data['x'].iloc[i] / (100/grid_size)), 
                        int(data['y'].iloc[i] / (100/grid_size)))
            if prev_grid != curr_grid:
                switches += 1
        return switches
    
    def _revisit_rate(self, data):
        grid_size = 5
        visits = {}
        for _, row in data.iterrows():
            key = (int(row['x'] / (100/grid_size)), int(row['y'] / (100/grid_size)))
            visits[key] = visits.get(key, 0) + 1
        
        revisits = sum(1 for count in visits.values() if count > 1)
        return revisits / len(visits) if visits else 0
    
    # ===== NEW ADVANCED FEATURE METHODS =====
    
    def _directional_entropy(self, data):
        if len(data) < 2:
            return 0
        
        angles = []
        for i in range(1, len(data)):
            dx = data['x'].iloc[i] - data['x'].iloc[i-1]
            dy = data['y'].iloc[i] - data['y'].iloc[i-1]
            if dx != 0 or dy != 0:
                angles.append(np.arctan2(dy, dx))
        
        if not angles:
            return 0
        
        # Bin into 8 directions
        bins = np.histogram(angles, bins=8, range=(-np.pi, np.pi))[0]
        probs = (bins + 1e-10) / np.sum(bins)
        return -np.sum(probs * np.log2(probs))
    
    def _autocorrelation(self, values):
        if len(values) < 2:
            return 0
        return np.corrcoef(values[:-1], values[1:])[0, 1]
    
    def _cluster_density(self, data):
        if len(data) < 5:
            return 0
        
        eps = 10  # pixels
        min_points = 3
        visited = set()
        clusters = 0
        
        for i in range(len(data)):
            if i in visited:
                continue
            
            neighbors = []
            for j in range(len(data)):
                if i == j:
                    continue
                dist = np.sqrt((data['x'].iloc[i] - data['x'].iloc[j])**2 +
                              (data['y'].iloc[i] - data['y'].iloc[j])**2)
                if dist < eps:
                    neighbors.append(j)
            
            if len(neighbors) >= min_points:
                clusters += 1
                visited.add(i)
                for n in neighbors:
                    visited.add(n)
        
        return clusters / len(data)
    
    def _first_fixation_bias(self, data):
        if len(data) == 0:
            return 50
        
        center_x, center_y = 50, 50
        first_x = data['x'].iloc[0]
        first_y = data['y'].iloc[0]
        
        dist = np.sqrt((first_x - center_x)**2 + (first_y - center_y)**2)
        max_dist = np.sqrt(50**2 + 50**2)
        
        return dist / max_dist * 100
    
    def _spatial_revisitation(self, data):
        if len(data) < 2:
            return 0
        
        grid_size = 10
        visited = {}
        revisits = 0
        
        for _, row in data.iterrows():
            key = (int(row['x'] / grid_size), int(row['y'] / grid_size))
            if key in visited:
                revisits += 1
            visited[key] = visited.get(key, 0) + 1
        
        return revisits / len(data)
    
    def _isi_variability(self, data):
        if len(data) < 2:
            return 0
        
        intervals = np.abs(np.diff(data['timestamp']))
        intervals = intervals[intervals > 0]
        
        if len(intervals) == 0:
            return 0
        
        return np.std(intervals) / np.mean(intervals)
    
    def _ambient_focal_ratio(self, data):
        durations = data['fixation_duration'].values
        short_fix = np.sum(durations < 200)
        long_fix = np.sum(durations > 400)
        
        return short_fix / long_fix if long_fix > 0 else short_fix
    
    def _amplitude_entropy(self, amplitudes):
        if len(amplitudes) < 2:
            return 0
        
        bins = np.histogram(amplitudes, bins=5)[0]
        probs = (bins + 1e-10) / np.sum(bins)
        return -np.sum(probs * np.log2(probs))
    
    def _scanpath_efficiency(self, data):
        if len(data) < 2:
            return 1
        
        # Straight-line distance
        dx = data['x'].iloc[-1] - data['x'].iloc[0]
        dy = data['y'].iloc[-1] - data['y'].iloc[0]
        straight_dist = np.sqrt(dx**2 + dy**2)
        
        # Actual path length
        actual_dist = self._scan_path_length(data)
        
        return straight_dist / actual_dist if actual_dist > 0 else 0
    
    def _fixation_duration_entropy(self, durations):
        if len(durations) < 2:
            return 0
        
        bins = np.histogram(durations, bins=5)[0]
        probs = (bins + 1e-10) / np.sum(bins)
        return -np.sum(probs * np.log2(probs))


def build_autoencoder(input_dim=43, latent_dim=16):
    """
    Build enhanced autoencoder for ASD baseline detection
    
    Args:
        input_dim: Number of input features (43)
        latent_dim: Size of latent representation
    
    Returns:
        Compiled Keras model
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(64, activation='relu')(encoder_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    encoded = keras.layers.Dense(latent_dim, activation='relu', name='latent')(x)
    
    # Decoder
    x = keras.layers.Dense(32, activation='relu')(encoded)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    decoder_output = keras.layers.Dense(input_dim, activation='linear')(x)
    
    # Build model
    autoencoder = keras.Model(encoder_input, decoder_output, name='enhanced_autoencoder')
    
    # Compile
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mae']
    )
    
    return autoencoder


def train_model(X_train, X_val, model_name='enhanced_baseline', epochs=100):
    """
    Train autoencoder with early stopping and model checkpointing
    
    Args:
        X_train: Training features
        X_val: Validation features
        model_name: Name for saving model
        epochs: Maximum training epochs
    
    Returns:
        Trained model, training history, validation MAE
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Build model
    model = build_autoencoder(input_dim=X_train.shape[1])
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=4,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_mae = model.evaluate(X_val, X_val, verbose=0)[1]
    print(f"\n✅ Final Validation MAE: {val_mae:.4f}")
    
    # CRITICAL: Save model immediately after training
    model_save_dir = f'models/{model_name}'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'model.keras')
    model.save(model_save_path)
    print(f"💾 Model saved to: {model_save_path}")
    
    return model, history, val_mae


def export_to_tfjs(model, scaler, output_dir, model_name):
    """
    Export model and scaler to TensorFlow.js format
    
    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        output_dir: Output directory path
        model_name: Name for model files
    """
    import tensorflowjs as tfjs
    
    # Create output directory
    model_dir = os.path.join(output_dir, f'{model_name}_tfjs')
    os.makedirs(model_dir, exist_ok=True)
    
    # Export model
    tfjs.converters.save_keras_model(model, model_dir)
    print(f"✅ Model exported to: {model_dir}")
    
    # Export scaler
    scaler_dict = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()
    }
    
    scaler_path = os.path.join(model_dir, 'scaler.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler_dict, f, indent=2)
    
    print(f"✅ Scaler exported to: {scaler_path}")
    
    # Calculate model size
    total_size = sum(
        os.path.getsize(os.path.join(model_dir, f)) 
        for f in os.listdir(model_dir)
    )
    print(f"📦 Total model size: {total_size / 1024:.1f} KB")


def main():
    """
    Main training script
    
    TODO: Replace with your actual dataset loading
    This is a template - you need to load your ASD datasets
    """
    print("🚀 Enhanced Eye-Tracking Model Training")
    print("="*60)
    
    # ===== STEP 1: LOAD YOUR DATA =====
    # Using real ASD dataset prepared from Eye-tracking Output
    
    # ===== STEP 1: LOAD PREPARED DATA =====
    print(f"\n{'='*60}")
    print("Loading Prepared Dataset (Real Children ASD Data)")
    print(f"{'='*60}")
    
    # Check for prepared data
    prepared_data_path = 'data/prepared/children_asd_43features.npy'
    metadata_path = 'data/prepared/children_asd_43features_metadata.json'
    
    if not os.path.exists(prepared_data_path):
        print(f"❌ Error: Prepared data not found at {prepared_data_path}")
        print("\nPlease run data preparation first:")
        print('  python prepare_training_data.py "data/autism/Eye-tracking Output" data/prepared/children_asd_43features.npy children_asd')
        return
    
    # Load feature matrix
    X = np.load(prepared_data_path)
    print(f"✅ Loaded feature matrix: {X.shape}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Dataset: {metadata.get('group_label', 'children_asd')}")
    print(f"👥 Samples: {metadata['n_samples']}")
    print(f"🔢 Features: {metadata['n_features']}")
    print(f"📋 Feature names: {len(metadata['feature_names'])} features")
    
    # Validate feature count
    if X.shape[1] != 43:
        print(f"⚠️ WARNING: Expected 43 features, got {X.shape[1]}")
    
    # Check for NaN or Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print(f"⚠️ WARNING: Found NaN or Inf values in features. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ===== STEP 2: PREPROCESS =====
    print(f"\n{'='*60}")
    print("Preprocessing Data")
    print(f"{'='*60}")
    
    # Split data
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Validation set: {X_val_scaled.shape}")
    
    # ===== STEP 3: TRAIN MODEL =====
    model, history, val_mae = train_model(
        X_train_scaled, 
        X_val_scaled,
        model_name='children_asd_enhanced',
        epochs=150
    )
    
    # ===== SAVE SCALER =====
    print(f"\n{'='*60}")
    print("Saving Scaler")
    print(f"{'='*60}")
    
    scaler_path = 'models/children_asd_enhanced/scaler.pkl'
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to: {scaler_path}")
    
    # ===== STEP 4: EXPORT TO TFJS (Optional - may fail due to package issues) =====
    print(f"\n{'='*60}")
    print("Attempting TensorFlow.js Export")
    print(f"{'='*60}")
    
    try:
        export_to_tfjs(
            model, 
            scaler, 
            output_dir='models/ACTIVE',
            model_name='children_asd_v2_tfjs'
        )
        print("✅ TensorFlow.js export successful!")
    except Exception as e:
        print(f"⚠️  TensorFlow.js export failed: {e}")
        print("   Model and scaler are saved in Keras format.")
        print("   Use export_model.py to manually convert to TFJS format.")
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Baseline MAE (28 features): 0.4069")
    
    if val_mae < 0.4069:
        improvement = ((0.4069 - val_mae) / 0.4069) * 100
        print(f"✅ Improvement: {improvement:.1f}% better!")
    
    print("\n📊 Performance Summary:")
    print(f"  Old model (28 features): MAE = 0.4069")
    print(f"  New model (43 features): MAE = {val_mae:.4f}")
    
    print("\nNext steps:")
    print("1. Update baseline_model_web.js modelPath to 'models/ACTIVE/children_asd_v2_tfjs/model.json'")
    print("2. Test in browser with real CSV data")
    print("3. Deploy to GitHub Pages")


if __name__ == '__main__':
    main()
