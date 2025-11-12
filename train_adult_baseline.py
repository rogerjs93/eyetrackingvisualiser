"""
Train Adult ASD Baseline Model from RawEyetrackingASD.mat
This creates a separate baseline for comparison with children model
"""
import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import pickle

class AdultASDModelTrainer:
    """Train baseline model on adult ASD data from .mat file"""
    
    def __init__(self, mat_file_path):
        self.mat_file_path = mat_file_path
        self.scaler = StandardScaler()
        self.model = None
        
    def extract_features_for_participant(self, participant_idx, eye_data):
        """Extract 28 features from one participant's data"""
        # Get all trials for this participant: (36 trials, 2 coords, 14000 samples)
        participant_data = eye_data[:, participant_idx, :, :]
        
        # Combine all trials
        x_coords = participant_data[:, 0, :].flatten()
        y_coords = participant_data[:, 1, :].flatten()
        
        # Filter valid data
        valid_mask = (x_coords > 0) & (y_coords > 0)
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        
        if len(x_valid) == 0:
            return None
        
        # Calculate velocity and acceleration
        dx = np.diff(x_valid)
        dy = np.diff(y_valid)
        velocity = np.sqrt(dx**2 + dy**2)
        acceleration = np.diff(velocity)
        
        # Extract 28 features (same as children model)
        features = []
        
        # 1-4: X coordinate statistics
        features.extend([np.mean(x_valid), np.std(x_valid), np.min(x_valid), np.max(x_valid)])
        
        # 5-8: Y coordinate statistics
        features.extend([np.mean(y_valid), np.std(y_valid), np.min(y_valid), np.max(y_valid)])
        
        # 9-12: Velocity statistics
        features.extend([np.mean(velocity), np.std(velocity), np.max(velocity), np.median(velocity)])
        
        # 13-15: Acceleration statistics
        features.extend([np.mean(acceleration), np.std(acceleration), np.max(np.abs(acceleration))])
        
        # 16-17: Fixation metrics
        fixation_threshold = 50
        fixations = velocity < fixation_threshold
        features.extend([
            np.sum(fixations) / len(velocity),
            np.mean(velocity[fixations]) if np.any(fixations) else 0
        ])
        
        # 18-19: Saccade metrics
        saccade_threshold = 200
        saccades = velocity > saccade_threshold
        features.extend([
            np.sum(saccades) / len(velocity),
            np.mean(velocity[saccades]) if np.any(saccades) else 0
        ])
        
        # 20-23: Gaze distribution
        features.extend([
            np.percentile(x_valid, 25), np.percentile(x_valid, 75),
            np.percentile(y_valid, 25), np.percentile(y_valid, 75)
        ])
        
        # 24-25: Coverage
        features.extend([
            np.sum(velocity),  # Path length
            (np.max(x_valid) - np.min(x_valid)) * (np.max(y_valid) - np.min(y_valid))  # Area
        ])
        
        # 26-27: Velocity variability
        features.extend([np.percentile(velocity, 90), np.percentile(velocity, 10)])
        
        # 28: Sample count
        features.append(len(x_valid))
        
        return np.array(features)
    
    def load_and_prepare_data(self):
        """Load .mat file and extract features from all participants"""
        print("="*70)
        print("Loading Adult ASD Dataset from .mat file")
        print("="*70)
        
        print(f"\n📂 Loading: {self.mat_file_path}")
        mat_data = scipy.io.loadmat(self.mat_file_path)
        eye_data = mat_data['eyeMovementsASD']
        
        print(f"✅ Loaded data shape: {eye_data.shape}")
        print(f"   {eye_data.shape[1]} participants")
        print(f"   {eye_data.shape[0]} trials per participant")
        print(f"   {eye_data.shape[3]} time samples per trial")
        
        print(f"\n🔄 Extracting features from all participants...")
        all_features = []
        metadata = []
        
        for p_idx in range(eye_data.shape[1]):
            print(f"   Participant {p_idx + 1}/{eye_data.shape[1]}...", end=" ")
            features = self.extract_features_for_participant(p_idx, eye_data)
            
            if features is not None:
                all_features.append(features)
                metadata.append({
                    'participant_id': p_idx + 1,
                    'source': 'mat_file_adult'
                })
                print("✓")
            else:
                print("✗ No valid data")
        
        features_array = np.array(all_features)
        print(f"\n✅ Extracted features from {len(all_features)} participants")
        print(f"   Feature matrix shape: {features_array.shape}")
        
        return features_array, metadata
    
    def build_autoencoder(self, input_dim):
        """Build same architecture as children model for comparison"""
        print("\n🏗️  Building adult baseline autoencoder...")
        print("   Architecture: 28→32→48→24→48→32→28 (same as optimized children model)")
        
        # Encoder
        encoder_input = keras.Input(shape=(input_dim,), name='encoder_input')
        x = keras.layers.Dense(32, activation='relu')(encoder_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(48, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)
        latent = keras.layers.Dense(24, activation='relu', name='latent')(x)
        
        # Decoder
        x = keras.layers.Dense(48, activation='relu')(latent)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        decoder_output = keras.layers.Dense(input_dim, activation='linear')(x)
        
        autoencoder = keras.Model(encoder_input, decoder_output, name='adult_asd_autoencoder')
        autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00652), loss='mae')
        
        print(f"   Total parameters: {autoencoder.count_params():,}")
        return autoencoder
    
    def train_model(self, features, epochs=100):
        """Train the adult baseline model"""
        print("\n" + "="*70)
        print("Training Adult ASD Baseline Model")
        print("="*70)
        
        # Normalize
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_val = train_test_split(features_scaled, test_size=0.2, random_state=42)
        print(f"\n   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Build model
        self.model = self.build_autoencoder(features.shape[1])
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train
        print("\n🎯 Training...")
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=4,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        val_loss = self.model.evaluate(X_val, X_val, verbose=0)
        print(f"\n✅ Training complete!")
        print(f"   Final validation MAE: {val_loss:.4f}")
        
        return history, val_loss
    
    def save_model(self, output_dir='models/baseline_adult_asd'):
        """Save the trained model and artifacts"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving adult baseline model to: {output_path}")
        
        # Save model
        model_path = output_path / 'adult_asd_baseline.keras'
        self.model.save(model_path)
        print(f"   ✓ Model: {model_path}")
        
        # Save scaler
        scaler_path = output_path / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✓ Scaler: {scaler_path}")
        
        # Save scaler as JSON
        scaler_json = {
            'mean': self.scaler.mean_.tolist(),
            'std': self.scaler.scale_.tolist()
        }
        scaler_json_path = output_path / 'scaler.json'
        with open(scaler_json_path, 'w') as f:
            json.dump(scaler_json, f, indent=2)
        print(f"   ✓ Scaler JSON: {scaler_json_path}")
        
        print(f"\n✅ Adult baseline model saved!")


def main():
    print("\n" + "="*70)
    print("ADULT ASD BASELINE MODEL TRAINER")
    print("="*70)
    print("\n📚 Dataset: RawEyetrackingASD.mat")
    print("   Age Range: Adult (assuming from .mat file structure)")
    print("   Architecture: Same as optimized children model for comparison")
    print("   Purpose: Compare adult vs children autism gaze patterns")
    
    # Initialize trainer
    mat_file = r"data\autism\autismdata2\RawEyetrackingASD.mat"
    trainer = AdultASDModelTrainer(mat_file)
    
    # Load and extract features
    features, metadata = trainer.load_and_prepare_data()
    
    # Train model
    history, val_mae = trainer.train_model(features, epochs=100)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*70)
    print("✅ ADULT BASELINE MODEL COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Participants: {len(features)}")
    print(f"   Features: {features.shape[1]}")
    print(f"   Validation MAE: {val_mae:.4f}")
    print(f"\n💡 Next steps:")
    print(f"   1. Compare with children model performance")
    print(f"   2. Analyze feature differences between age groups")
    print(f"   3. Deploy to web interface with age selection")


if __name__ == "__main__":
    main()
