"""
Advanced Baseline Model Trainer - Multiple Approaches
Compares different model architectures and techniques for autism baseline
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

class AdvancedBaselineTrainer:
    """
    Train multiple baseline models and compare performance
    """
    
    def __init__(self, data_dir='data/autism', output_dir='models/baseline_advanced'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.features = None
        self.scaler = StandardScaler()
        self.results = {}
    
    def load_and_prepare_data(self):
        """Load autism dataset and extract features"""
        print("=" * 70)
        print("Loading Autism Dataset")
        print("=" * 70)
        
        # Import from existing baseline_model_builder
        from baseline_model_builder import BaselineModelBuilder
        builder = BaselineModelBuilder(str(self.data_dir))
        
        print("\n1️⃣ Loading participants...")
        data_files, metadata = builder.load_autism_data()
        
        print(f"\n2️⃣ Extracting features from {len(data_files)} participants...")
        features = []
        for csv_file, meta in zip(data_files, metadata):
            feature_vector, _ = builder.extract_features(csv_file)
            features.append(feature_vector)
        
        self.features = np.array(features)
        print(f"  ✅ Features shape: {self.features.shape}")
        
        # Normalize
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        return self.features_scaled
    
    # ========================
    # Approach 1: Optimized Autoencoder
    # ========================
    
    def build_tuned_autoencoder(self, hp):
        """Build autoencoder with hyperparameter tuning"""
        input_dim = self.features.shape[1]
        
        # Tunable hyperparameters
        encoder_dim_1 = hp.Int('encoder_1', min_value=32, max_value=128, step=32)
        encoder_dim_2 = hp.Int('encoder_2', min_value=16, max_value=64, step=16)
        latent_dim = hp.Int('latent', min_value=8, max_value=24, step=4)
        dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.4, step=0.1)
        learning_rate = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
        
        # Encoder
        encoder_input = keras.Input(shape=(input_dim,), name='encoder_input')
        x = keras.layers.Dense(encoder_dim_1, activation='relu')(encoder_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(encoder_dim_2, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        latent = keras.layers.Dense(latent_dim, activation='relu', name='latent')(x)
        
        # Decoder
        x = keras.layers.Dense(encoder_dim_2, activation='relu')(latent)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(encoder_dim_1, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        decoder_output = keras.layers.Dense(input_dim, activation='linear')(x)
        
        # Model
        autoencoder = keras.Model(encoder_input, decoder_output)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    def train_optimized_autoencoder(self):
        """Train autoencoder with hyperparameter search"""
        print("\n" + "=" * 70)
        print("Approach 1: Optimized Autoencoder with Hyperparameter Tuning")
        print("=" * 70)
        
        # Keras Tuner
        tuner = kt.BayesianOptimization(
            self.build_tuned_autoencoder,
            objective='val_mae',
            max_trials=15,
            directory=self.output_dir,
            project_name='autoencoder_tuning'
        )
        
        # Split data
        split = int(0.8 * len(self.features_scaled))
        X_train = self.features_scaled[:split]
        X_val = self.features_scaled[split:]
        
        print(f"\n🔍 Searching for best hyperparameters (15 trials)...")
        tuner.search(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=50,
            batch_size=4,
            callbacks=[keras.callbacks.EarlyStopping(patience=10)],
            verbose=0
        )
        
        # Get best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print(f"\n✅ Best hyperparameters found:")
        print(f"  Encoder 1: {best_hp.get('encoder_1')}")
        print(f"  Encoder 2: {best_hp.get('encoder_2')}")
        print(f"  Latent: {best_hp.get('latent')}")
        print(f"  Dropout: {best_hp.get('dropout')}")
        print(f"  Learning Rate: {best_hp.get('lr'):.6f}")
        
        # Evaluate
        val_mae = best_model.evaluate(X_val, X_val, verbose=0)[1]
        self.results['optimized_autoencoder'] = {
            'model': best_model,
            'val_mae': val_mae,
            'hyperparameters': {
                'encoder_1': int(best_hp.get('encoder_1')),
                'encoder_2': int(best_hp.get('encoder_2')),
                'latent': int(best_hp.get('latent')),
                'dropout': float(best_hp.get('dropout')),
                'lr': float(best_hp.get('lr'))
            }
        }
        
        print(f"  Validation MAE: {val_mae:.4f}")
        
        # Save model
        best_model.save(self.output_dir / 'optimized_autoencoder.keras')
        
        return best_model, val_mae
    
    # ========================
    # Approach 2: Ensemble Autoencoder
    # ========================
    
    def train_ensemble_autoencoder(self, n_models=5):
        """Train ensemble of autoencoders"""
        print("\n" + "=" * 70)
        print(f"Approach 2: Ensemble of {n_models} Autoencoders")
        print("=" * 70)
        
        models = []
        val_maes = []
        
        # Split data
        split = int(0.8 * len(self.features_scaled))
        X_train = self.features_scaled[:split]
        X_val = self.features_scaled[split:]
        
        for i in range(n_models):
            print(f"\n🔄 Training model {i+1}/{n_models}...")
            
            # Build model with different initialization
            model = self.build_simple_autoencoder()
            
            # Train with different random seed
            np.random.seed(i * 42)
            tf.random.set_seed(i * 42)
            
            history = model.fit(
                X_train, X_train,
                validation_data=(X_val, X_val),
                epochs=100,
                batch_size=4,
                callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
                verbose=0
            )
            
            val_mae = min(history.history['val_mae'])
            val_maes.append(val_mae)
            models.append(model)
            
            print(f"  Model {i+1} MAE: {val_mae:.4f}")
        
        # Ensemble evaluation
        ensemble_predictions = []
        for model in models:
            pred = model.predict(X_val, verbose=0)
            ensemble_predictions.append(pred)
        
        ensemble_pred = np.mean(ensemble_predictions, axis=0)
        ensemble_mae = np.mean(np.abs(X_val - ensemble_pred))
        
        print(f"\n✅ Ensemble Results:")
        print(f"  Individual MAEs: {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
        print(f"  Ensemble MAE: {ensemble_mae:.4f} (improvement: {((np.mean(val_maes) - ensemble_mae) / np.mean(val_maes) * 100):.1f}%)")
        
        self.results['ensemble_autoencoder'] = {
            'models': models,
            'individual_maes': val_maes,
            'ensemble_mae': ensemble_mae
        }
        
        # Save models
        for i, model in enumerate(models):
            model.save(self.output_dir / f'ensemble_model_{i}.keras')
        
        return models, ensemble_mae
    
    def build_simple_autoencoder(self):
        """Build standard autoencoder"""
        input_dim = self.features.shape[1]
        
        encoder_input = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(64, activation='relu')(encoder_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        latent = keras.layers.Dense(16, activation='relu')(x)
        
        x = keras.layers.Dense(32, activation='relu')(latent)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        decoder_output = keras.layers.Dense(input_dim, activation='linear')(x)
        
        model = keras.Model(encoder_input, decoder_output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    # ========================
    # Approach 3: Isolation Forest
    # ========================
    
    def train_isolation_forest(self):
        """Train Isolation Forest for anomaly detection"""
        print("\n" + "=" * 70)
        print("Approach 3: Isolation Forest")
        print("=" * 70)
        
        # Train on all data (no labels needed)
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # Assume 10% outliers
            random_state=42
        )
        
        iso_forest.fit(self.features_scaled)
        
        # Get anomaly scores
        scores = iso_forest.score_samples(self.features_scaled)
        
        print(f"\n✅ Isolation Forest trained")
        print(f"  Mean anomaly score: {np.mean(scores):.4f}")
        print(f"  Std anomaly score: {np.std(scores):.4f}")
        
        self.results['isolation_forest'] = {
            'model': iso_forest,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
        
        # Save model
        with open(self.output_dir / 'isolation_forest.pkl', 'wb') as f:
            pickle.dump(iso_forest, f)
        
        return iso_forest
    
    # ========================
    # Approach 4: One-Class SVM
    # ========================
    
    def train_one_class_svm(self):
        """Train One-Class SVM"""
        print("\n" + "=" * 70)
        print("Approach 4: One-Class SVM")
        print("=" * 70)
        
        svm = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.1  # Expected fraction of outliers
        )
        
        svm.fit(self.features_scaled)
        
        # Get decision scores
        scores = svm.decision_function(self.features_scaled)
        
        print(f"\n✅ One-Class SVM trained")
        print(f"  Mean decision score: {np.mean(scores):.4f}")
        print(f"  Std decision score: {np.std(scores):.4f}")
        
        self.results['one_class_svm'] = {
            'model': svm,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
        
        # Save model
        with open(self.output_dir / 'one_class_svm.pkl', 'wb') as f:
            pickle.dump(svm, f)
        
        return svm
    
    # ========================
    # Approach 5: Feature Selection + Autoencoder
    # ========================
    
    def train_with_feature_selection(self, k_features=20):
        """Train autoencoder with selected features"""
        print("\n" + "=" * 70)
        print(f"Approach 5: Feature Selection (Top {k_features} features)")
        print("=" * 70)
        
        # Select top k features using mutual information
        selector = SelectKBest(score_func=mutual_info_regression, k=k_features)
        
        # Use reconstruction error as target for feature selection
        dummy_model = self.build_simple_autoencoder()
        dummy_model.fit(self.features_scaled, self.features_scaled, epochs=10, verbose=0)
        recon_errors = np.mean(np.abs(self.features_scaled - dummy_model.predict(self.features_scaled, verbose=0)), axis=1)
        
        X_selected = selector.fit_transform(self.features_scaled, recon_errors)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        print(f"\n✅ Selected {k_features} most informative features")
        print(f"  Feature indices: {selected_indices.tolist()}")
        
        # Train autoencoder on selected features
        split = int(0.8 * len(X_selected))
        X_train = X_selected[:split]
        X_val = X_selected[split:]
        
        # Build smaller autoencoder
        input_dim = k_features
        encoder_input = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(32, activation='relu')(encoder_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        latent = keras.layers.Dense(8, activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(latent)
        x = keras.layers.BatchNormalization()(x)
        decoder_output = keras.layers.Dense(input_dim, activation='linear')(x)
        
        model = keras.Model(encoder_input, decoder_output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        history = model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=100,
            batch_size=4,
            callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=0
        )
        
        val_mae = min(history.history['val_mae'])
        
        print(f"  Validation MAE: {val_mae:.4f}")
        print(f"  Parameters: {model.count_params()} (reduced from original)")
        
        self.results['feature_selection'] = {
            'model': model,
            'selector': selector,
            'selected_indices': selected_indices.tolist(),
            'val_mae': val_mae
        }
        
        # Save
        model.save(self.output_dir / 'feature_selected_autoencoder.keras')
        with open(self.output_dir / 'feature_selector.pkl', 'wb') as f:
            pickle.dump(selector, f)
        
        return model, selector, val_mae
    
    # ========================
    # Compare All Approaches
    # ========================
    
    def compare_all_approaches(self):
        """Train and compare all approaches"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("=" * 80)
        
        # Load data
        self.load_and_prepare_data()
        
        # Train all models
        try:
            self.train_optimized_autoencoder()
        except Exception as e:
            print(f"⚠️ Optimized autoencoder failed: {e}")
        
        try:
            self.train_ensemble_autoencoder(n_models=5)
        except Exception as e:
            print(f"⚠️ Ensemble failed: {e}")
        
        try:
            self.train_isolation_forest()
        except Exception as e:
            print(f"⚠️ Isolation forest failed: {e}")
        
        try:
            self.train_one_class_svm()
        except Exception as e:
            print(f"⚠️ SVM failed: {e}")
        
        try:
            self.train_with_feature_selection(k_features=20)
        except Exception as e:
            print(f"⚠️ Feature selection failed: {e}")
        
        # Summary
        self.print_comparison_summary()
    
    def print_comparison_summary(self):
        """Print comparison of all models"""
        print("\n" + "=" * 80)
        print("📊 FINAL COMPARISON")
        print("=" * 80)
        
        print("\n| Approach                      | Validation MAE | Notes                    |")
        print("|-------------------------------|----------------|--------------------------|")
        
        if 'optimized_autoencoder' in self.results:
            mae = self.results['optimized_autoencoder']['val_mae']
            print(f"| Optimized Autoencoder         | {mae:.4f}      | Hyperparameter tuned     |")
        
        if 'ensemble_autoencoder' in self.results:
            mae = self.results['ensemble_autoencoder']['ensemble_mae']
            print(f"| Ensemble Autoencoder (5)      | {mae:.4f}      | Most robust              |")
        
        if 'feature_selection' in self.results:
            mae = self.results['feature_selection']['val_mae']
            print(f"| Feature Selection (20 feat)   | {mae:.4f}      | Smaller model            |")
        
        if 'isolation_forest' in self.results:
            print(f"| Isolation Forest              | N/A            | No reconstruction        |")
        
        if 'one_class_svm' in self.results:
            print(f"| One-Class SVM                 | N/A            | No reconstruction        |")
        
        print("\n✅ All models saved to:", self.output_dir)
        
        # Save comparison results
        results_summary = {}
        for name, result in self.results.items():
            if isinstance(result, dict):
                summary = {k: v for k, v in result.items() if k != 'model' and k != 'models' and k != 'selector'}
                results_summary[name] = summary
        
        with open(self.output_dir / 'comparison_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"📄 Results summary: {self.output_dir / 'comparison_results.json'}")

if __name__ == '__main__':
    trainer = AdvancedBaselineTrainer()
    trainer.compare_all_approaches()
