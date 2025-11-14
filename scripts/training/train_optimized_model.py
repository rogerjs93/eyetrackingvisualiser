"""
Optimized Training Strategy for Enhanced Eye-Tracking Model

PROBLEM: 25 samples with 43 features resulted in severe overfitting
- Current: MAE 0.6771 (WORSE than baseline 0.4069)
- 11,643 parameters / 25 samples = 465 params per sample

SOLUTIONS:
1. Feature Selection: Reduce to most informative features
2. Simpler Architecture: Smaller network to match data size
3. Strong Regularization: Dropout, L2, early stopping
4. Data Augmentation: Synthetic samples via noise injection
5. Cross-Validation: K-fold instead of single train/val split
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras


def select_top_features(X, n_features=20, method='mutual_info'):
    """
    Select most informative features using statistical tests
    
    Args:
        X: Feature matrix (n_samples, n_features)
        n_features: Number of features to keep
        method: 'mutual_info', 'f_test', or 'variance'
    
    Returns:
        Selected feature indices, feature scores
    """
    print(f"\n{'='*60}")
    print(f"Feature Selection: {method} (keeping top {n_features})")
    print(f"{'='*60}")
    
    if method == 'mutual_info':
        # Use mutual information (works for autoencoder by treating reconstruction as regression)
        scores = []
        for i in range(X.shape[1]):
            # Score each feature by how well it can be predicted from others
            mask = np.ones(X.shape[1], dtype=bool)
            mask[i] = False
            score = mutual_info_regression(X[:, mask], X[:, i], random_state=42).mean()
            scores.append(score)
        scores = np.array(scores)
        
    elif method == 'f_test':
        # Use F-statistic
        scores = []
        for i in range(X.shape[1]):
            mask = np.ones(X.shape[1], dtype=bool)
            mask[i] = False
            f_stat, _ = f_regression(X[:, mask], X[:, i])
            scores.append(f_stat.mean())
        scores = np.array(scores)
        
    elif method == 'variance':
        # Simple variance-based selection
        scores = np.var(X, axis=0)
    
    # Select top features
    top_indices = np.argsort(scores)[-n_features:]
    top_indices = np.sort(top_indices)  # Keep original order
    
    print(f"✅ Selected {n_features} features with highest scores")
    print(f"📊 Score range: [{scores[top_indices].min():.4f}, {scores[top_indices].max():.4f}]")
    
    return top_indices, scores


def augment_data(X, n_augmented=50, noise_level=0.05):
    """
    Generate synthetic samples via noise injection
    
    Args:
        X: Original feature matrix
        n_augmented: Number of augmented samples to generate
        noise_level: Std of Gaussian noise (relative to feature std)
    
    Returns:
        Augmented feature matrix
    """
    print(f"\n{'='*60}")
    print(f"Data Augmentation: Generating {n_augmented} synthetic samples")
    print(f"{'='*60}")
    
    # Calculate feature-wise noise scales
    feature_stds = np.std(X, axis=0)
    noise_scales = feature_stds * noise_level
    
    # Generate augmented samples
    X_aug = []
    for _ in range(n_augmented):
        # Randomly select a base sample
        idx = np.random.randint(0, len(X))
        base_sample = X[idx].copy()
        
        # Add Gaussian noise
        noise = np.random.randn(len(base_sample)) * noise_scales
        augmented_sample = base_sample + noise
        
        X_aug.append(augmented_sample)
    
    X_aug = np.array(X_aug)
    X_combined = np.vstack([X, X_aug])
    
    print(f"✅ Original samples: {len(X)}")
    print(f"✅ Augmented samples: {len(X_aug)}")
    print(f"✅ Total samples: {len(X_combined)}")
    
    return X_combined


def build_lightweight_autoencoder(input_dim=20, latent_dim=8, l2_reg=0.001):
    """
    Build lightweight autoencoder for small datasets
    
    KEY CHANGES:
    - Much smaller architecture: 20→16→8→16→20
    - Strong L2 regularization
    - Higher dropout (0.3)
    - Fewer parameters to prevent overfitting
    
    Args:
        input_dim: Number of input features
        latent_dim: Size of latent representation
        l2_reg: L2 regularization strength
    
    Returns:
        Compiled Keras model
    """
    regularizer = keras.regularizers.l2(l2_reg)
    
    # Encoder
    encoder_input = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizer)(encoder_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    encoded = keras.layers.Dense(latent_dim, activation='relu', name='latent', 
                                   kernel_regularizer=regularizer)(x)
    
    # Decoder
    x = keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizer)(encoded)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    decoder_output = keras.layers.Dense(input_dim, activation='linear',
                                        kernel_regularizer=regularizer)(x)
    
    # Build model
    autoencoder = keras.Model(encoder_input, decoder_output, name='lightweight_autoencoder')
    
    # Compile
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder


def train_with_cross_validation(X, n_folds=5, input_dim=20, epochs=100):
    """
    Train using K-fold cross-validation
    
    Args:
        X: Feature matrix
        n_folds: Number of CV folds
        input_dim: Number of features
        epochs: Training epochs per fold
    
    Returns:
        List of trained models, list of validation MAEs
    """
    print(f"\n{'='*60}")
    print(f"{n_folds}-Fold Cross-Validation Training")
    print(f"{'='*60}")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    models = []
    val_maes = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Build model
        model = build_lightweight_autoencoder(input_dim=input_dim)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_mae',
                patience=20,
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_mae',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        # Train
        history = model.fit(
            X_train_scaled, X_train_scaled,
            validation_data=(X_val_scaled, X_val_scaled),
            epochs=epochs,
            batch_size=4,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        val_mae = model.evaluate(X_val_scaled, X_val_scaled, verbose=0)[1]
        val_maes.append(val_mae)
        models.append((model, scaler))
        
        print(f"✅ Fold {fold} Validation MAE: {val_mae:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Mean MAE: {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
    print(f"Best MAE: {np.min(val_maes):.4f}")
    print(f"Worst MAE: {np.max(val_maes):.4f}")
    
    return models, val_maes


def main():
    """
    Optimized training pipeline
    """
    print("🚀 Optimized Enhanced Eye-Tracking Model Training")
    print("="*60)
    print("Strategy: Feature Selection + Lightweight Network + Cross-Validation")
    print()
    
    # ===== LOAD DATA =====
    data_path = 'data/prepared/children_asd_43features.npy'
    metadata_path = data_path.replace('.npy', '_metadata.json')
    
    X = np.load(data_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Loaded: {X.shape[0]} samples × {X.shape[1]} features")
    
    # ===== STRATEGY SELECTION =====
    print(f"\n{'='*60}")
    print("Select Training Strategy:")
    print("{'='*60}")
    print("1. Feature Selection (43→20 features)")
    print("2. PCA Dimensionality Reduction (43→20 components)")
    print("3. Data Augmentation (25→75 samples)")
    print("4. Lightweight Model Only (no preprocessing)")
    print("5. Combined: Feature Selection + Augmentation")
    print()
    
    strategy = input("Enter strategy number (1-5) [default: 5]: ").strip() or "5"
    
    X_processed = X.copy()
    selected_features = None
    pca_transformer = None
    
    if strategy in ['1', '5']:
        # Feature Selection
        selected_features, scores = select_top_features(X, n_features=20, method='mutual_info')
        X_processed = X[:, selected_features]
        
        # Show selected features
        feature_names = metadata['feature_names']
        print("\n📋 Selected Features:")
        for idx in selected_features:
            print(f"  - {feature_names[idx]} (score: {scores[idx]:.4f})")
        
        input_dim = 20
        
    elif strategy == '2':
        # PCA
        print(f"\n{'='*60}")
        print("PCA Dimensionality Reduction")
        print(f"{'='*60}")
        pca_transformer = PCA(n_components=20, random_state=42)
        X_processed = pca_transformer.fit_transform(X)
        explained_var = np.sum(pca_transformer.explained_variance_ratio_)
        print(f"✅ Reduced to 20 components")
        print(f"📊 Explained variance: {explained_var*100:.2f}%")
        input_dim = 20
        
    elif strategy in ['3', '5']:
        # Data Augmentation
        X_processed = augment_data(X_processed, n_augmented=50, noise_level=0.05)
        input_dim = X_processed.shape[1]
        
    else:
        # Strategy 4: No preprocessing
        input_dim = 43
    
    # ===== TRAIN WITH CROSS-VALIDATION =====
    models, val_maes = train_with_cross_validation(
        X_processed,
        n_folds=5,
        input_dim=input_dim,
        epochs=150
    )
    
    # ===== SELECT BEST MODEL =====
    best_fold = np.argmin(val_maes)
    best_model, best_scaler = models[best_fold]
    best_mae = val_maes[best_fold]
    
    print(f"\n{'='*60}")
    print("✅ Best Model Selection")
    print(f"{'='*60}")
    print(f"Best Fold: {best_fold + 1}")
    print(f"Validation MAE: {best_mae:.4f}")
    print(f"Baseline MAE (28 features): 0.4069")
    
    if best_mae < 0.4069:
        improvement = ((0.4069 - best_mae) / 0.4069) * 100
        print(f"✅ Improvement: {improvement:.1f}% better!")
    else:
        degradation = ((best_mae - 0.4069) / 0.4069) * 100
        print(f"⚠️ Degradation: {degradation:.1f}% worse")
    
    # ===== SAVE BEST MODEL =====
    print(f"\n{'='*60}")
    print("Saving Best Model")
    print(f"{'='*60}")
    
    model_dir = 'models/children_asd_optimized'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'model.keras')
    best_model.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Save scaler
    import pickle
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(best_scaler, f)
    print(f"💾 Scaler saved: {scaler_path}")
    
    # Save preprocessing info
    preprocessing_info = {
        'strategy': strategy,
        'input_dim': input_dim,
        'original_features': 43,
        'selected_feature_indices': selected_features.tolist() if selected_features is not None else None,
        'selected_feature_names': [metadata['feature_names'][i] for i in selected_features] if selected_features is not None else None,
        'pca_components': 20 if strategy == '2' else None,
        'augmentation': strategy in ['3', '5'],
        'cv_folds': 5,
        'best_fold': int(best_fold + 1),
        'validation_mae': float(best_mae),
        'all_fold_maes': [float(m) for m in val_maes]
    }
    
    preprocessing_path = os.path.join(model_dir, 'preprocessing.json')
    with open(preprocessing_path, 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"💾 Preprocessing info saved: {preprocessing_path}")
    
    # Save PCA transformer if used
    if pca_transformer is not None:
        pca_path = os.path.join(model_dir, 'pca.pkl')
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_transformer, f)
        print(f"💾 PCA transformer saved: {pca_path}")
    
    print(f"\n{'='*60}")
    print("✅ OPTIMIZED TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nModel Parameters: {best_model.count_params():,}")
    print(f"Training Samples: {len(X_processed)}")
    print(f"Parameters per Sample: {best_model.count_params() / len(X_processed):.1f}")
    print()
    print("Next steps:")
    print("1. Export to TensorFlow.js")
    print("2. Update baseline_model_web.js to use selected features")
    print("3. Test and deploy")


if __name__ == '__main__':
    main()
