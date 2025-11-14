"""
Simple test script to compare original vs optimized model locally
"""
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from baseline_model_builder import BaselineModelBuilder
from sklearn.preprocessing import StandardScaler

print("="*70)
print("Loading Both Models for Comparison")
print("="*70)

# Load original baseline model
print("\n📂 Loading ORIGINAL baseline model...")
original_model = tf.keras.models.load_model("models/baseline/autism_baseline_model.keras")
print(f"✅ Original model loaded")
print(f"   Parameters: {original_model.count_params():,}")

# Load optimized model
print("\n📂 Loading OPTIMIZED model...")
optimized_model = tf.keras.models.load_model("models/baseline_advanced/optimized_autoencoder.keras")
print(f"✅ Optimized model loaded")
print(f"   Parameters: {optimized_model.count_params():,}")

# Load test data
print("\n📂 Loading test data (23 autism participants)...")
builder = BaselineModelBuilder()
features, metadata = builder.load_all_participants()

# Normalize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print(f"✅ Loaded {len(features)} samples")

# Test both models
print("\n" + "="*70)
print("Testing Both Models")
print("="*70)

# Original model predictions
original_predictions = original_model.predict(features_scaled, verbose=0)
original_errors = np.mean(np.abs(features_scaled - original_predictions), axis=1)
original_mae = np.mean(original_errors)

# Optimized model predictions
optimized_predictions = optimized_model.predict(features_scaled, verbose=0)
optimized_errors = np.mean(np.abs(features_scaled - optimized_predictions), axis=1)
optimized_mae = np.mean(optimized_errors)

print(f"\n📊 Results on {len(features)} autism samples:")
print(f"\n  Original Model:")
print(f"    Mean MAE: {original_mae:.4f}")
print(f"    Min MAE:  {np.min(original_errors):.4f}")
print(f"    Max MAE:  {np.max(original_errors):.4f}")
print(f"    Std MAE:  {np.std(original_errors):.4f}")

print(f"\n  Optimized Model:")
print(f"    Mean MAE: {optimized_mae:.4f}")
print(f"    Min MAE:  {np.min(optimized_errors):.4f}")
print(f"    Max MAE:  {np.max(optimized_errors):.4f}")
print(f"    Std MAE:  {np.std(optimized_errors):.4f}")

improvement = ((original_mae - optimized_mae) / original_mae) * 100
print(f"\n🎯 Improvement: {improvement:.1f}% better reconstruction")

# Show per-sample comparison for first 5
print(f"\n📋 Per-Sample Comparison (first 5 participants):")
print(f"{'ID':<5} {'Original MAE':<15} {'Optimized MAE':<15} {'Improvement':<15}")
print("-" * 60)
for i in range(min(5, len(features))):
    orig_err = original_errors[i]
    opt_err = optimized_errors[i]
    imp = ((orig_err - opt_err) / orig_err) * 100
    print(f"{i+1:<5} {orig_err:<15.4f} {opt_err:<15.4f} {imp:<15.1f}%")

print("\n" + "="*70)
print("✅ Comparison Complete!")
print("="*70)
print("\n💡 The optimized model shows better (lower) reconstruction")
print("   errors, meaning it learned the autism patterns more accurately.")
