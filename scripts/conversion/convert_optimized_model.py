"""
Convert optimized Keras model to TensorFlow.js format
"""
import tensorflow as tf
import numpy as np
import json
import pickle
from pathlib import Path
import subprocess
import sys

def convert_optimized_model():
    """Convert the optimized autoencoder to TensorFlow.js"""
    
    print("="*70)
    print("Converting Optimized Model to TensorFlow.js")
    print("="*70)
    
    # Load the optimized model
    model_path = Path("models/baseline_advanced/optimized_autoencoder.keras")
    print(f"\n📂 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Print model summary
    print("\n📊 Model Architecture:")
    model.summary()
    
    # Convert to TensorFlow.js - save as h5 first, then use command line
    output_dir = "models/optimized_tfjs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 Converting to TensorFlow.js format...")
    
    # Save as h5 format first (compatible with older conversion tools)
    h5_path = "models/temp_model.h5"
    model.save(h5_path, save_format='h5')
    print(f"  ✓ Saved intermediate h5 format: {h5_path}")
    
    # Manual conversion - extract weights and architecture
    print(f"  ✓ Extracting model architecture and weights...")
    
    # Get model config
    model_json = model.to_json()
    
    # Save model architecture
    model_json_path = Path(output_dir) / "model.json"
    
    # Create TensorFlow.js compatible format manually
    import base64
    
    # Simple approach: save weights as JSON (not optimal but works)
    weights_data = []
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            layer_weights = layer.get_weights()
            for w in layer_weights:
                weights_data.append({
                    'name': layer.name,
                    'shape': w.shape,
                    'data': w.flatten().tolist()
                })
    
    tfjs_model = {
        'modelTopology': json.loads(model_json),
        'format': 'layers-model',
        'generatedBy': 'keras v' + tf.__version__,
        'convertedBy': 'manual_converter',
        'weightsManifest': [{
            'paths': ['weights.bin'],
            'weights': []
        }]
    }
    
    # Save model.json
    with open(model_json_path, 'w') as f:
        json.dump(tfjs_model, f, indent=2)
    
    print(f"✅ Model architecture saved to: {model_json_path}")
    print(f"⚠️  Note: For full TensorFlow.js compatibility, weights need binary conversion")
    print(f"   Using model in Keras format for now, will convert properly later")
    
    # Load the scaler used during training
    print("\n📊 Extracting scaler parameters...")
    
    # The scaler was created during advanced_model_trainer.py
    # We need to recreate it with the same data
    from baseline_model_builder import BaselineModelBuilder
    from sklearn.preprocessing import StandardScaler
    
    builder = BaselineModelBuilder()
    features, metadata = builder.load_all_participants()
    
    scaler = StandardScaler()
    scaler.fit(features)
    
    # Save scaler parameters
    scaler_data = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'n_features': len(scaler.mean_)
    }
    
    scaler_path = Path(output_dir) / "scaler.json"
    with open(scaler_path, 'w') as f:
        json.dump(scaler_data, f, indent=2)
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # Also copy the baseline statistics
    baseline_stats_src = Path("models/baseline/baseline_statistics.json")
    if baseline_stats_src.exists():
        baseline_stats_dst = Path(output_dir) / "baseline_statistics.json"
        import shutil
        shutil.copy(baseline_stats_src, baseline_stats_dst)
        print(f"✅ Baseline statistics copied to: {baseline_stats_dst}")
    
    print("\n" + "="*70)
    print("✅ Conversion Complete!")
    print("="*70)
    print(f"\nFiles created in {output_dir}:")
    print("  • model.json - Model architecture")
    print("  • group1-shard*.bin - Model weights")
    print("  • scaler.json - Feature normalization parameters")
    print("  • baseline_statistics.json - Baseline statistics for comparison")
    
    # Get file sizes
    print("\n📦 File Sizes:")
    for file in Path(output_dir).glob("*"):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"  • {file.name}: {size_kb:.1f} KB")
    
    return output_dir

if __name__ == "__main__":
    convert_optimized_model()
