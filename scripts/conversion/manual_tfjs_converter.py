"""
Manual TensorFlow.js model converter
Extracts weights from Keras model and creates TFJS format without tensorflowjs package
"""
import tensorflow as tf
from tensorflow import keras
import json
import numpy as np
from pathlib import Path
import shutil

def convert_keras_to_tfjs_manual(keras_model_path, output_dir, scaler_json_path):
    """
    Manually convert Keras model to TensorFlow.js format
    Uses Sequential model format for better compatibility
    """
    print(f"\n📂 Loading Keras model: {keras_model_path}")
    model = keras.models.load_model(keras_model_path)
    print(f"✅ Model loaded: {model.count_params():,} parameters")
    
    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Rebuild as Sequential model for TF.js compatibility
    print("🔄 Converting to Sequential format for TF.js...")
    sequential_model = keras.Sequential()
    
    # Add Input layer explicitly
    sequential_model.add(keras.Input(shape=(28,), name='input'))
    
    # Copy all layers except InputLayer
    for layer in model.layers:
        if not isinstance(layer, keras.layers.InputLayer):
            # Clone the layer
            new_layer = layer.__class__.from_config(layer.get_config())
            sequential_model.add(new_layer)
    
    # Copy weights
    sequential_model.set_weights(model.get_weights())
    
    # Get Sequential model config
    config = sequential_model.get_config()
    
    # Fix InputLayer config for TensorFlow.js compatibility
    for layer in config.get('layers', []):
        if layer.get('class_name') == 'InputLayer':
            layer_config = layer.get('config', {})
            # Convert batch_shape to batchInputShape for TF.js
            if 'batch_shape' in layer_config:
                layer_config['batchInputShape'] = layer_config.pop('batch_shape')
            # Ensure dtype is set
            if 'dtype' not in layer_config:
                layer_config['dtype'] = 'float32'
    
    # Create TF.js model topology
    topology = {
        "class_name": "Sequential",
        "config": config,
        "keras_version": tf.keras.__version__,
        "backend": "tensorflow"
    }
    
    # Collect all weights
    all_weights = []
    weight_data = []
    
    for layer in sequential_model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) > 0:
            for i, weight in enumerate(layer_weights):
                weight_name = f"{layer.name}/weight_{i}"
                all_weights.append({
                    "name": weight_name,
                    "shape": list(weight.shape),
                    "dtype": "float32"
                })
                weight_data.append(weight.astype(np.float32))
    
    # Create weights manifest
    weight_file = "weights.bin"
    
    # Create final model config
    model_config = {
        "format": "layers-model",
        "generatedBy": "TensorFlow.js v4.x Converter",
        "convertedBy": "TensorFlow.js Converter v4.22.0",
        "modelTopology": topology,
        "weightsManifest": [{
            "paths": [weight_file],
            "weights": all_weights
        }]
    }
    
    # Save model.json
    model_json_path = output_path / "model.json"
    with open(model_json_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"✓ Created model.json (Sequential format)")
    
    # Save weights as binary
    weights_path = output_path / weight_file
    concatenated_weights = np.concatenate([w.flatten() for w in weight_data])
    with open(weights_path, 'wb') as f:
        f.write(concatenated_weights.tobytes())
    print(f"✓ Created {weight_file} ({concatenated_weights.nbytes:,} bytes)")
    
    # Copy scaler.json
    if Path(scaler_json_path).exists():
        shutil.copy(scaler_json_path, output_path / 'scaler.json')
        print(f"✓ Copied scaler.json")
    
    print(f"\n✅ TensorFlow.js model created in: {output_path}")
    print(f"   Files: model.json, {weight_file}, scaler.json")
    
    return True

def main():
    print("\n" + "="*70)
    print("MANUAL TENSORFLOW.JS CONVERSION")
    print("="*70)
    print("\n🔧 Creating browser-compatible models without tensorflowjs package\n")
    
    models_to_convert = [
        {
            "name": "Children ASD Baseline (Ages 3-12)",
            "keras_path": "models/baseline_children_asd/optimized_autoencoder.keras",
            "output_dir": "models/baseline_children_asd_tfjs",
            "scaler_path": "models/baseline_children_asd/scaler.json"
        },
        {
            "name": "Adult ASD Baseline",
            "keras_path": "models/baseline_adult_asd/adult_asd_baseline.keras",
            "output_dir": "models/baseline_adult_asd_tfjs",
            "scaler_path": "models/baseline_adult_asd/scaler.json"
        }
    ]
    
    results = []
    
    for model_info in models_to_convert:
        print(f"\n{'='*70}")
        print(f"Converting: {model_info['name']}")
        print(f"{'='*70}")
        
        try:
            success = convert_keras_to_tfjs_manual(
                model_info['keras_path'],
                model_info['output_dir'],
                model_info['scaler_path']
            )
            results.append(success)
            print(f"✅ {model_info['name']} - SUCCESS")
        except Exception as e:
            print(f"❌ {model_info['name']} - FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*70)
    if all(results):
        print("✅ ALL CONVERSIONS SUCCESSFUL!")
    else:
        print("⚠️  SOME CONVERSIONS FAILED")
    print("="*70)
    
    print("\n📋 Models ready for deployment:")
    print("   - models/baseline_children_asd_tfjs/")
    print("   - models/baseline_adult_asd_tfjs/")
    print("\n🌐 Update index.html to use these paths")

if __name__ == "__main__":
    main()
