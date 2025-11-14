"""
Simple TensorFlow.js converter using model.to_json() format
"""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

def convert_keras_to_tfjs(keras_model_path, output_dir, scaler_mean, scaler_std):
    """
    Convert Keras model to TensorFlow.js Sequential format (TF.js compatible)
    """
    print(f"\n🔄 Converting {keras_model_path}...")
    
    # Load model
    original_model = tf.keras.models.load_model(keras_model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Rebuild as Sequential model (TF.js doesn't handle Functional API well)
    print("  🔄 Converting to Sequential model...")
    sequential_model = tf.keras.Sequential()
    
    # Add all layers except InputLayer
    for layer in original_model.layers:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            sequential_model.add(layer)
    
    # Build the model with correct input shape
    input_shape = original_model.input_shape[1:]  # Remove batch dimension
    sequential_model.build((None,) + input_shape)
    
    # Get Sequential model config
    model_json_config = json.loads(sequential_model.to_json())
    
    # Ensure InputLayer has correct format
    if 'layers' in model_json_config.get('config', {}):
        layers = model_json_config['config']['layers']
        if layers and layers[0].get('class_name') == 'InputLayer':
            config = layers[0].get('config', {})
            # Ensure batch_input_shape format
            if 'batch_shape' in config:
                config['batch_input_shape'] = config.pop('batch_shape')
            elif 'input_shape' in config:
                # Convert input_shape to batch_input_shape
                config['batch_input_shape'] = [None] + list(config.pop('input_shape'))
            config.pop('sparse', None)
            config.pop('ragged', None)
    
    # Extract all weights
    all_weight_data = []
    weight_specs = []
    
    for layer in sequential_model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) == 0:
            continue
        
        for i, weight in enumerate(layer_weights):
            # Build full path: layer_name/weight_name
            # e.g., "dense/kernel", "batch_normalization/gamma"
            weight_tensor = layer.weights[i]
            weight_base_name = weight_tensor.name.split('/')[-1].replace(':0', '')
            full_weight_name = f"{layer.name}/{weight_base_name}"
            
            all_weight_data.append(weight.flatten())
            weight_specs.append({
                "name": full_weight_name,
                "shape": list(weight.shape),
                "dtype": "float32"
            })
    
    # Concatenate all weights into single binary
    concatenated_weights = np.concatenate(all_weight_data).astype(np.float32)
    
    # Create model.json in TensorFlow.js format
    tfjs_model = {
        "format": "layers-model",
        "generatedBy": "keras v" + tf.__version__,
        "convertedBy": "TensorFlow.js Converter",
        "modelTopology": model_json_config,
        "weightsManifest": [{
            "paths": ["group1-shard1of1.bin"],
            "weights": weight_specs
        }]
    }
    
    # Save model.json
    model_json_path = output_path / "model.json"
    with open(model_json_path, 'w') as f:
        json.dump(tfjs_model, f, indent=2)
    
    # Save weights binary
    weights_path = output_path / "group1-shard1of1.bin"
    with open(weights_path, 'wb') as f:
        f.write(concatenated_weights.tobytes())
    
    # Save scaler
    scaler_path = output_path / "scaler.json"
    with open(scaler_path, 'w') as f:
        json.dump({
            "mean": scaler_mean.tolist(),
            "std": scaler_std.tolist()
        }, f, indent=2)
    
    print(f"✅ Conversion complete!")
    print(f"   📄 Model: {model_json_path}")
    print(f"   💾 Weights: {weights_path.stat().st_size:,} bytes ({len(weight_specs)} tensors)")
    print(f"   📊 Scaler: {scaler_path}")
    
    return model_json_path

def main():
    """Convert all three baseline models"""
    
    # Children model
    print("\n" + "="*60)
    print("CONVERTING CHILDREN MODEL")
    print("="*60)
    
    with open('models/baseline_children_asd/scaler.json', 'r') as f:
        children_scaler = json.load(f)
    children_std = np.sqrt(np.array(children_scaler['var'])) if 'var' in children_scaler else np.array(children_scaler['std'])
    convert_keras_to_tfjs(
        keras_model_path='models/baseline_children_asd/optimized_autoencoder.keras',
        output_dir='models/baseline_children_asd_tfjs',
        scaler_mean=np.array(children_scaler['mean']),
        scaler_std=children_std
    )
    
    # Adult model
    print("\n" + "="*60)
    print("CONVERTING ADULT MODEL")
    print("="*60)
    
    with open('models/baseline_adult_asd/scaler.json', 'r') as f:
        adult_scaler = json.load(f)
    adult_std = np.sqrt(np.array(adult_scaler['var'])) if 'var' in adult_scaler else np.array(adult_scaler['std'])
    convert_keras_to_tfjs(
        keras_model_path='models/baseline_adult_asd/adult_asd_baseline.keras',
        output_dir='models/baseline_adult_asd_tfjs',
        scaler_mean=np.array(adult_scaler['mean']),
        scaler_std=adult_std
    )
    
    # Neurotypical model
    print("\n" + "="*60)
    print("CONVERTING NEUROTYPICAL MODEL")
    print("="*60)
    
    with open('models/baseline_neurotypical/scaler.json', 'r') as f:
        neuro_scaler = json.load(f)
    neuro_std = np.sqrt(np.array(neuro_scaler['var'])) if 'var' in neuro_scaler else np.array(neuro_scaler['std'])
    convert_keras_to_tfjs(
        keras_model_path='models/baseline_neurotypical/neurotypical_baseline.keras',
        output_dir='models/baseline_neurotypical_tfjs',
        scaler_mean=np.array(neuro_scaler['mean']),
        scaler_std=neuro_std
    )
    
    print("\n" + "="*60)
    print("✅ ALL CONVERSIONS COMPLETE!")
    print("="*60)
    print("\n🎯 Test locally at: http://localhost:8000")

if __name__ == "__main__":
    main()
