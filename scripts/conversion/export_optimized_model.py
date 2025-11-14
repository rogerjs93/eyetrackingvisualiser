"""
Manual TensorFlow.js exporter for optimized models.
Directly exports weights and architecture to TFJS format, bypassing library compatibility issues.
"""
import tensorflow as tf
import json
import os
import numpy as np
import pickle
from pathlib import Path

def export_weights_to_tfjs(model, output_dir):
    """Export model weights to TensorFlow.js format"""
    
    # Collect all weights
    weight_groups = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) > 0:
            weight_groups.append({
                'name': layer.name,
                'weights': layer_weights
            })
    
    # Create weight manifest
    weight_list = []
    weight_data = []
    
    for group in weight_groups:
        for i, weight_array in enumerate(group['weights']):
            # Convert to float32
            weight_array = weight_array.astype(np.float32)
            
            weight_entry = {
                'name': f"{group['name']}/weight_{i}",
                'shape': list(weight_array.shape),
                'dtype': 'float32'
            }
            weight_list.append(weight_entry)
            
            # Flatten and append to binary data
            weight_data.append(weight_array.flatten())
    
    # Concatenate all weights into single binary file
    all_weights = np.concatenate(weight_data)
    weights_bin = all_weights.tobytes()
    
    # Save binary weights
    weights_path = os.path.join(output_dir, 'group1-shard1of1.bin')
    with open(weights_path, 'wb') as f:
        f.write(weights_bin)
    
    print(f"✅ Weights saved: {weights_path} ({len(weights_bin)/1024:.1f} KB)")
    
    return weight_list, len(weights_bin)

def fix_tfjs_compatibility(config):
    """
    Recursively fix Keras config for TensorFlow.js compatibility
    - Convert 'L2' regularizer to 'l2' (lowercase)
    - Can add more fixes as needed
    """
    if isinstance(config, dict):
        # Fix L2 regularizer class name
        if config.get('class_name') == 'L2':
            config['class_name'] = 'l2'
        
        # Recursively process all dict values
        for key, value in config.items():
            config[key] = fix_tfjs_compatibility(value)
    elif isinstance(config, list):
        # Recursively process list items
        return [fix_tfjs_compatibility(item) for item in config]
    
    return config

def create_model_json(model, weight_list, weights_size, output_dir):
    """Create model.json with architecture and weight manifest"""
    
    # Build layer topology
    model_topology = {
        'class_name': 'Sequential',
        'config': {
            'name': model.name,
            'layers': []
        }
    }
    
    for layer in model.layers:
        layer_config = layer.get_config()
        
        # Fix InputLayer config for TensorFlow.js compatibility
        if layer.__class__.__name__ == 'InputLayer' and 'batch_shape' in layer_config:
            # TensorFlow.js expects 'batchInputShape' not 'batch_shape'
            layer_config['batchInputShape'] = layer_config.pop('batch_shape')
        
        # Fix regularizers and other compatibility issues
        layer_config = fix_tfjs_compatibility(layer_config)
        
        model_topology['config']['layers'].append({
            'class_name': layer.__class__.__name__,
            'config': layer_config
        })
    
    # Create full model JSON
    model_json = {
        'format': 'layers-model',
        'generatedBy': 'manual-exporter',
        'convertedBy': 'Python 3.11',
        'modelTopology': model_topology,
        'weightsManifest': [
            {
                'paths': ['group1-shard1of1.bin'],
                'weights': weight_list
            }
        ]
    }
    
    # Save model.json
    model_json_path = os.path.join(output_dir, 'model.json')
    with open(model_json_path, 'w') as f:
        json.dump(model_json, f, indent=2)
    
    print(f"✅ Model JSON saved: {model_json_path}")
    
    return model_json_path

def export_optimized_model():
    """Export the optimized children ASD model to TensorFlow.js format"""
    
    print("=" * 60)
    print("🚀 Manual TensorFlow.js Model Exporter")
    print("=" * 60)
    
    # Load model
    model_path = 'models/children_asd_optimized/model.keras'
    print(f"\n📂 Loading model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully")
    print(f"   Architecture: {' → '.join([str(l.units if hasattr(l, 'units') else '?') for l in model.layers])}")
    print(f"   Total parameters: {model.count_params():,}")
    
    # Create output directory
    output_dir = 'models/ACTIVE/children_asd_optimized_tfjs'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Export weights
    print(f"\n⚙️  Exporting weights...")
    weight_list, weights_size = export_weights_to_tfjs(model, output_dir)
    
    # Create model.json
    print(f"\n📝 Creating model.json...")
    create_model_json(model, weight_list, weights_size, output_dir)
    
    # Export scaler
    print(f"\n📊 Exporting scaler...")
    scaler_path = 'models/children_asd_optimized/scaler.pkl'
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        scaler_json = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'feature_count': len(scaler.mean_)
        }
        
        scaler_json_path = os.path.join(output_dir, 'scaler.json')
        with open(scaler_json_path, 'w') as f:
            json.dump(scaler_json, f, indent=2)
        
        print(f"✅ Scaler exported: {scaler_json_path}")
        print(f"   Features: {scaler_json['feature_count']}")
    else:
        print(f"⚠️  Scaler not found: {scaler_path}")
    
    # Export preprocessing metadata
    print(f"\n🔧 Exporting preprocessing metadata...")
    preprocessing_path = 'models/children_asd_optimized/preprocessing.json'
    if os.path.exists(preprocessing_path):
        with open(preprocessing_path, 'r') as f:
            preprocessing = json.load(f)
        
        preprocessing_out_path = os.path.join(output_dir, 'preprocessing.json')
        with open(preprocessing_out_path, 'w') as f:
            json.dump(preprocessing, f, indent=2)
        
        print(f"✅ Preprocessing metadata exported: {preprocessing_out_path}")
        print(f"   Selected features: {len(preprocessing.get('selected_features', []))}")
        print(f"   Feature indices: {preprocessing.get('selected_feature_indices', [])}")
    else:
        print(f"⚠️  Preprocessing metadata not found: {preprocessing_path}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("✅ EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\n📋 Exported files:")
    for file in sorted(Path(output_dir).glob('*')):
        if file.is_file():
            size = file.stat().st_size / 1024  # KB
            print(f"  ✓ {file.name} ({size:.1f} KB)")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Update baseline_model_web.js to use feature selection")
    print(f"  2. Load model from: {output_dir}/model.json")
    print(f"  3. Test in browser with sample CSV")
    
    return True

if __name__ == "__main__":
    success = export_optimized_model()
    if success:
        print("\n✨ Model ready for deployment!")
    else:
        print("\n❌ Export failed")
