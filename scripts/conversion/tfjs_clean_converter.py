"""
Clean TensorFlow.js converter using official SavedModel -> TF.js conversion
This bypasses the broken tensorflowjs pip package
"""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import subprocess
import tempfile
import shutil

def convert_to_clean_tfjs(keras_model_path, output_dir, scaler_mean, scaler_std):
    """
    Convert Keras model to TensorFlow.js using SavedModel intermediate format
    This is the most reliable method
    """
    print(f"\n🔄 Converting {keras_model_path} to TF.js format...")
    
    # Load the Keras model
    model = tf.keras.models.load_model(keras_model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for SavedModel
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = Path(tmpdir) / "saved_model"
        
        # Save as SavedModel format
        print("  📦 Saving as SavedModel...")
        model.save(saved_model_path, save_format='tf')
        
        # Use tensorflowjs_converter command line tool
        print("  🔄 Converting to TensorFlow.js...")
        cmd = [
            'tensorflowjs_converter',
            '--input_format=tf_saved_model',
            '--output_format=tfjs_layers_model',
            '--signature_name=serving_default',
            '--saved_model_tags=serve',
            str(saved_model_path),
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("  ⚠️  tensorflowjs_converter failed, using manual method...")
            # Fall back to manual conversion
            convert_manually(model, output_path)
        else:
            print("  ✅ Conversion successful!")
    
    # Save scaler.json
    scaler_path = output_path / "scaler.json"
    with open(scaler_path, 'w') as f:
        json.dump({
            "mean": scaler_mean.tolist(),
            "std": scaler_std.tolist()
        }, f, indent=2)
    
    # Get file sizes
    model_json_path = output_path / "model.json"
    weights_files = list(output_path.glob("*.bin"))
    total_weights_size = sum(f.stat().st_size for f in weights_files)
    
    print(f"✅ Conversion complete!")
    print(f"   📄 Model: {model_json_path}")
    print(f"   💾 Weights: {total_weights_size:,} bytes")
    print(f"   📊 Scaler: {scaler_path}")
    
    return model_json_path

def convert_manually(model, output_path):
    """
    Manual conversion as fallback - saves model with proper weight naming
    """
    # Use model.save with SavedModel format then manually parse
    import h5py
    temp_h5 = output_path / "temp_model.h5"
    model.save(temp_h5, save_format='h5')
    
    # Read HDF5 and convert
    with h5py.File(temp_h5, 'r') as f:
        # Extract model config
        model_config = json.loads(model.to_json())
        
        # Extract weights properly
        weight_groups = []
        all_weight_data = []
        
        for layer in model.layers:
            if len(layer.weights) == 0:
                continue
            
            layer_weights = layer.get_weights()
            group_weights = []
            
            for i, w in enumerate(layer_weights):
                weight_name = layer.weights[i].name.replace(':0', '')
                group_weights.append({
                    "name": weight_name,
                    "shape": list(w.shape),
                    "dtype": "float32"
                })
                all_weight_data.append(w.flatten())
        
            weight_groups.append({
                "paths": [f"group{len(weight_groups)}-shard1of1.bin"],
                "weights": group_weights
            })
        
        # Create model.json
        model_json = {
            "format": "layers-model",
            "generatedBy": "TensorFlow.js Converter",
            "convertedBy": "Manual Converter v1.0",
            "modelTopology": model_config,
            "weightsManifest": weight_groups
        }
        
        # Save model.json
        with open(output_path / "model.json", 'w') as out:
            json.dump(model_json, out, indent=2)
        
        # Save weight shards
        for i, weights in enumerate([all_weight_data]):
            shard_data = np.concatenate(weights).astype(np.float32)
            with open(output_path / f"group{i}-shard1of1.bin", 'wb') as out:
                out.write(shard_data.tobytes())
    
    # Clean up temp file
    temp_h5.unlink()

def main():
    """Convert both children and adult models"""
    
    # Children model
    print("\n" + "="*60)
    print("CONVERTING CHILDREN MODEL")
    print("="*60)
    
    with open('models/baseline_children_asd/scaler.json', 'r') as f:
        children_scaler = json.load(f)
    # Children scaler has variance, need to take sqrt
    children_std = np.sqrt(np.array(children_scaler['var'])) if 'var' in children_scaler else np.array(children_scaler['std'])
    convert_to_clean_tfjs(
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
    # Adult scaler has std directly
    adult_std = np.sqrt(np.array(adult_scaler['var'])) if 'var' in adult_scaler else np.array(adult_scaler['std'])
    convert_to_clean_tfjs(
        keras_model_path='models/baseline_adult_asd/adult_asd_baseline.keras',
        output_dir='models/baseline_adult_asd_tfjs',
        scaler_mean=np.array(adult_scaler['mean']),
        scaler_std=adult_std
    )
    
    print("\n" + "="*60)
    print("✅ ALL CONVERSIONS COMPLETE!")
    print("="*60)
    print("\n🎯 Next steps:")
    print("1. Test locally: python -m http.server 8000")
    print("2. Visit: http://localhost:8000")
    print("3. Check browser console for model loading")
    print("4. If successful, commit and push to GitHub Pages")

if __name__ == "__main__":
    main()
