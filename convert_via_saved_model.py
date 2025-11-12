"""
Alternative converter: Keras -> SavedModel -> TensorFlow.js
Avoids tensorflowjs Python library dependency issues
"""

import tensorflow as tf
from pathlib import Path
import shutil
import subprocess
import json

def convert_model():
    """Convert Keras model via SavedModel intermediate format"""
    
    print("=" * 70)
    print("🔄 Converting Baseline Model to TensorFlow.js (Alternative Method)")
    print("=" * 70)
    
    # Paths
    keras_model = Path('models/baseline/autism_baseline_model.keras')
    saved_model_dir = Path('models/baseline_saved_model')
    tfjs_output = Path('models/baseline_tfjs')
    
    if not keras_model.exists():
        print(f"❌ Error: Model not found at {keras_model}")
        return
    
    # Step 1: Load Keras model
    print(f"\n📂 Loading Keras model from: {keras_model}")
    model = tf.keras.models.load_model(keras_model)
    print(f"✅ Model loaded: {len(model.layers)} layers, {model.count_params():,} parameters")
    
    # Step 2: Save as TensorFlow SavedModel (Keras 3 uses export)
    print(f"\n🔄 Converting to SavedModel format...")
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    
    # Keras 3 uses export() instead of save() for SavedModel
    model.export(saved_model_dir)
    print(f"✅ SavedModel created at: {saved_model_dir}")
    
    # Step 3: Use command-line tensorflowjs_converter
    print(f"\n🔄 Converting SavedModel to TensorFlow.js...")
    tfjs_output.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_graph_model',
        '--signature_name=serving_default',
        '--saved_model_tags=serve',
        str(saved_model_dir),
        str(tfjs_output)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ TensorFlow.js conversion complete!")
    except FileNotFoundError:
        print("❌ Error: tensorflowjs_converter not found")
        print("   Install with: pip install tensorflowjs")
        return
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        print(f"   Error: {e.stderr}")
        return
    
    # Clean up SavedModel
    shutil.rmtree(saved_model_dir)
    print("🧹 Cleaned up temporary SavedModel")
    
    # Check output
    files = list(tfjs_output.glob('*'))
    total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
    
    print(f"\n📦 Generated files:")
    for file in sorted(files):
        size = file.stat().st_size / 1024
        size_str = f"{size/1024:.1f} MB" if size > 1024 else f"{size:.1f} KB"
        print(f"   • {file.name} ({size_str})")
    
    print(f"\n📊 Total size: {total_size:.1f} MB")
    print(f"\n🌐 Model ready for web browsers!")
    print(f"   Load with: tf.loadGraphModel('models/baseline_tfjs/model.json')")
    
    # Create metadata file for web app
    metadata = {
        "modelType": "graphModel",
        "inputShape": [28],
        "outputShape": [28],
        "layers": len(model.layers),
        "parameters": int(model.count_params()),
        "framework": "TensorFlow.js",
        "format": "tfjs_graph_model"
    }
    
    with open(tfjs_output / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📝 Created metadata.json with model information")
    print("\n" + "=" * 70)
    print("✅ Ready for web deployment!")
    print("=" * 70)

if __name__ == '__main__':
    convert_model()
