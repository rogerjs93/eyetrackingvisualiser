"""
Convert TensorFlow/Keras baseline model to TensorFlow.js format
This allows the model to run directly in web browsers.
"""

import tensorflowjs as tfjs
import tensorflow as tf
from pathlib import Path

def convert_model():
    """Convert Keras model to TensorFlow.js format."""
    
    print("=" * 70)
    print("🔄 Converting Baseline Model to TensorFlow.js")
    print("=" * 70)
    
    # Load Keras model
    model_path = Path('models/baseline/autism_baseline_model.keras')
    output_path = Path('models/baseline_tfjs')
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please run baseline_model_builder.py first to create the model.")
        return
    
    print(f"\n📂 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"✅ Model loaded successfully")
    print(f"   Architecture: {len(model.layers)} layers")
    print(f"   Parameters: {model.count_params():,}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to TensorFlow.js format
    print(f"\n🔄 Converting to TensorFlow.js format...")
    print(f"   Output directory: {output_path}")
    
    tfjs.converters.save_keras_model(model, str(output_path))
    
    # Check generated files
    generated_files = list(output_path.glob('*'))
    total_size = sum(f.stat().st_size for f in generated_files) / (1024 * 1024)
    
    print(f"\n✅ Conversion complete!")
    print(f"\n📦 Generated files:")
    for file in sorted(generated_files):
        size = file.stat().st_size / 1024
        if size > 1024:
            size_str = f"{size/1024:.1f} MB"
        else:
            size_str = f"{size:.1f} KB"
        print(f"   • {file.name} ({size_str})")
    
    print(f"\n📊 Total size: {total_size:.1f} MB")
    print(f"\n🌐 The model can now be loaded in web browsers using TensorFlow.js!")
    print(f"   Model URL: models/baseline_tfjs/model.json")
    
    print("\n" + "=" * 70)
    print("✅ Ready for web deployment!")
    print("=" * 70)

if __name__ == '__main__':
    convert_model()
