import openvino as ov
import nncf
from pathlib import Path
import shutil
import json

# 🔧 CORRECTED PATH: Reading from your actual Windows mount path
fp16_dir = Path("/mnt/c/Users/thiru/models/qwen2.5-coder-3b-OV-FP16")
int8_fast_dir = Path("/home/thiru/models/qwen2.5-coder-3b-OV-INT8-FAST")
int8_fast_dir.mkdir(parents=True, exist_ok=True)

# 1. Read the successful FP16 model structure (with its native KV cache hooks)
print(f"🔄 Loading FP16 model graph directly from: {fp16_dir}")
core = ov.Core()
model = core.read_model(fp16_dir / "openvino_model.xml")

# 2. Compress weights natively via NNCF in system RAM
print("⚡ Compressing linear layers to INT8 (This bypasses the file system pipeline)...")
compressed_model = nncf.compress_weights(model)

# 3. Save the completed model in a single, continuous, stable write pass
print("💾 Saving structural INT8 model files to native Linux workspace...")
ov.save_model(compressed_model, int8_fast_dir / "openvino_model.xml")

# 4. Copy all configuration and tokenizer metadata
print("📋 Copying layout configuration and tokenizer metadata files...")
for file_item in fp16_dir.iterdir():
    if file_item.suffix not in ['.xml', '.bin']:
        shutil.copy(file_item, int8_fast_dir / file_item.name)

# 5. Fix config.json configuration parameters
print("🔧 Updating configuration parameters to authorize tracking caches...")
config_path = int8_fast_dir / "config.json"
if config_path.exists():
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Force set parameters to make Optimum wrapper accept the graph structure
    config_data["use_cache"] = True
    
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

# 6. Generate the openvino_config file that defines the model as stateful
ov_config_path = int8_fast_dir / "openvino_config.json"
with open(ov_config_path, "w") as f:
    json.dump({"stateful": True}, f, indent=2)

print("\n🎉 Success! Your optimized INT8-FAST model is compiled and patched.")
