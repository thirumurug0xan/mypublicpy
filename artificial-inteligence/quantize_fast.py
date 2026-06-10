import openvino as ov
import nncf
from pathlib import Path
import shutil

# Paths set to Windows mount folder targets
fp16_dir = Path("/mnt/c/Users/thiru/models/qwen2.5-coder-3b-OV-FP16")
int8_fast_dir = Path("/mnt/c/Users/thiru/models/qwen2.5-coder-3b-OV-INT8-FAST")
int8_fast_dir.mkdir(parents=True, exist_ok=True)

print("🔄 Reading FP16 model structure and KV cache map from Windows drive...")
core = ov.Core()
model = core.read_model(fp16_dir / "openvino_model.xml")

print("⚡ Running in-memory INT8 weight compression...")
compressed_model = nncf.compress_weights(model)

print("💾 Saving the fast cache-enabled INT8 model files to Windows...")
ov.save_model(compressed_model, int8_fast_dir / "openvino_model.xml")

print("📋 Moving accompanying tokenizer configuration files...")
for file_item in fp16_dir.iterdir():
    if file_item.suffix not in ['.xml', '.bin']:
        shutil.copy(file_item, int8_fast_dir / file_item.name)

print("🎉 Success! Your high-speed INT8 model is saved on /mnt/c/")
