import time
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

# Point directly to OpenVINO's officially pre-compiled model repository
model_dir = "OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov"

print("Downloading and loading pre-compiled OpenVINO model...")
# Added fix_mistral_regex to clear your original tokenizer warning
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, fix_mistral_regex=True)

# Load the working model straight to your CPU
model = OVModelForCausalLM.from_pretrained(model_dir, device="CPU")

# Chat Prompt
messages = [
    {"role": "system", "content": "You are a helpful and concise AI assistant."},
    {"role": "user", "content": "Explain quantum computing in one simple sentence."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

print("\nGenerating response...")
start_time = time.time()

output_tokens = model.generate(
    **inputs, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.7,
    top_p=0.9
)

generation_time = time.time() - start_time
response = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\n--- AI Response ---")
print(response)
print("-------------------")
print(f"Generation took: {generation_time:.2f} seconds")
