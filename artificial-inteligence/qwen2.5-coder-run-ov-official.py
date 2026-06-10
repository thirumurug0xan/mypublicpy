import sys
import time
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, TextStreamer

# 🚀 POINT TO INTEL'S OFFICIAL PRE-EXPORTED INT8 3B MODEL
model_dir = "OpenVINO/Qwen2.5-Coder-3B-Instruct-int8-ov"

print("Loading Official Intel 3B INT8 model onto GPU...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, fix_mistral_regex=True)

ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "CACHE_DIR": "./ov_cache_3b"  # Caches compilation states so your 2nd run starts instantly
}

# Load the model with KV Cache natively recognized
model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device="GPU",
    use_cache=True,  # ⚡ Natively supported now!
    ov_config=ov_config
)

print("\nChat system ready on GPU! Type 'exit' or 'quit' to stop.")
print("==========================================================")

history = [
    {"role": "system", "content": "You are a helpful and concise AI assistant."}
]

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting chat. Goodbye!")
        break

    if not user_input.strip():
        continue

    history.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\nAI: ", end="")
    sys.stdout.flush()

    live_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    start_time = time.time()

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=live_streamer
    )

    end_time = time.time()

    input_len = inputs.input_ids.shape[1]
    new_tokens_count = len(output_tokens[0]) - input_len
    total_time = end_time - start_time

    full_response = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": full_response})

    tokens_per_sec = new_tokens_count / total_time
    print(f"\n\n[Stats: {new_tokens_count} tokens generated in {total_time:.2f}s | Speed: {tokens_per_sec:.2f} tokens/sec]")
    print("-" * 50)
