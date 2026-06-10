import sys
import time
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, TextStreamer

# 1. Point to the working remote repository path
model_dir = "OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov"

print("Loading model and tokenizer onto GPU...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, fix_mistral_regex=True)

# 2. OpenVINO Configuration parameters to optimize GPU pipeline performance
ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "CACHE_DIR": "./ov_cache"  # Caches GPU kernels so your 2nd run starts instantly
}

# 3. Changed device target to "GPU"
model = OVModelForCausalLM.from_pretrained(
    model_dir, 
    device="GPU", 
    ov_config=ov_config
)

# Custom token counter streamer implementation
class TokenCountingStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.token_count = 0
        self.start_time = None

    def put(self, value):
        if self.start_time is None:
            self.start_time = time.time()
        self.token_count += value.numel()
        super().put(value)

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
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu") # OpenVINO launcher handles transfer to GPU internally

    print("\nAI: ", end="")
    sys.stdout.flush()

    streamer = TokenCountingStreamer(tokenizer)
    
    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=512, 
        do_sample=True, 
        temperature=0.7,
        top_p=0.9,
        streamer=streamer
    )
    
    end_time = time.time()
    
    input_len = inputs.input_ids.shape[1]
    new_tokens_count = len(output_tokens[0]) - input_len
    total_time = end_time - streamer.start_time if streamer.start_time else 0.01

    full_response = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": full_response})

    tokens_per_sec = new_tokens_count / total_time
    print(f"\n\n[Stats: {new_tokens_count} tokens generated in {total_time:.2f}s | Speed: {tokens_per_sec:.2f} tokens/sec]")
    print("-" * 50)
