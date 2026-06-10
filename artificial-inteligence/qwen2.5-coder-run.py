import time
import os
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, TextStreamer

# Point directly to the folder containing the fast in-memory compiled model
MODEL_DIR = "/home/thiru/models/qwen2.5-coder-3b-OV-INT8-FAST"

# 1. Initialize Tokenizer natively
print("⏳ Loading tokenizer engine...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, fix_mistral_regex=True)

# 2. Load the model with use_cache=False to get past the initial check
print("🚀 Compiling graph structures for OpenVINO GPU Target...")
model = OVModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device="GPU",
    use_cache=False,  # 🔧 Bypasses the initial validation crash
    ov_config={"PERFORMANCE_HINT": "LATENCY"}
)

# 🚀 THE SPEED PATCH: Re-enable token generation cache right before execution loops
model.generation_config.use_cache = True

# Initialize Conversation History
conversation_history = [
    {"role": "system", "content": "You are a helpful, senior-level AI software engineering assistant."}
]

print("\n🤖 Qwen 2.5 Coder is active on the GPU engine with full caching acceleration!")
print("Type 'exit' or 'quit' to safely close the session.")
print("=" * 65)

# 3. Interactive Conversation Loop
while True:
    try:
        user_prompt = input("\n👤 User: ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        conversation_history.append({"role": "user", "content": user_prompt})

        formatted_prompt = tokenizer.apply_chat_template(
            conversation_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        encoded_inputs = tokenizer(formatted_prompt, return_tensors="pt")

        print("\n🤖 Qwen: ", end="", flush=True)
        live_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        start_timestamp = time.time()

        # This will now utilize the internal stateful execution cache of OpenVINO!
        raw_generation_output = model.generate(
            **encoded_inputs,
            streamer=live_streamer,
            max_new_tokens=512,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        execution_duration = time.time() - start_timestamp

        input_token_count = encoded_inputs.input_ids.shape[1]
        total_token_count = raw_generation_output.shape[1]
        generated_token_count = total_token_count - input_token_count
        tokens_per_second = generated_token_count / execution_duration

        decoded_response = tokenizer.decode(
            raw_generation_output[0][input_token_count:], 
            skip_special_tokens=True
        )
        conversation_history.append({"role": "assistant", "content": decoded_response})

        print(f"\n\n📊 [Speed: {tokens_per_second:.2f} tokens/sec | Engine: {generated_token_count} tokens processed in {execution_duration:.2f}s]")
        print("=" * 65)

    except KeyboardInterrupt:
        print("\nExiting.")
        break
