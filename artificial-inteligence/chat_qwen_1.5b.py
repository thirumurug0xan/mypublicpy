import os
import sys
import time
import warnings
import threading
from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

def main():
    # Resolve the model path dynamically relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "ai-chat-interface", "qwen-qwen2-5-1-5b-instruct-ov")
    
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at {model_path}")
        print("Please check if the ai-chat-interface/qwen-qwen2-5-1-5b-instruct-ov directory exists.")
        sys.exit(1)

    print("=" * 60)
    print("      Qwen 2.5 1.5B Instruct OpenVINO Loader & Chat")
    print("=" * 60)
    print(f"Model Path: {model_path}")
    
    # Prompt the user for device selection (GPU is recommended, AUTO or CPU as fallbacks)
    device = "GPU"
    print(f"Default target device: {device}")
    print("Press Enter to use default, or type device name (e.g., CPU, AUTO):")
    try:
        user_device = input("> ").strip().upper()
        if user_device:
            device = user_device
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

    # Validate device selection
    valid_devices = ["GPU", "CPU", "AUTO", "XPU"]
    if device not in valid_devices and not device.startswith("HETERO:"):
        print(f"Warning: Unknown device '{device}'. Defaulting to AUTO mode.")
        device = "AUTO"

    # Define device search order
    if device == "AUTO":
        devices_to_try = ["GPU", "CPU"]
    elif device == "GPU":
        devices_to_try = ["GPU", "CPU"]
    elif device == "XPU":
        devices_to_try = ["HETERO:GPU,CPU", "GPU", "CPU"]
    else:
        devices_to_try = [device, "CPU"] if device != "CPU" else ["CPU"]

    print(f"\nLoading model and tokenizer... (This may take a minute to compile the first time)")
    start_time = time.perf_counter()

    # Try loading tokenizer (with fix_mistral_regex=True to fix warnings)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except Exception:
        # Fallback if fix_mistral_regex is not supported by the version
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = None
    loaded_device = None
    # Try with use_cache=True first, then fallback to use_cache=False
    use_cache_configs = [True, False]

    for dev in devices_to_try:
        for use_cache in use_cache_configs:
            try:
                cache_str = "use_cache=True" if use_cache else "use_cache=False"
                print(f"Attempting to load on {dev} ({cache_str})...")
                model = OVModelForCausalLM.from_pretrained(
                    model_path,
                    device=dev,
                    compile=True,
                    use_cache=use_cache
                )
                loaded_device = dev
                break
            except Exception as e:
                err_msg = str(e)
                print(f"Failed loading on {dev} (use_cache={use_cache}): {err_msg}")
                # If it's not a use_cache related error (e.g. device not found or out of memory)
                # and we are on GPU, we fail fast on this device and proceed to next device in search order
                if "use_cache" not in err_msg and "use_cache=False" not in err_msg:
                    break
        if model is not None:
            break

    if model is None:
        print("Error: Failed to load model on any device.")
        sys.exit(1)

    load_time = time.perf_counter() - start_time
    print(f"Successfully loaded and compiled model in {load_time:.2f} seconds on {loaded_device}!")
    
    history = []
    print("\n" + "=" * 60)
    print(" Chat session started! Type 'exit' or 'quit' to end. ")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
                
            history.append({"role": "user", "content": user_input})
            
            # Format the conversation history into chat template format
            text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt")
            
            # Streaming setup
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512)
            
            # Start generation in a separate thread so we can stream tokens concurrently
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            print("Assistant: ", end="", flush=True)
            response_chunks = []
            
            for new_text in streamer:
                print(new_text, end="", flush=True)
                response_chunks.append(new_text)
            print() # Print newline at the end of the streaming response
            
            full_response = "".join(response_chunks)
            history.append({"role": "assistant", "content": full_response})
            
            # Keep history to last 20 turns (10 user + 10 assistant) to avoid context bloat
            if len(history) > 20:
                history = history[-20:]
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")
            
if __name__ == "__main__":
    main()
