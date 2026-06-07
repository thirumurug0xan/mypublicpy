import time
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

def run_benchmark():
    model_path = "./qwen-0.5b-ov"
    device = "GPU"
    
    print(f"Loading model and tokenizer from '{model_path}' onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = OVModelForCausalLM.from_pretrained(model_path, device=device, compile=True)
    
    # Define a standard prompt for benchmarking
    prompt = "Explain the concept of artificial intelligence in two paragraphs."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    input_len = inputs["input_ids"].shape[-1]
    max_new_tokens = 100
    num_runs = 5
    
    print("\n--- Warming up GPU ---")
    # Warmup run to compile pipelines on the hardware
    _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    print(f"--- Starting Accurate Benchmark ({num_runs} iterations) ---")
    
    ttft_list = []
    pure_throughput_list = []
    decode_time_list = []
    
    for i in range(num_runs):
        # 1. Measure Prefill Phase (Time to First Token)
        start_prefill = time.perf_counter()
        _ = model.generate(**inputs, max_new_tokens=1)
        ttft = time.perf_counter() - start_prefill
        ttft_list.append(ttft)
        
        # 2. Measure Total Generation Time (Prefill + Decode)
        start_total = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        total_time = time.perf_counter() - start_total
        
        # 4. Extract Pure Generation Metadata
        total_tokens = outputs[0].shape[-1]
        new_tokens_generated = total_tokens - input_len
        
        # Subtract TTFT from total time to isolate the generation (decode) loop time
        # We subtract 1 token from generation count since its processing was covered in TTFT
        pure_decode_time = total_time - ttft
        pure_decode_tokens = new_tokens_generated - 1
        
        pure_throughput = pure_decode_tokens / pure_decode_time
        
        pure_throughput_list.append(pure_throughput)
        decode_time_list.append(pure_decode_time)
        
        print(f"Run {i+1}: TTFT = {ttft:.4f}s | Pure Gen Throughput = {pure_throughput:.2f} tok/sec")

    # --- Print Summary Statistics ---
    avg_ttft = sum(ttft_list) / num_runs
    avg_throughput = sum(pure_throughput_list) / num_runs
    avg_decode_time = sum(decode_time_list) / num_runs
    
    print("\n" + "="*40)
    print("        ISOLATED BENCHMARK RESULTS     ")
    print("="*40)
    print(f"Model:                Qwen2.5-0.5B-Instruct (INT8)")
    print(f"Device:               Intel {device}")
    print(f"Prompt tokens:        {input_len}")
    print(f"Avg TTFT (Prefill):   {avg_ttft:.4f} seconds")
    print(f"Avg Pure Throughput:  {avg_throughput:.2f} tokens/second")
    print(f"Avg Pure Decode Time: {avg_decode_time:.4f} seconds (for {max_new_tokens - 1} tokens)")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
