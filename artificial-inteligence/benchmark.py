import time
import torch
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

def run_benchmark():
    model_path = "./smollm-ov"
    device = "GPU"
    
    print(f"Loading model and tokenizer from '{model_path}' onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Using compile=True ensures the model is loaded and compiled to the GPU immediately
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
    # Warmup run to compile pipelines and discard initial lazy-loading overhead
    _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    print(f"--- Starting Benchmark ({num_runs} iterations) ---")
    
    ttft_list = []
    throughput_list = []
    total_time_list = []
    
    for i in range(num_runs):
        # 1. Measure Time to First Token (TTFT)
        start_time = time.perf_counter()
        # Generate exactly 1 token to capture the prompt processing + initial token generation time
        _ = model.generate(**inputs, max_new_tokens=1)
        ttft = time.perf_counter() - start_time
        ttft_list.append(ttft)
        
        # 2. Measure overall generation speed
        start_time = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        total_time = time.perf_counter() - start_time
        
        # Calculate generated tokens
        total_tokens = outputs[0].shape[-1]
        new_tokens_generated = total_tokens - input_len
        
        # Throughput = new tokens generated / total time taken
        throughput = new_tokens_generated / total_time
        
        throughput_list.append(throughput)
        total_time_list.append(total_time)
        
        print(f"Run {i+1}: TTFT = {ttft:.4f}s | Throughput = {throughput:.2f} tok/sec | Tokens: {new_tokens_generated}")

    # --- Print Summary Statistics ---
    avg_ttft = sum(ttft_list) / num_runs
    avg_throughput = sum(throughput_list) / num_runs
    avg_total_time = sum(total_time_list) / num_runs
    
    print("\n" + "="*40)
    print("           BENCHMARK RESULTS           ")
    print("="*40)
    print(f"Model:                SmolLM2-135M-Instruct (INT8)")
    print(f"Device:               Intel {device}")
    print(f"Prompt tokens:        {input_len}")
    print(f"Avg TTFT:             {avg_ttft:.4f} seconds")
    print(f"Avg Throughput:       {avg_throughput:.2f} tokens/second")
    print(f"Avg Gen Time:         {avg_total_time:.4f} seconds (for ~{max_new_tokens} tokens)")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
