import os
import warnings
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

# Suppress the Mistral/Qwen regex tokenization warning cluttering the terminal
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

def main():
    model_path = "./qwen-0.5b-ov"
    device = "GPU"
    
    print(f"Loading {model_path} onto Intel GPU... Please wait.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = OVModelForCausalLM.from_pretrained(model_path, device=device, compile=True)
    
    # Initialize the chat history list
    history = []
    
    print("\n" + "="*50)
    print(" Qwen 2.5 0.5B Chat initialized! Type 'exit' to quit. ")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
                
            # 1. Append the new message to the existing conversation context
            history.append({"role": "user", "content": user_input})
            
            # 2. Format the entire history using the model's chat template
            text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt")
            
            print("Assistant: ", end="", flush=True)
            
            # 3. Generate the response
            outputs = model.generate(**inputs, max_new_tokens=512)
            
            # 4. Extract and decode only the newly generated tokens
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            print(response)
            
            # 5. Append the assistant's response to history so it remembers next turn
            history.append({"role": "assistant", "content": response})
            
            # Optional: Keep the history from growing infinitely to prevent slowdowns
            if len(history) > 20:  # Keeps last 10 turns of conversation
                history = history[-20:]
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
