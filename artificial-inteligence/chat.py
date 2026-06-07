# save as chat.py
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

model_path = "./smollm-ov"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = OVModelForCausalLM.from_pretrained(model_path, device="GPU")
for i in range(10):
        msg = input('Lets chat:')
        messages = [{"role": "user", "content": msg}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")

        outputs = model.generate(**inputs, max_new_tokens=200)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print("Assistant:", response)
