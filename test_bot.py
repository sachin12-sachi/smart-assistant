from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Run a simple test prompt
user_input = "Hello!"
input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
output_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

print("Bot:", response)
