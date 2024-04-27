# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m")
model = AutoModelForCausalLM.from_pretrained("JackFram/llama-160m")
prompt = "Here are some travel tips for Tokyo:\n"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

