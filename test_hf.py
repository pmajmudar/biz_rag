from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


model = AutoModelForCausalLM.from_pretrained(
    "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs)
out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(out)