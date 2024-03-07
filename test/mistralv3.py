import pickle
import sys

from pympler import asizeof
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda"  # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model_inputs = tokenizer(["My favourite condiment is"], return_tensors="pt")


mistral_1 = pickle.load(open("mistralai_base", 'rb'))
mistral_2 = pickle.load(open("mistralai_base", 'rb'))

first_batch = pickle.load(open("mistralai_1", 'rb'))
second_batch = pickle.load(open("mistralai_2", 'rb'))

mistral_1.base_model.layers.extend(first_batch)
mistral_2.base_model.layers.extend(first_batch)


first_predict = mistral_1.generate(**model_inputs, max_new_tokens=100, do_sample=True)
second_predict = mistral_2.generate(first_predict, max_new_tokens=100, do_sample=True)

print(tokenizer.batch_decode(second_predict)[0])
