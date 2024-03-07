import pickle
import sys

from pympler import asizeof
from torch.nn import ModuleList
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model_inputs = tokenizer(["My favourite condiment is"], return_tensors="pt")

model = pickle.load(open('mistralai_0.pkl', 'rb'))
for i in range(32):
    layers = pickle.load(open(f'mistralai_{i}.pkl', 'rb'))
    model.base_model.layers.append(layers)
    model_inputs = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    model.base_model.layers.clear()
print(model_inputs)
