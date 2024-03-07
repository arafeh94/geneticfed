import pickle
import sys

from pympler import asizeof
from torch.nn import ModuleList
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

for i, layer in enumerate(model.base_model.layers):
    pickle.dump(layer, open(f"mistralai_{i}.pkl", 'wb'))
model.base_model.layers = ModuleList()
pickle.dump(model, open(f"mistralai_0.pkl", 'wb'))
