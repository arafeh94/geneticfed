import sys

from pympler import asizeof
from transformers import AutoModelForCausalLM, AutoTokenizer

import pickle
import sys

from pympler import asizeof
from torch.nn import ModuleList
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

first_batch = model.base_model.layers[:3]
second_batch = model.base_model.layers[3:]
model.base_model.layers = ModuleList()
pickle.dump(model, open("mistralai_base", 'wb'))
pickle.dump(first_batch, open("mistralai_1", 'wb'))
pickle.dump(second_batch, open("mistralai_2", 'wb'))

# MistralForCausalLM(
#   (model): MistralModel(
#     (embed_tokens): Embedding(32000, 4096)
#     (layers): ModuleList(
#       (0-31): 32 x MistralDecoderLayer(
#         (self_attn): MistralAttention(
#           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
#           (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
#           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (rotary_emb): MistralRotaryEmbedding()
#         )
#         (mlp): MistralMLP(
#           (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
#           (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
#           (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): MistralRMSNorm()
#         (post_attention_layernorm): MistralRMSNorm()
#       )
#     )
#     (norm): MistralRMSNorm()
#   )
#   (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
# )