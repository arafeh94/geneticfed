import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.model.base_model import BaseModel
from src.apis import federated_tools
from src.apis.ewc import ElasticWeightConsolidation
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from src.data.data_loader import preload

train, test = preload('mnist').as_tensor().split(0.8)

crit = nn.CrossEntropyLoss()

ewc = ElasticWeightConsolidation(BaseModel(28 * 28, 100, 10), crit=crit, lr=1e-4)
from tqdm import tqdm

for _ in range(4):
    for input, target in tqdm(train.batch(100)):
        ewc.forward_backward_update(input, target)
ewc.register_ewc_params(train, 100, 400)

model_2 = BaseModel(28 * 28, 100, 10)
federated_tools.train(model_2, train_data=train.batch(100), epochs=4, lr=1e-4)


def accu(model, dataloader):
    model = model.eval()
    acc = 0
    for input, target in dataloader:
        o = model(input)
        acc += (o.argmax(dim=1).long() == target).float().mean()
    return acc / len(dataloader)


print(accu(ewc.model, test.batch(50)))
print(federated_tools.infer(model_2, test.batch(50)))
