import copy
import logging
import random
import sys
import matplotlib.pyplot as plt

from torch import nn

from libs.model.base_model import BaseModel
from libs.model.linear.lr import LogisticRegression
from src.apis import federated_tools, utils
from src.apis.ewc import ElasticWeightConsolidation
from src.data.data_loader import preload

epoch, batch = 4, 9999
model = BaseModel(28 * 28, 100, 10).to('cuda')
ewc = ElasticWeightConsolidation(copy.deepcopy(model), nn.CrossEntropyLoss(), lr=0.01, weight=0.1)
train, test = preload('mnist').split(0.8)
train = train.as_tensor('cuda')
test = test.as_tensor()

tasks = [
    train.filter(lambda x, y: y == 0),
    train.filter(lambda x, y: y == 1),
    train.filter(lambda x, y: y == 2),
    train.filter(lambda x, y: y == 3),
    train.filter(lambda x, y: y == 4),
    train.filter(lambda x, y: y == 5),
    train.filter(lambda x, y: y == 6),
    train.filter(lambda x, y: y == 7),
    train.filter(lambda x, y: y == 8),
    train.filter(lambda x, y: y == 9),
]

for key, item in enumerate(tasks):
    tasks[key] = tasks[key].as_tensor('cuda')


def model_train(task, is_ewc):
    if is_ewc:
        for epochs in range(epoch):
            for input, target in task.batch(batch):
                ewc.model = ewc.model.to('cuda')
                ewc.forward_backward_update(input, target)
            ewc.register_ewc_params(task, batch, 300)
        res = federated_tools.infer(ewc.model, test.batch(batch))
    else:
        global model
        model = model.to('cuda')
        federated_tools.train(model, task.batch(batch), lr=0.01, epochs=epoch, logging=False)
        res = federated_tools.infer(model, test.batch(batch))
    print('\tis ewc: {}, results: {}'.format(is_ewc, res))
    return res


acc_ewc = []
acc_nor = []
for i in range(50):
    print('-------------R{}-Start-------------'.format(i))
    res_ewc = None
    res_nor = None
    for index, task in enumerate(tasks):
        print('\t-------------T{}-Start-------------'.format(index))
        res_nor = model_train(task, False)
        res_ewc = model_train(task, True)
        print('\t-------------T{}-End-------------\n'.format(index))
    acc_ewc.append(res_ewc[0])
    acc_nor.append(res_nor[0])
    print('-------------R{}-End-------------\n'.format(i))

plt.plot(range(len(acc_ewc)), acc_ewc, label='EWC', marker='o')
plt.plot(range(len(acc_ewc)), acc_nor, label='Normal', marker='+')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('EWC Enabled vs Normal, Epochs: {}, Try: {}'.format(epoch, 2))
plt.legend()

# Display the plot
plt.show()
