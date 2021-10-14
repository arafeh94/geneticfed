import pickle

from matplotlib import pyplot as plt

from libs.model.cv.cnn import Cifar10Model
from src import tools
from src.apis import lambdas
from src.data.data_loader import preload, urls
from src.data.data_provider import PickleDataProvider

data = PickleDataProvider(urls['cifar10']).collect().map(lambdas.reshape((32, 32, 3))).map(
    lambdas.transpose((2, 0, 1))).shuffle(47).as_tensor().split(0.8)
train_data = data[0]
test_data = data[1]
print(data)

model = Cifar10Model()
tools.train(model, train_data.batch(50), epochs=10, lr=0.001)
print(tools.infer(model, test_data.batch(50)))
pickle.dump(model.state_dict(), open('mdl', 'wb'))

for i in range(10):
    print(f'round: {i}')
    loaded_state_dict = pickle.load(open('mdl', 'rb'))
    model = Cifar10Model()
    model.load_state_dict(loaded_state_dict)
    tools.train(model, train_data.batch(50), epochs=10, lr=0.001)
    print(tools.infer(model, test_data.batch(50)))
    pickle.dump(model.state_dict(), open('mdl', 'wb'))
