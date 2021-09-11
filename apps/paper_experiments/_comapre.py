from src.apis import files
from src.federated.subscribers import FedSave


def filterer(tag: str) -> bool:
    return True


files.accuracies.show_saved_accuracy_plot(filterer)
# pt = FedSave.unpack()
# print(pt['basic'].history)
