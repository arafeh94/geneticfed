from src.apis import files
from src.federated.subscribers import FedSave


def filterer(tag: str) -> bool:
    return True


print(files.accuracies.get_saved_accuracy())
files.accuracies.show_saved_accuracy_plot(filterer)
