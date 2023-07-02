from src.apis.extensions import Wandb

arr1 = [1, 2, 3, 4, 5, 6, 7, 1, 3, 4, 5, 2, 34, 3]
arr2 = [5, 2, 3, 1, 4, 5, 6, 7, 3, 2, 1, ]
for i in range(len(arr1)):
    Wandb.instance('arr1').log({'val': arr1[i]})
for i in range(len(arr2)):
    Wandb.instance('arr2').log({'val': arr2[i]})
