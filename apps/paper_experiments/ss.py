dic = {
    0: {
        'acc': 1
    },
    1: {
        'acc': 20
    },
    2: {
        'acc': 3
    }
}



print(max(dic, key=lambda k: dic[k]['acc']))
