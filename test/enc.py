from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload

mnist = preload('mnist', ShardDistributor(50, 1))

m1 = train(mnist[0])
test(mnist[0], m1)

m2 = train(mnist[1])
test(mnist[2], m2)

aggregate(m1, m2)
