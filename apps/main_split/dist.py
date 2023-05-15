from src.data.data_distributor import PipeDistributor


def clustered(client_cluster_size, client_sample_size=600):
    pipes = [
        PipeDistributor.pick_by_label_id([1, 2], client_sample_size, client_cluster_size),
        PipeDistributor.pick_by_label_id([3, 4], client_sample_size, client_cluster_size),
        PipeDistributor.pick_by_label_id([5, 6], client_sample_size, client_cluster_size),
        PipeDistributor.pick_by_label_id([7, 8], client_sample_size, client_cluster_size),
        PipeDistributor.pick_by_label_id([9, 0], client_sample_size, client_cluster_size),
    ]
    return PipeDistributor(pipes)



