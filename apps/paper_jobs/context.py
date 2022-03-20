import argparse
import hashlib


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, help='epochs count')
    parser.add_argument('-b', '--batch', type=int, help='batch count')
    parser.add_argument('-r', '--round', type=int, help='number of rounds')
    parser.add_argument('-cr', '--clients_ratio', type=int, help='selected client percentage for fl')
    parser.add_argument('-lr', '--learn_rate', type=float, help='learn rate')
    parser.add_argument('-t', '--tag', type=str, help='tag to save the results')
    parser.add_argument('-cn', '--nb_clusters', type=int, help='number of clusters', default=1)
    parser.add_argument('-cs', '--chrm_size', type=int, help='chromosome', default=10)
    return parser.parse_args()


def hashed(parsed_args):
    return hashlib.md5(str(vars(parsed_args)).encode()).hexdigest()
