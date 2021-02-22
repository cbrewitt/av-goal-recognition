import itertools
import argparse
import sys
import os
import numpy as np
import logging

from dnn.train import train

grid_search_params = {
    "hidden_dim": np.logspace(1, 14, 14, base=2, dtype=int),
    "lr": np.logspace(-5, -1, 10),
    "dropout": np.arange(0.0, 1.0, 0.1)
}


def setup_logging(logger):
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


logger = logging.getLogger()
setup_logging(logger)

if __name__ == '__main__':
    dataset = sys.argv[1]
    scenario = sys.argv[2]
    i = int(sys.argv[3])

    if not os.path.exists("grid_search"):
        os.mkdir("grid_search")

    grid_params = list(product_dict(**grid_search_params))
    params = grid_params[i]
    params.update({"dataset": dataset, "shuffle": True, "batch_size": 1000, "max_epoch": 1000, "scenario": scenario,
                   "save_path": f"grid_search/{scenario}_{dataset}_{params['hidden_dim']}_{params['lr']:.3f}_{params['dropout']:.1f}"})
    params = argparse.Namespace(**params)
    train(params)
