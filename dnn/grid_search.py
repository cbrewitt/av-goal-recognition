import itertools
import argparse
import sys
import os
import numpy as np
import logging
import copy
import torch
import pandas as pd
import sys
from dnn.train import train

grid_search_params = {
    "hidden_dim": np.logspace(1, 14, 14, base=2, dtype=int),
    "lr": np.logspace(-5, -1, 10),
    "dropout": np.arange(0.0, 1.0, 0.1)
}


def setup_logging(logger):
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
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
    type = sys.argv[1]  # search/best
    dataset = sys.argv[2]  # features/trajectory
    scenario = sys.argv[3]  # heckstrasse/bendplatz/frankenberg/round
    i = int(sys.argv[4])

    path_string = "grid_search/{0}_{1}_{2}_{3}_{4:.1f}"

    if type == "search":
        if not os.path.exists("grid_search"):
            os.mkdir("grid_search")

        grid_params = list(product_dict(**grid_search_params))
        params = grid_params[i]
        save_path = path_string.format(scenario, dataset, params['hidden_dim'], params['lr'], params['dropout'])
        params.update({"dataset": dataset, "shuffle": True, "batch_size": 100, "max_epoch": 2000, "scenario": scenario,
                       "save_path": save_path})
        params = argparse.Namespace(**params)
        train(params)

    elif type == "best":
        results = []
        for params in product_dict(**grid_search_params):
            save_path = path_string.format(scenario, dataset, params['hidden_dim'], params['lr'], params['dropout'])
            try:
                best = torch.load(save_path + "_best.pt")
                latest = torch.load(save_path + "_latest.pt")
            except IOError as e:
                logger.exception(str(e), exc_info=e)
                continue
            result = copy.copy(params)
            result.update({"best_loss": best["losses"].min(), "best_acc": best["accs"].max(),
                           "best_epoch": best["epoch"], "avg_loss": latest["losses"].mean(),
                           "loss_sem": latest["losses"].std() / np.sqrt(len(latest["losses"])),
                           "avg_acc": latest["accs"].mean(),
                           "accs_sem": latest["accs"].std() / np.sqrt(len(latest["accs"]))},
                          )
            results.append(result)
        results = pd.DataFrame(results)
        results.to_csv("grid_search/results.csv")

        logger.info(f"The best parameter by loss:\n{results.loc[results.idxmin()['best_loss']]}")