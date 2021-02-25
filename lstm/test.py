import argparse
import json

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from lstm.model import LSTMModel
from lstm.train import load_save_dataset, logger


def main(config):
    torch.random.manual_seed(42)

    if hasattr(config, "config"):
        config = argparse.Namespace(**json.load(open(config.config)))
    logger.info(config)

    test_dataset = load_save_dataset(config, "test")
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=len(test_dataset))
    test_data = [_ for _ in test_loader][0]
    logger.info(f"Running testing")

    model_dict = torch.load(config.model_path)
    model = LSTMModel(test_dataset.dataset.shape[-1],
                      config.lstm_hidden_dim,
                      config.fc_hidden_dim,
                      test_dataset.labels.unique().shape[-1])
    model.load_state_dict(model_dict["model_state_dict"])

    trajectories = test_data[0]
    target = test_data[1]
    lengths = test_data[2]
    input = pack_padded_sequence(trajectories, lengths, batch_first=True, enforce_sorted=False)

    model.eval()

    output, (encoding, lengths) = model(input, use_encoding=True)

    matches = (encoding.argmax(axis=-1) == target.unsqueeze(-1)).to(float)
    mask = (torch.arange(encoding.shape[1])[None, :] >= lengths[:, None])
    matches = matches.masked_fill(mask, 0)

    step = config.step
    step_mask = torch.arange(encoding.shape[1])[None, :] % (lengths[:, None] * step - 1).ceil() == 0
    step_mask = step_mask.masked_fill(mask, 0)
    steps = step_mask.to(float).sum(1)
    count = int(1 / step + 1)
    assert (steps == count).all()
    accuracy = matches.masked_select(step_mask).view((matches.shape[0], count))

    return accuracy.detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Location of config file ending with *.json. Specifying this will"
                                                   "overwrite all other arguments.")
    parser.add_argument("--scenario", type=str, help="Scenario to train on")
    parser.add_argument("--dataset", type=str, help="Whether to use trajectory or features dataset")
    parser.add_argument("--shuffle", "-s", action="store_true", help="Shuffle dataset before sampling")
    parser.add_argument("--save_path", type=str, help="Save path for model checkpoints.")

    args = parser.parse_args()

    main(args)