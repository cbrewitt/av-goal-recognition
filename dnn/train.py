import json
import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from dnn.dataset import DATASET_MAP
from dnn.model import LSTMModel

logger = logging.getLogger()


def save_checkpoint(path, epoch, model, optimizer, losses, accs):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'accs': accs
    }, path)


def run_evaluation(model, loss_fn, data_loader, device, use_encoding=False):
    val_data = [_ for _ in data_loader][0]
    logger.info(f"Running validation")
    trajectories = val_data[0].to(device)
    target = val_data[1].to(device)
    lengths = val_data[2]
    input = pack_padded_sequence(trajectories, lengths, batch_first=True, enforce_sorted=False)

    output, (encoding, lengths) = model(input, use_encoding=use_encoding)

    val_loss = loss_fn(output, target)
    accuracy = sum(output.argmax(axis=1) == target) / target.shape[0]
    if not use_encoding:
        return val_loss, accuracy
    else:
        t = encoding.shape[1]
        encoding_losses = nn.CrossEntropyLoss(reduction="none")(
            encoding.transpose(2, 1), target.repeat(t, 1).T)
        return val_loss, accuracy, encoding_losses


def train(config):
    if hasattr(config, "config"):
        config = argparse.Namespace(**json.load(open(config.config)))
    logger.info(config)

    # Process and load the datasets
    dataset_cls = DATASET_MAP[config.dataset]
    dataset = dataset_cls(config.scenario, "train")
    data_loader = DataLoader(dataset, shuffle=config.shuffle, batch_size=min(config.batch_size, len(dataset)))
    val_dataset = dataset_cls(config.scenario, "valid")
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=len(val_dataset))

    # Create model and send to device
    model = LSTMModel(dataset.dataset.shape[-1],
                      config.hidden_dim,
                      dataset.labels.unique().shape[-1])
    logger.info(f"Model created: {str(model)}")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Create loss function
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # Create optimizer and learning rate scheduler
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.9)

    losses = []
    accs = []

    logger.info("Starting gradient descent:")
    for epoch in range(config.max_epoch):
        for i_batch, sample_batched in enumerate(data_loader):
            trajectories = sample_batched[0].to(device)
            target = sample_batched[1].to(device)
            lengths = sample_batched[2]
            input = pack_padded_sequence(trajectories, lengths, batch_first=True, enforce_sorted=False)

            optim.zero_grad()
            output, _ = model(input, use_encoding=False)
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()

            logger.info(f"Epoch: {epoch}; Step: {len(data_loader) * epoch + i_batch}; Loss: {loss.item()}")

        val_loss, accuracy = run_evaluation(model, loss_fn, val_loader, device, use_encoding=False)
        save_checkpoint(config.save_path + f"_latest.pt", epoch, model, optim, losses, accs)
        if len(losses) < 1 or val_loss < min(losses):
            save_checkpoint(config.save_path + f"_best.pt", epoch, model, optim, losses, accs)
        losses.append(val_loss.item())
        accs.append(accuracy.item())

        schedule.step(val_loss)

        logger.info(f"Validation Loss: {val_loss.item()}; Accuracy {accuracy.item()} "
                    f"LR: {optim.param_groups[0]['lr']}")
    return losses, accs


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Location of config file ending with *.json. Specifying this will"
                                                   "overwrite all other arguments.")
    parser.add_argument("--scenario", type=str, help="Scenario to train on")
    parser.add_argument("--dataset", type=str, help="Whether to use trajectory or features dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size of a sample.")
    parser.add_argument("--shuffle", "-s", action="store_true", help="Shuffle dataset before sampling")
    parser.add_argument("--max_epoch", type=int, help="The maximum number of epochs to train.")
    parser.add_argument("--lr", type=float, help="Starting learning rate.")
    parser.add_argument("--dropout", type=float, help="Dropout regularisation for the LSTM,.")
    parser.add_argument("--hidden_dim", type=int, help="Dimensions of the LSTM hidden layer.")
    parser.add_argument("--save_path", type=str, help="Save path for model checkpoints.")

    args = parser.parse_args()
    train(args)
