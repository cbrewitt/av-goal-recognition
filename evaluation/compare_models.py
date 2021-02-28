import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
import precog.evaluate_results as eval_precog
import lstm.test as eval_lstm
from core.base import get_data_dir


def draw_line_with_sem(group, ax, key, value):
    accuracy = group.mean()
    accuracy.index = np.arange(accuracy.index.size)
    accuracy_sem = group.std() / np.sqrt(group.count())
    accuracy_sem.index = np.arange(accuracy_sem.index.size)
    accuracy.rename(columns={key: value}).plot(ax=ax)
    plt.fill_between(accuracy_sem.index, (accuracy + accuracy_sem)[key].to_numpy(),
                     (accuracy - accuracy_sem)[key].to_numpy(), alpha=0.2)
    return ax


def main():
    scenarios = ["heckstrasse", "bendplatz", "frankenberg", "round"]
    lstm_datasets = ["trajectory"]

    sns.set_style("darkgrid")
    precog_results = eval_precog.main(json.load(open("../precog/evaluate_config.json")), get_dataframe=True)
    models, predictions, unique_samples, accuracy = pickle.load(open(get_data_dir() + "grit_eval_data.p", "rb"))
    for scenario_name in scenarios:
        fig, ax = plt.subplots()

        # Plot GRIT and prior
        for model_name, model in models.items():
            unique_samples = predictions[scenario_name][model_name]
            fraction_observed_grouped = unique_samples[['model_correct', 'fraction_observed']].groupby(
                'fraction_observed')
            draw_line_with_sem(fraction_observed_grouped, ax,
                               "model_correct", {"prior_baseline": "GRIT-no-DT", "trained_trees": "GRIT"}[model_name])

        # Plot PRECOG
        precog_scenario = precog_results[scenario_name]
        fraction_observed_grouped = precog_scenario[["adj_accuracy", "rounded_frac_before_goal"]] \
            .groupby("rounded_frac_before_goal")
        draw_line_with_sem(fraction_observed_grouped, ax, "adj_accuracy", "PRECOG")

        # Plot LSTM
        for dataset in lstm_datasets:
            test_config = argparse.Namespace(**{
                "dataset": dataset,
                "shuffle": True,
                "scenario": scenario_name,
                "model_path": f"checkpoint/{scenario_name}_{dataset}_best.pt",
                "lstm_hidden_dim": 100,
                "fc_hidden_dim": 500,
                "lstm_layers": 2,
                "step": 0.1
            })
            lstm_corrects, lstm_probs = eval_lstm.main(test_config)
            xs = np.arange(lstm_corrects.shape[1])
            accuracy = lstm_corrects.mean(0)
            sem = lstm_corrects.std(0) / np.sqrt(lstm_corrects.shape[0])
            ax.plot(xs, accuracy, label=f"LSTM-{dataset}")
            plt.fill_between(xs, accuracy + sem, accuracy - sem, alpha=0.2)

        ax.set_xticks(np.arange(fraction_observed_grouped.ngroups))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 1, fraction_observed_grouped.ngroups)])
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)])
        ax.get_legend().remove()
        plt.xlabel('Fraction of trajectory observed')
        plt.ylabel('Accuracy')
        plt.title(f"{scenario_name}")
        plt.ylim([0, 1])
        fig.savefig(f"../images/{scenario_name}_accuracy.pdf")
        # plt.show()

        fig, ax = plt.subplots()
        for model_name, model in models.items():
            unique_samples = predictions[scenario_name][model_name]
            fraction_observed_grouped = unique_samples[['model_entropy', 'fraction_observed']].groupby(
                'fraction_observed')
            draw_line_with_sem(fraction_observed_grouped, ax, "model_entropy",
                               {"prior_baseline": "GRIT-no-DT", "trained_trees": "GRIT"}[model_name])

        precog_scenario = precog_results[scenario_name]
        fraction_observed_grouped = precog_scenario[["entropy", "rounded_frac_before_goal"]] \
            .groupby("rounded_frac_before_goal")
        draw_line_with_sem(fraction_observed_grouped, ax, "entropy", "PRECOG")

        for dataset in lstm_datasets:
            test_config = argparse.Namespace(**{
                "dataset": dataset,
                "shuffle": True,
                "scenario": scenario_name,
                "model_path": f"checkpoint/{scenario_name}_{dataset}_best.pt",
                "lstm_hidden_dim": 100,
                "fc_hidden_dim": 500,
                "lstm_layers": 2,
                "step": 0.1
            })
            lstm_corrects, lstm_probs = eval_lstm.main(test_config)
            xs = np.arange(lstm_corrects.shape[1])
            h_goals = lstm_probs * np.log2(lstm_probs)
            h_goals = -h_goals.sum(axis=-1)
            h_uniform = np.log2(lstm_probs.shape[-1])
            h = h_goals / h_uniform
            entropy = h.mean(0)
            sem = h.std(0) / np.sqrt(h.shape[0])
            ax.plot(xs, entropy, label=f"LSTM-{dataset}")
            plt.fill_between(xs, entropy + sem, entropy - sem, alpha=0.2)

        ax.set_xticks(np.arange(fraction_observed_grouped.ngroups))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 1, fraction_observed_grouped.ngroups)])
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)])
        plt.xlabel('Fraction of trajectory observed')
        plt.ylabel('Normalised Entropy'.format(scenario_name))
        plt.title(f"{scenario_name}")
        plt.ylim([0, 1])
        plt.legend()
        fig.savefig(f"../images/{scenario_name}_entropy.pdf")
        # plt.show()


if __name__ == '__main__':
    main()
