import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D

import precog.evaluate_results as eval_precog
import lstm.test as eval_lstm
from core.base import get_data_dir

colors = list(sns.color_palette("tab10"))
markers = ["o", "x", "s", "P"]


def draw_line_with_sem(group, ax, key, value, i=0):
    accuracy = group.mean()
    accuracy.index = np.arange(accuracy.index.size)
    accuracy_sem = group.std() / np.sqrt(group.count())
    accuracy_sem.index = np.arange(accuracy_sem.index.size)
    accuracy.rename(columns={key: value}).plot(ax=ax)
    plt.fill_between(accuracy_sem.index, (accuracy + accuracy_sem)[key].to_numpy(),
                     (accuracy - accuracy_sem)[key].to_numpy(), alpha=0.2)
    xs = range(0, accuracy.shape[0])
    ys = accuracy.values
    ax.scatter(xs, ys, marker=markers[i], color=colors[i], s=80)
    return ax


def main():
    scenarios = ["round", "heckstrasse", "bendplatz", "frankenberg"]
    lstm_datasets = ["trajectory"]

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.size"] = 17

    sns.set_style("darkgrid")
    precog_results = eval_precog.main(json.load(open("../precog/evaluate_config.json")), get_dataframe=True)
    models, predictions, unique_samples, accuracy = pickle.load(open(get_data_dir() + "grit_eval_data.p", "rb"))
    for scenario_name in scenarios:
        fig, ax = plt.subplots()

        i = 0

        # Plot GRIT and prior
        for model_name, model in models.items():
            unique_samples = predictions[scenario_name][model_name]
            fraction_observed_grouped = unique_samples[['model_correct', 'fraction_observed']].groupby(
                'fraction_observed')
            draw_line_with_sem(fraction_observed_grouped, ax, i=i, key="model_correct",
                               value={"prior_baseline": "GRIT-no-DT", "trained_trees": "GRIT"}[model_name])
            i += 1

        # Plot PRECOG
        precog_scenario, timings = precog_results[scenario_name]
        fraction_observed_grouped = precog_scenario[["adj_accuracy", "rounded_frac_before_goal"]] \
            .groupby("rounded_frac_before_goal")
        draw_line_with_sem(fraction_observed_grouped, ax, "adj_accuracy", "PRECOG", i=i)
        i += 1

        # Plot LSTM
        for dataset in lstm_datasets:
            test_config = argparse.Namespace(**{
                "dataset": dataset,
                "shuffle": True,
                "scenario": scenario_name,
                "model_path": f"checkpoint/{scenario_name}_{dataset}_best.pt",
                "lstm_hidden_dim": 64,
                "fc_hidden_dim": 725,
                "lstm_layers": 1,
                "step": 0.1
            })
            lstm_corrects, lstm_probs, _ = eval_lstm.main(test_config)
            xs = np.arange(lstm_corrects.shape[1])
            accuracy = lstm_corrects.mean(0)
            sem = lstm_corrects.std(0) / np.sqrt(lstm_corrects.shape[0])
            ax.plot(xs, accuracy, label=f"LSTM")
            plt.fill_between(xs, accuracy + sem, accuracy - sem, alpha=0.2)
            xs = range(0, accuracy.size)
            ys = accuracy
            ax.scatter(xs, ys, marker=markers[i], color=colors[i], s=80)
            i += 1

        ax.set_yticks(np.linspace(0.0, 1.0, 11))
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)])
        ax.set_xticks(np.arange(fraction_observed_grouped.ngroups))
        ax.get_legend().remove()
        if scenario_name == "round":
            ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 1, fraction_observed_grouped.ngroups)])
            plt.xlabel('Fraction of trajectory observed')
        else:
            ax.set_xticklabels([])
        plt.ylabel('Accuracy')
        plt.title(f"{scenario_name}")
        plt.ylim([-0.1, 1.1])
        fig.tight_layout()
        fig.savefig(f"../images/{scenario_name}_accuracy.pdf", bbox_inches='tight', pad_inches=0)
        # plt.show()

        i = 0

        fig, ax = plt.subplots()
        for model_name, model in models.items():
            unique_samples = predictions[scenario_name][model_name]
            fraction_observed_grouped = unique_samples[['model_entropy_norm', 'fraction_observed']].groupby(
                'fraction_observed')
            draw_line_with_sem(fraction_observed_grouped, ax, "model_entropy_norm",
                               {"prior_baseline": "GRIT-no-DT", "trained_trees": "GRIT"}[model_name], i=i)
            i += 1

        precog_scenario, timings = precog_results[scenario_name]
        fraction_observed_grouped = precog_scenario[["entropy", "rounded_frac_before_goal"]] \
            .groupby("rounded_frac_before_goal")
        draw_line_with_sem(fraction_observed_grouped, ax, "entropy", "PRECOG", i=i)
        i += 1

        for dataset in lstm_datasets:
            test_config = argparse.Namespace(**{
                "dataset": dataset,
                "shuffle": True,
                "scenario": scenario_name,
                "model_path": f"checkpoint/{scenario_name}_{dataset}_best.pt",
                "lstm_hidden_dim": 64,
                "fc_hidden_dim": 725,
                "lstm_layers": 1,
                "step": 0.1
            })
            lstm_corrects, lstm_probs, _ = eval_lstm.main(test_config)
            xs = np.arange(lstm_corrects.shape[1])
            h_goals = lstm_probs * np.log2(lstm_probs)
            h_goals = -h_goals.sum(axis=-1)
            h_uniform = np.log2(lstm_probs.shape[-1])
            h = h_goals / h_uniform
            entropy = h.mean(0)
            sem = h.std(0) / np.sqrt(h.shape[0])
            ax.plot(xs, entropy, label=f"LSTM")
            plt.fill_between(xs, entropy + sem, entropy - sem, alpha=0.2)
            xs = range(0, accuracy.size)
            ys = entropy
            ax.scatter(xs, ys, marker=markers[i], color=colors[i], s=80)

        ax.set_yticks(np.linspace(-0.1, 1.1, 11))
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)])
        ax.set_xticks(np.arange(fraction_observed_grouped.ngroups))
        if scenario_name == "round":
            ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 1, fraction_observed_grouped.ngroups)])
            plt.xlabel('Fraction of trajectory observed')
        else:
            ax.set_xticklabels([])
        plt.ylabel('Normalised Entropy')
        plt.title(f"{scenario_name}")
        plt.ylim([-0.1, 1.1])
        if scenario_name == "heckstrasse":
            custom_lines = [Line2D([0], [0], color=c, marker=m, lw=2, markersize=12) for c, m in zip(colors, markers)]
            plt.legend(custom_lines, ["GRIT-no-DT", "GRIT", "GR-ESP", "LSTM"])
        else:
            ax.get_legend().remove()
        fig.tight_layout()
        fig.savefig(f"../images/{scenario_name}_entropy.pdf", bbox_inches='tight', pad_inches=0)
        # plt.show()


if __name__ == '__main__':
    main()
