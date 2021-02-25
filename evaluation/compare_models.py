import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import precog.evaluate_results as eval_precog
# import lstm.evaluate_results as eval_lstm
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
    scenarios = ["heckstrasse", "bendplatz", "frankenberg"]

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

        ax.set_xticks(np.arange(fraction_observed_grouped.ngroups))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 1, fraction_observed_grouped.ngroups)])
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)])
        plt.xlabel('Fraction of trajectory observed')
        plt.ylabel('Accuracy ({})'.format(scenario_name))
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

        ax.set_xticks(np.arange(fraction_observed_grouped.ngroups))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 1, fraction_observed_grouped.ngroups)])
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)])
        plt.xlabel('Fraction of trajectory observed')
        plt.ylabel('Normalised Entropy ({})'.format(scenario_name))
        plt.ylim([0, 1])
        fig.savefig(f"../images/{scenario_name}_entropy.pdf")
        # plt.show()


if __name__ == '__main__':
    main()
