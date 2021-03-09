import argparse
import json
import pickle

import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from core.base import get_data_dir
import precog.evaluate_results as eval_precog
import lstm.test as eval_lstm

scenarios = ["heckstrasse", "bendplatz", "frankenberg", "round"]

timings_info = []
timings_df = []

precog_results = eval_precog.main(json.load(open("../precog/evaluate_config.json")), get_dataframe=True)
models, predictions, unique_samples, accuracy = pickle.load(open(get_data_dir() + "grit_eval_data.p", "rb"))
for scenario_name in scenarios:
    unique_samples = predictions[scenario_name]["trained_trees"]
    grit = pd.DataFrame(unique_samples["duration"])
    grit["scenario"] = scenario_name
    grit["model"] = "grit_trained_trees"
    timings_df.append(grit)
    timings_info.append({"scenario": scenario_name, "model": "grit_trained_trees",
                         "mean": grit["duration"].mean() * 1000, "sem": grit.sem()})

    results, timings = precog_results[scenario_name]
    timings["duration"] = (timings["rollout"] + timings["adjustment"]) / \
                              timings["n_rollout"] / timings["n_batch"] / timings["n_agents"] / timings["n_samples"]
    precog = pd.DataFrame(timings["duration"])
    precog["scenario"] = scenario_name
    precog["model"] = "gr-esp"
    timings_df.append(precog)
    timings_info.append({"scenario": scenario_name, "model": "precog",
                         "mean": timings["duration"].mean() * 1000})

    test_config = argparse.Namespace(**{
        "dataset": "trajectory",
        "shuffle": True,
        "scenario": scenario_name,
        "model_path": f"checkpoint/{scenario_name}_trajectory_best.pt",
        "lstm_hidden_dim": 64,
        "fc_hidden_dim": 725,
        "lstm_layers": 1,
        "step": 0.1
    })
    lstm_corrects, lstm_probs, duration = eval_lstm.main(test_config)
    lstm = pd.DataFrame([{"scenario": scenario_name, "model": "lstm", "duration": duration / lstm_corrects.size}])
    timings_df.append(lstm)
    timings_info.append({"scenario": scenario_name, "model": "lstm",
                         "mean": duration / lstm_corrects.size * 1000,
                         "sem": 0.0})

timings_info = pd.DataFrame(timings_info)
timings = pd.concat(timings_df, ignore_index=True)
sns.set_style("darkgrid")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 17

# fig, ax = plt.subplots()
# plt.yscale("log")
# sns.boxplot(data=timings, x="model", y="duration", hue="scenario", ax=ax)
# ax.set_xticklabels(["GRIT", "GR-ESP", "LSTM"])
# plt.ylabel("Mean inference time (ms)")
# plt.tight_layout()
# plt.savefig(f"../images/boxplot_inference_times.png", bbox_inches='tight', pad_inches=0)
# plt.show()

plt.clf()
n = timings_info["model"].unique().size
d = 1 / n
xs = np.linspace(0, 1, n)
ys = timings_info.groupby("model")["mean"].mean().values  # To miliseconds
idx = np.argsort(ys)
yerr = timings_info.groupby("model")["mean"].sem().values
plt.yscale("log")
plt.bar(xs, ys[idx], width=d)  # , color=list(sns.color_palette("tab10"))[1:])
plt.errorbar(xs, ys[idx], yerr=yerr[idx], fmt='none', ecolor="r", elinewidth=3, capsize=5)
# for x, y, yerr in zip(xs, ys[idx], yerr[idx]):
#     plt.text(x - d / 4, 0.1 * y, f"{y:.2f}", color="white")
plt.xticks(xs, np.array(["GRIT", "LSTM", "GR-ESP"])[idx])
plt.yticks(np.logspace(-1, 1, 5), ["$10^{" + f"{i:.1f}" + "}$" for i in np.linspace(-1, 1, 5)])
plt.xlabel("")
plt.ylabel("Mean inference time (ms)")
plt.tight_layout()
plt.savefig(f"../images/inference_times.pdf", bbox_inches='tight', pad_inches=0)
# plt.show()