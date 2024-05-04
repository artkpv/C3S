"""
Generate report from results_data.csv files in directories of experiments.

Author: Artyom Karpov, www.artkpv.net
"""

# %%

import sys
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from IPython.display import display
import plotly.graph_objects as go
import plotly
import json
import plotly.express as px


def main(dirs):
    """Given directories with results from experiments, in argv,
    it creates an aggregate report."""
    if not dirs:
        print("Usage: python report.py <dir1> <dir2> ...")
        return
    accuracies = pd.DataFrame()
    probe_evals = []
    for dir in dirs:
        extract_results(accuracies, probe_evals, dir)

    # draw_violin_plots(accuracies)

    draw_questions_forgetting(probe_evals)


def extract_results(accuracies, probe_evals, results_dir):
    results_data = Path(results_dir) / "results_data.csv"
    cfg_path = Path(results_dir) / ".hydra/config.yaml"
    if not results_data.exists():
        print(f"{results_data} does not exist.")
        return
    if not cfg_path.exists():
        print(f"{cfg_path} does not exist.")
        return
    cfg = OmegaConf.load(cfg_path)
    if "model" not in cfg:
        print(f"Model not found in {cfg_path}")
        return
    if "name" not in cfg.model:
        print(f"Model name not found in {cfg_path}")
        return
    df = pd.read_csv(results_data)

    df["model.name"] = cfg.model.name
    df["model.layer"] = cfg.model.layer
    accuracies = pd.concat([accuracies, df], ignore_index=True)

    eval_log_path = next(Path(results_dir).glob("*_eval.log"), None)
    probe_evals.append({})
    if eval_log_path:
        # For each line that contains 'known_questions':
        with open(eval_log_path) as f:
            for line in f:
                if "'known_questions'" in line:
                    try:
                        # Parse python dump object:
                        probe = eval(line[line.index("{") :])
                        probe_name = f"{probe['method']}_{probe['method_dataset']}"
                        probe["cfg"] = cfg
                        probe_evals[-1][probe_name] = probe
                    except Exception as e:
                        print(f"Error parsing probe eval log: {e}: \n{line}")


def draw_questions_forgetting(probe_evals):
    df = pd.DataFrame(columns=["kept_num", "kept_ratio", "method_dataset", "model"])
    for probe_eval in probe_evals:
        if "CCS_one" not in probe_eval:
            continue
        # Simple (one) probe:
        one_probe = probe_eval["CCS_one"]
        for probe_name in probe_eval:
            if probe_name == "CCS_one":
                continue
            if "CCS" not in probe_name:
                continue
            probe = probe_eval[probe_name]
            kept = set(one_probe["known_questions"]) & set(probe["known_questions"])

            df.loc[len(df)] = {
                "kept_num": len(kept),
                "kept_ratio": len(kept) / len(set(one_probe["known_questions"])),
                "method_dataset": probe["method_dataset"],
                "model": probe["cfg"].model.name,
            }

    if len(df) == 0:
        print("No CCS probes found.")
        return

    display(df)
    display(df.describe())
    display(df[df["model"] == 'llama370b'].describe())

    for model in df["model"].unique():
        print(model)
        new_df = df[df["model"] == model]
        # Display in bar plot, mean and std for kept_num grouped by method_dataset and model:
        new_df = (
            new_df.groupby(["method_dataset", "model"])
            .agg(
                mean_kept_ratio=("kept_ratio", "mean"),
                std_kept_ratio=("kept_ratio", "std"),
            )
            .reset_index()
        )

        fig = px.bar(
            new_df,
            x="method_dataset",
            y="mean_kept_ratio",
            error_y="std_kept_ratio",
            color="method_dataset",
            title=f"Fraction of the questions of CCS (one) probes.",
            labels={
                'mean_kept_ratio': 'Fraction',
                'method_dataset': 'Probe train dataset ',
                },
        )
        # Hide legend:
        fig.update_layout(showlegend=False)
        
        fig.show()


def draw_violin_plots(accuracies):
    for model_name in accuracies["model.name"].unique():
        print(model_name)
        # Create a new dataframe with only the rows that have model_name:
        new_df = accuracies[accuracies["model.name"] == model_name]
        datasets = new_df["dataset"].unique()
        method_datasets = new_df["method_dataset"].unique()
        methods = new_df["method"].unique()

        datasets_figs = []
        col_colors = ["blue", "green", "red", "purple", "orange", "brown"]
        count = 0
        for dataset in datasets:
            datasets_figs.append([])

            def add_plot(sub_df, name=None):
                if sub_df.empty:
                    return 0
                datasets_figs[-1].append(
                    go.Violin(
                        y=sub_df["metric_value"],
                        points="all",
                        pointpos=0,
                        meanline={"visible": True},
                        name=name or f"{dataset}, {method} ({method_dataset})",
                        # Set color:
                        marker={"color": col_colors[count % len(col_colors)]},
                    ),
                )
                return 1

            for mi, method in enumerate(methods):
                for method_dataset in method_datasets:
                    # Create a new dataframe with only the rows that have method and dataset:
                    sub_df = new_df[
                        (new_df["method"] == method)
                        & (new_df["dataset"] == dataset)
                        & (new_df["method_dataset"] == method_dataset)
                    ]
                    count += add_plot(sub_df)
            # Add Random method:
            sub_df = new_df[
                (new_df["method"] == "Random") & (new_df["dataset"] == dataset)
            ]
            count += add_plot(sub_df, name=f"{dataset}, Random")

        fig = go.Figure()
        cols = max(1, max(len(x) for x in datasets_figs))
        rows = len(datasets_figs)
        fig = plotly.tools.make_subplots(cols=cols, rows=rows, shared_yaxes=True)
        for i, dataset_figs in enumerate(datasets_figs):
            for j, figs in enumerate(dataset_figs):
                fig.add_trace(figs, row=i + 1, col=j + 1)
        # Set number of rows:
        fig.update_layout(height=175 * (count // cols + 1))
        fig.update_layout(showlegend=False)
        fig.show()

    # For method_dataset column, where it is null, fill it with empty string:
    accuracies["method_dataset"] = accuracies["method_dataset"].fillna("")

    # Make final table with all results where foreach dataset and method, there is a column with the mean of the metric_value, and a count of the number of experiments:
    accuracies = (
        accuracies.groupby(
            ["model.name", "model.layer", "method", "method_dataset", "dataset"]
        )
        .agg(
            mean_metric_value=("metric_value", "mean"),
            std_metric_value=("metric_value", "std"),
            count=("metric_value", "count"),
        )
        .reset_index()
    )

    # Make mean_metric_value as a percentage:
    # final['mean_metric_value'] = (final['mean_metric_value'] * 100).fillna(0).round(1)
    # Add std to the mean in format  00.0±00.0:
    accuracies["mean_metric_value"] = (
        (accuracies["mean_metric_value"] * 100).round(1).astype(str)
        + "±"
        + (accuracies["std_metric_value"] * 100).round(1).astype(str)
    )

    # Remove model.name column, rename columns:
    accuracies = accuracies.drop(columns=["model.name", "model.layer"])
    accuracies = accuracies.rename(
        columns={
            #'model.layer': 'Layer',
            "dataset": "Dataset",
            "method": "Method",
            "method_dataset": "Method Dataset",
            "mean_metric_value": "Accuracy (%)",
            "count": "Count",
        }
    )

    # Reorder columns, start with Dataset and sort rows accordingly to the order of columns:
    accuracies = accuracies[
        [
            "Dataset",
            "Method",
            "Method Dataset",
            "Accuracy (%)",
            "Count",
        ]
    ].sort_values(
        by=[
            "Dataset",
            "Method",
            "Method Dataset",
        ]
    )

    display(accuracies)


if __name__ == "__main__":
    dirs = sys.argv[1:]
    main([l.strip() for l in open('artifacts/all_results.txt', 'r').readlines()])

# %%
