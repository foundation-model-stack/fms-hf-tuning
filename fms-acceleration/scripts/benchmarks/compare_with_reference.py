# Standard
from copy import copy
import argparse
import os

# Third Party
from numpy import linalg
import matplotlib.pyplot as plt
import pandas as pd

# default columns to compare
DEFAULT_PLOT_COLUMNS = [
    "mem_torch_mem_alloc_in_bytes",
    "mem_peak_torch_mem_alloc_in_bytes",
    "train_loss",
    "train_tokens_per_second",
]
# Used as combined identifier of experiment
DEFAULT_INDICES = [
    "framework_config",
    "peft_method",
    "model_name_or_path",
    "num_gpus",
    "per_device_train_batch_size",
]

DEFAULT_IGNORED_COLUMNS = [
    "epoch",
    "train_runtime",
    "train_steps_per_second",
    "train_samples_per_second",
    "mem_nvidia_mem_reserved",
]

DEFAULT_REFERENCE_FILEPATH = "scripts/benchmarks/refs/a100_80gb.csv"
BENCHMARK_FILENAME = "benchmarks.csv"
RAW_FILENAME = "raw_summary.csv"
OUTLIERS_FILENAME = "outliers.csv"


def plot_chart(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y, s=10)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axline((0, 0), slope=1)


def compare_results(df, ref, plot_columns, threshold_ratio=0.1):
    num_plots = len(plot_columns)

    charts = []
    total_outliers = []
    # filter ref to only those rows seen in df
    ref = ref[ref.index.isin(df.index.tolist())]
    for idx in range(num_plots):
        _, ax = plt.subplots(figsize=(8, 8))
        column = plot_columns[idx]
        assert (column in ref.columns) and (
            column in df.columns
        ), f"Column Name `{column}` not in Dataframe"

        ref_series = ref[column].fillna(0)
        df_series = df[column].fillna(0)
        # Extract outliers base on some threshold % difference on referance
        cmp = ref_series.to_frame()
        cmp['metric'] = column
        cmp = cmp.join(df_series.to_frame(), lsuffix='_ref')
        cmp = cmp.rename(columns={f'{column}_ref': 'reference', column: 'new'})
        cmp['ds'] = cmp.apply(
            lambda x: (
                abs(x.reference - x.new) / (x.reference + 1e-9)
            ), axis=1
        )
        outliers = cmp[cmp.ds > threshold_ratio]
        outliers = outliers.drop('ds', axis=1)

        plot_chart(
            ax,
            cmp['reference'],
            cmp['new'],
            title=f"Metric: {column}",
            xlabel="Reference",
            ylabel="New",
        )
        charts.append((ax, f"compare-{column}.jpg"))
        total_outliers.append(outliers)

    outliers_df = pd.concat(total_outliers)
    return outliers_df, outliers, charts


def read_df(file_path_or_dataframe, indices, plot_columns):
    if isinstance(file_path_or_dataframe, str):
        df = pd.read_csv(file_path_or_dataframe)
    else:
        df = file_path_or_dataframe
    df.set_index(indices, inplace=True)
    # all other columns not for plotting or explicitly ignored are hyperparameters
    argument_columns = [
        col
        for col in df.columns
        if col not in (DEFAULT_IGNORED_COLUMNS + DEFAULT_PLOT_COLUMNS)
    ]
    return df[plot_columns], df[argument_columns]


def main(
    result_dir, reference_benchmark_filepath, plot_columns, threshold_ratio, indices
):
    ref, args_ref = read_df(reference_benchmark_filepath, indices, plot_columns)

    # NOTE: this is a bit of a hack, if the new bench is a smaller bench, then we
    # supplement the data from the raw summary
    new_benchmark_filepath = os.path.join(result_dir, BENCHMARK_FILENAME) 
    try:
        df, args_df  = read_df(new_benchmark_filepath, indices, plot_columns)
    except KeyError:
        raw_filepath = os.path.join(result_dir, RAW_FILENAME)
        print (
            f"New '{new_benchmark_filepath}' is probably a partial bench. Supplementing "
            f"missing columns from raw data '{raw_filepath}'."
        )
        df2 = pd.read_csv(new_benchmark_filepath)
        df = pd.read_csv(raw_filepath)
        df, args_df = read_df(
            pd.concat([df, df2[[x for x in df2.columns if x not in df.columns]]], axis=1),
            indices, plot_columns
        )

    # Analyse between both sets of results and retrieve outliers
    # - this has a side effect of plotting the charts
    outliers_df, outliers, charts = compare_results(
        df, ref, plot_columns, threshold_ratio=threshold_ratio
    )
    # this logic is brittle and will not hold if new benchmark is not 
    # of the exact same format as the reference benchmark,
    # so put a try-catch. 
    try:
        # Find arguments that are different between ref and new
        # to highlight as possible cause of anomaly
        diff = args_df.compare(args_ref, align_axis=1).rename(
            columns={"self": "new", "other": "ref"}, level=-1
        )
        diff = diff[diff.index.isin([outlier for outlier in outliers])]
        if not diff.empty:
            outliers_df = outliers_df.set_index(indices).merge(
                diff, left_index=True, right_index=True
            )
    except ValueError: 
        print (
            f"New '{new_benchmark_filepath}' is probably a partial bench. So unable"
            "to properly compare if the arguments are consistent with old bench."
        )
    outliers_df.to_csv(os.path.join(result_dir, OUTLIERS_FILENAME))
    for chart, filename in charts:
        chart.figure.savefig(os.path.join(result_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Acceleration Benchmarking Comparison Tool",
        description="This script analyses benchmark outputs against a given reference",
    )
    parser.add_argument(
        "--result_dir",
        default="benchmark_outputs",
        help="benchmark result directory to use for comparison",
    )
    parser.add_argument(
        "--reference_benchmark_filepath",
        default="scripts/benchmarks/refs/a100_80gb.csv",
        help="file path of the csv to compare on",
    )
    parser.add_argument(
        "--threshold_ratio",
        default=0.1,
        help="the acceptable relative difference from the reference value.",
    )

    parser.add_argument(
        "--indices",
        default=DEFAULT_INDICES,
        nargs="+",
        help="list of column names to use as index for merging between old and new benchmark results",
    )

    parser.add_argument(
        "--plot_columns", 
        default=DEFAULT_PLOT_COLUMNS, 
        nargs="+",
        help="list of metric names in benchmark results to analyze visually",
    )

    args = parser.parse_args()
    main(
        result_dir=args.result_dir,
        reference_benchmark_filepath=args.reference_benchmark_filepath,
        plot_columns=args.plot_columns,
        threshold_ratio=args.threshold_ratio,
        indices=args.indices,
    )
