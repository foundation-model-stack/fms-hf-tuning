# Standard
from typing import List
import argparse

# First Party
# import this because of alot of internal contants
from scripts.benchmarks.benchmark import DIR_SAMP_CONFIGS, gather_report


def main(
    *directories: str,
    output_filename: str = "results.csv",
    remove_columns: List[str] = None,
    keep_columns: List[str] = None,
):
    "gather outputs from a list of directories and output to a csv"

    df, constant = gather_report(directories, raw=False)

    errors = []
    try:
        # remove error messages if any
        errors = df.error_messages
        errors = errors.loc[errors.isna() == False]
        df = df.loc[df.error_messages.isna()]
    except:
        pass

    # filter result columns to keep by the inverse of remove_columns
    if remove_columns:
        df = df[df.columns[~df.columns.isin(remove_columns)]]

    # assume keep and remove are disjoint
    kept = 0
    if keep_columns:
        for c in keep_columns:
            if c in constant:
                df[c] = constant[c]
                kept += 1

    df = df.reset_index(drop=True)
    try:
        df = df.drop("output_dir", axis=1)
    except KeyError:
        pass # output_dir not found

    df.reindex(sorted(df.columns), axis=1).to_csv(output_filename, index=False)
    print("***************** Report Created ******************")
    print(f"Total lines: '{len(df)}'")
    print(f"Number columns included: '{len(df.columns)}'")
    print(f"Number columns excluded: '{len(constant)-kept}'")
    print(f"Excluding number of exceptions caught: '{len(errors)}'")
    print(f"Written report to '{output_filename}'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Acceleration Benchmarking Reporting Tool",
        description="This script gathers a set benchmarks to produce a CSV report",
    )
    parser.add_argument(
        "bench_outputs",
        nargs="+",
        help="list of directories from which to gather bench outputs.",
    )
    parser.add_argument(
        "--result_file",
        default="results.csv",
        help="name of final csv report file.",
    )
    parser.add_argument(
        "--remove_columns",
        nargs="*",
        help="list of columns to ignore from results.csv",
    )
    parser.add_argument(
        "--keep_columns",
        nargs="*",
        help="list of columns to always include into results.csv",
    )

    args = parser.parse_args()
    main(
        *args.bench_outputs,
        output_filename=args.result_file,
        remove_columns=args.remove_columns,
        keep_columns=args.keep_columns,
    )
