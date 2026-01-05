import argparse

import os
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pu_labels", type=str, default="data/pu_labels.csv")
    parser.add_argument("--year_csv", type=str, default="MAX_MX_year.csv")
    parser.add_argument("--year_cutoff", type=int, default=2024)
    parser.add_argument("--output_labels", type=str, default="output_labels.csv")

    args = parser.parse_args()

    assert os.path.exists(args.pu_labels)
    assert os.path.exists(args.year_csv)
    assert 2011 <= args.year_cutoff

    return args


def main():
    args = get_args()

    print("Loading data")
    pu_labels = pd.read_csv(args.pu_labels, index_col=0)
    year_df = pd.read_csv(args.year_csv)


    print(f"Cutting off after {args.year_cutoff}")

    cutoff_max_phases = year_df.loc[year_df["year"]>args.year_cutoff, "MAX"].to_list()

    print(f"The MAX phases past this cutoff year are: {cutoff_max_phases}")

    allowed_max_phases = [x for x in cutoff_max_phases if x in pu_labels.index]

    print(f"The MAX phases {allowed_max_phases} will be revised")

    # Simply remove the constraint from these samples
    pu_labels.loc[allowed_max_phases, "class"] = 0

    print(f"After cut-off, there remains {(pu_labels['class']>0).sum()} known convertible materials")

    pu_labels.to_csv(args.output_labels, index=True)


if __name__ == "__main__":
    main()