import pandas as pd


def main():
    fname = "counts_pre_filtering.csv"
    counts = pd.read_csv(fname)
    counts.lang_pair = counts.lang_pair.str.upper()
    counts.columns = ['Language pair', "Split", "Sentences"]

    latex_table = counts.to_latex(index=False)
    print(latex_table)


if __name__ == "__main__":
    main()
