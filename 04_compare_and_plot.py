import glob, json
import matplotlib.pyplot as plt
import pandas as pd
import config as cfg

def main():
    files = sorted(glob.glob(f"{cfg.RESULTS_DIR}/summary_*.json"))
    if not files:
        raise SystemExit("No summary files found.")
    df = pd.DataFrame([json.load(open(f)) for f in files])
    print(df.to_string(index=False))
    df.to_csv(f"{cfg.RESULTS_DIR}/comparison_table.csv", index=False)

    if {"auto", "fp8"}.issubset(set(df["config_tag"])):
        base = df.loc[df["config_tag"] == "auto"].iloc[0]
        df["task_acc_pct_of_baseline"] = (
            100 * df["sql_exact_match_accuracy"] / base["sql_exact_match_accuracy"]
        )
        df["ppl_pct_of_baseline"] = (
            100 * base["perplexity_wikitext2"] / df["perplexity_wikitext2"]
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        x = range(len(df))
        w = 0.35
        ax.bar([i-w/2 for i in x], df["task_acc_pct_of_baseline"], w, label="SQL exact-match")
        ax.bar([i+w/2 for i in x], df["ppl_pct_of_baseline"], w, label="Inverse perplexity")
        ax.axhline(100, linestyle="--", linewidth=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["config_tag"])
        ax.set_ylabel("% of auto baseline retained")
        ax.set_title("SQL task quality vs. generic quality")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{cfg.RESULTS_DIR}/comparison_plot.png", dpi=150)

if __name__ == "__main__":
    main()
