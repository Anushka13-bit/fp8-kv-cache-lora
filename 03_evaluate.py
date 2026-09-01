import argparse
import json
import re
import config as cfg

def normalize_sql(query):
    query = query.strip().lower().rstrip(";").strip()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"\s*([(),])\s*", r"\1", query)
    return query

def score_sql(path):
    total = correct = 0
    mismatches = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            total += 1
            if normalize_sql(row["gold_sql"]) == normalize_sql(row["pred_sql"]):
                correct += 1
            else:
                mismatches.append(row)
    return correct / total if total else 0.0, total, mismatches

def main(tag):
    acc, total, mismatches = score_sql(
        f"{cfg.RESULTS_DIR}/sql_preds_{tag}.jsonl"
    )
    ppl = json.load(
        open(f"{cfg.RESULTS_DIR}/perplexity_{tag}.json")
    )["perplexity"]

    summary = {
        "config_tag": tag,
        "sql_exact_match_accuracy": acc,
        "num_test_examples": total,
        "perplexity_wikitext2": ppl,
    }
    with open(f"{cfg.RESULTS_DIR}/summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{cfg.RESULTS_DIR}/mismatches_{tag}.jsonl", "w") as f:
        for row in mismatches[:20]:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    main(parser.parse_args().tag)
