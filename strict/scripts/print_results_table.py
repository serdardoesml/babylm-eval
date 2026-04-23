#!/usr/bin/env python3
"""Print an aggregate markdown table of strict-track results from results/.

Results are read from:
  results/{model}/main/{eval_type}/causal/{task}/{subtask}/best_temperature_report.txt

The average accuracy is taken from the line following '### AVERAGE ACCURACY'.
"""

from pathlib import Path


def parse_average_score(report_path: Path) -> float | None:
    """Return the average score from a best_temperature_report.txt file.

    Handles both '### AVERAGE ACCURACY' and '### AVERAGE SPEARMAN'S RHO' headers.
    """
    lines = report_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("### AVERAGE"):
            if i + 1 < len(lines):
                try:
                    return float(lines[i + 1].strip())
                except ValueError:
                    return None
    return None


def fmt_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main():
    results_dir = Path(__file__).parent.parent / "results"

    # {model: {row_key: score}}
    # row_key = "{eval_type}/{task}/{subtask}"
    all_models: dict[str, dict[str, float]] = {}
    row_order: list[str] = []

    for report in sorted(results_dir.glob("*/main/*/causal/*/*/best_temperature_report.txt")):
        # parts: results / model / main / eval_type / causal / task / subtask / filename
        parts = report.parts
        model = parts[-7]
        eval_type = parts[-5]
        task = parts[-3]
        subtask = parts[-2]

        score = parse_average_score(report)
        if score is None:
            continue

        row_key = f"{eval_type}/{task}/{subtask}"
        if row_key not in row_order:
            row_order.append(row_key)

        if model not in all_models:
            all_models[model] = {}
        all_models[model][row_key] = score

    if not all_models:
        print("No results found.")
        return

    models = sorted(all_models.keys())

    print(fmt_row(["task"] + models))
    print(fmt_row(["---"] + ["---"] * len(models)))

    for row_key in row_order:
        vals = [all_models[model].get(row_key) for model in models]
        best = max((v for v in vals if v is not None), default=None)
        row = [row_key]
        for val in vals:
            if val is None:
                row.append("")
            elif val == best:
                row.append(f"**{val:.2f}**")
            else:
                row.append(f"{val:.2f}")
        print(fmt_row(row))


if __name__ == "__main__":
    main()
