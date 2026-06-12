import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    "Llama-3.2-1B-Instruct",
    "deepseek-moe-16b-chat",
]

MODEL_LABELS = {
    "Llama-3.2-1B-Instruct": "Llama 3.2 1B Instruct (Decoder-Only)",
    "deepseek-moe-16b-chat": "DeepSeek-MoE 16B Chat (MoE)",
}

SCHEMES = ["Baseline", "W8A8", "W4A16", "FP8", "NVFP4"]

COLORS = {
    "Baseline": "#4A90D9",
    "W8A8": "#E06C75",
    "W4A16": "#98C379",
    "FP8": "#E5C07B",
    "NVFP4": "#C678DD",
}


def _model_scheme_key(model, scheme):
    return f"{model}-{scheme}"


def _slug(model):
    return model.lower().replace(".", "").replace("-", "_")


def load_lm_eval_results(results_dir):
    data = {}
    if not os.path.isdir(results_dir):
        return data
    for entry in os.listdir(results_dir):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        results_file = None
        for root, _, files in os.walk(entry_path):
            for f in sorted(files, reverse=True):
                if f.startswith("results") and f.endswith(".json"):
                    results_file = os.path.join(root, f)
                    break
            if results_file:
                break
        if not results_file:
            continue
        with open(results_file) as f:
            result = json.load(f)
        results_block = result.get("results", {})
        ppl = None
        for task_key in ["wikitext", "wikitext2"]:
            if task_key in results_block:
                ppl = results_block[task_key].get("word_perplexity,none")
                if ppl is None:
                    ppl = results_block[task_key].get("word_perplexity")
                break
        if ppl is not None:
            data[entry] = {"perplexity": ppl}
    return data



def _bar_chart(schemes, values, colors, ylabel, title, output_path, annotate_fmt=".1f"):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(schemes, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:{annotate_fmt}}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_quality(data, output_dir):
    for model in MODELS:
        schemes, values, colors = [], [], []
        for scheme in SCHEMES:
            key = _model_scheme_key(model, scheme)
            if key in data:
                schemes.append(scheme)
                values.append(data[key]["perplexity"])
                colors.append(COLORS[scheme])
        if not values:
            continue
        label = MODEL_LABELS.get(model, model)
        _bar_chart(schemes, values, colors,
                   "Perplexity (lower is better)",
                   f"Perplexity (wikitext) — {label}",
                   os.path.join(output_dir, f"perplexity_{_slug(model)}.png"))


MODEL_SHORT = {
    "Llama-3.2-1B-Instruct": "Llama 1B (DO)",
    "deepseek-moe-16b-chat": "DeepSeek 16B (MoE)",
}

QUANT_SCHEMES = ["W8A8", "FP8", "NVFP4", "W4A16"]


def plot_pct_change_perplexity(quality, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(QUANT_SCHEMES))
    width = 0.35
    offsets = [-width / 2, width / 2]

    active_models = []
    for model in MODELS:
        baseline_key = _model_scheme_key(model, "Baseline")
        if baseline_key in quality:
            active_models.append(model)

    if not active_models:
        plt.close()
        return

    for i, model in enumerate(active_models):
        baseline_key = _model_scheme_key(model, "Baseline")
        baseline_val = quality[baseline_key]["perplexity"]
        pct_changes = []
        for scheme in QUANT_SCHEMES:
            key = _model_scheme_key(model, scheme)
            if key in quality:
                val = quality[key]["perplexity"]
                pct = ((val - baseline_val) / baseline_val) * 100
                pct_changes.append(pct)
            else:
                pct_changes.append(0)

        bar_colors = ["#F44336" if pct > 0 else "#4CAF50" for pct in pct_changes]

        bars = ax.bar(x + offsets[i], pct_changes, width * 0.9,
                      color=bar_colors, edgecolor="white", linewidth=0.5,
                      alpha=1.0 if i == 0 else 0.65,
                      hatch="" if i == 0 else "//",
                      label=MODEL_SHORT.get(model, model))
        for bar, pct in zip(bars, pct_changes):
            va = "bottom" if pct >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{pct:+.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("% Change vs Baseline", fontsize=11)
    ax.set_title("Perplexity — % Change vs Baseline", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(QUANT_SCHEMES, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "pct_change_perplexity.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="metrics/guidellm", help="Directory with guidellm JSON results")
    parser.add_argument("--lm-eval-dir", default="metrics/lm-eval", help="Directory with lm-eval results")
    parser.add_argument("--output-dir", default="metrics", help="Directory for output plots")
    args = parser.parse_args()

    quality = load_lm_eval_results(args.lm_eval_dir)

    if quality:
        plot_quality(quality, args.output_dir)
        plot_pct_change_perplexity(quality, args.output_dir)
