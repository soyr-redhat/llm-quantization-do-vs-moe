import csv
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    "Llama-3.2-1B-Instruct",
    "Qwen1.5-MoE-A2.7B",
]

MODEL_LABELS = {
    "Llama-3.2-1B-Instruct": "Llama 3.2 1B Instruct (Decoder-Only)",
    "Qwen1.5-MoE-A2.7B": "Qwen 1.5 MoE A2.7B (MoE)",
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
    return model if scheme == "Baseline" else f"{model}-{scheme}"


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
            for f in files:
                if f == "results.json":
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


def load_guidellm_results(results_dir):
    data = {}
    if not os.path.isdir(results_dir):
        return data
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            result = json.load(f)

        name = fname.replace(".json", "")
        benchmarks = result.get("benchmarks", [result])
        bench = benchmarks[-1] if isinstance(benchmarks, list) else benchmarks

        data[name] = {
            "tps": bench.get("output_token_throughput", 0),
            "ttft": bench.get("ttft_mean", 0) or bench.get("time_to_first_token_mean", 0),
        }
    return data


def load_disk_sizes(output_dir):
    sizes = {}
    for dirpath, _, filenames in os.walk(output_dir):
        model_name = os.path.basename(dirpath)
        total = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for f in filenames
        )
        if total > 0:
            sizes[model_name] = total / (1024 ** 3)
    return sizes


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


def plot_tps(data, output_dir):
    for model in MODELS:
        schemes, values, colors = [], [], []
        for scheme in SCHEMES:
            key = _model_scheme_key(model, scheme)
            if key in data:
                schemes.append(scheme)
                values.append(data[key]["tps"])
                colors.append(COLORS[scheme])
        if not values:
            continue
        label = MODEL_LABELS.get(model, model)
        _bar_chart(schemes, values, colors,
                   "Tokens/s", f"Output Token Throughput — {label}",
                   os.path.join(output_dir, f"tps_{_slug(model)}.png"))


def plot_ttft_vs_tps(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, vals in data.items():
        parts = key.rsplit("-", 1)
        scheme = parts[-1] if len(parts) > 1 and parts[-1] in SCHEMES else "Baseline"
        model = parts[0] if scheme != "Baseline" else key

        marker = "o" if "Llama" in model else "s"
        ax.scatter(
            vals["tps"],
            vals["ttft"] * 1000,
            c=COLORS.get(scheme, "#999"),
            marker=marker,
            s=120,
            label=f"{scheme} ({'DO' if 'Llama' in model else 'MoE'})",
            edgecolors="black",
            linewidth=0.5,
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=9)
    ax.set_xlabel("Output Tokens/s")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token vs. Throughput", fontsize=13)
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


def plot_disk_sizes(sizes, output_dir):
    for model in MODELS:
        schemes, values, colors = [], [], []
        for scheme in SCHEMES:
            key = _model_scheme_key(model, scheme)
            if key in sizes:
                schemes.append(scheme)
                values.append(sizes[key])
                colors.append(COLORS[scheme])
        if not values:
            continue
        label = MODEL_LABELS.get(model, model)
        _bar_chart(schemes, values, colors,
                   "Size (GB)", f"Model Size on Disk — {label}",
                   os.path.join(output_dir, f"disk_sizes_{_slug(model)}.png"),
                   annotate_fmt=".2f")


def export_summary_csv(perf, quality, sizes, output_path="metrics/summary.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for model in MODELS:
        arch = "Decoder-Only" if "Llama" in model else "MoE"
        for scheme in SCHEMES:
            key = _model_scheme_key(model, scheme)
            row = {
                "model": model,
                "architecture": arch,
                "scheme": scheme,
                "size_gb": f"{sizes.get(key, 0):.2f}" if key in sizes else "",
                "tps": f"{perf[key]['tps']:.1f}" if key in perf else "",
                "ttft_ms": f"{perf[key]['ttft'] * 1000:.1f}" if key in perf else "",
                "perplexity": f"{quality[key]['perplexity']:.2f}" if key in quality else "",
            }
            if any(row[c] for c in ["size_gb", "tps", "ttft_ms", "perplexity"]):
                rows.append(row)

    if not rows:
        return

    fields = ["model", "architecture", "scheme", "size_gb", "tps", "ttft_ms", "perplexity"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="metrics/guidellm", help="Directory with guidellm JSON results")
    parser.add_argument("--lm-eval-dir", default="metrics/lm-eval", help="Directory with lm-eval results")
    parser.add_argument("--output-dir", default="metrics", help="Directory for output plots")
    parser.add_argument("--model-dir", default="output", help="Directory with quantized models for disk size")
    args = parser.parse_args()

    data = load_guidellm_results(args.results_dir)
    quality = load_lm_eval_results(args.lm_eval_dir)
    sizes = load_disk_sizes(args.model_dir)

    if data:
        plot_tps(data, args.output_dir)
        plot_ttft_vs_tps(data, os.path.join(args.output_dir, "ttft_vs_tps.png"))
    if quality:
        plot_quality(quality, args.output_dir)
    if sizes:
        plot_disk_sizes(sizes, args.output_dir)

    export_summary_csv(data, quality, sizes, os.path.join(args.output_dir, "summary.csv"))
