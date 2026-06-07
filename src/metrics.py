import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Base models
MODELS = [
    "Llama-3.2-1B-Instruct",
    "Qwen1.5-MoE-A2.7B",
]

SCHEMES = ["Baseline", "W8A8", "W4A16", "FP8", "NVFP4"]

COLORS = {
    "Baseline": "#4A90D9",
    "W8A8": "#E06C75",
    "W4A16": "#98C379",
    "FP8": "#E5C07B",
    "NVFP4": "#C678DD",
}

def load_guidellm_results(results_dir):
    data = {}
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


def plot_tps(data, output_path="metrics/tps.png"):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 6), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        schemes_present = []
        values = []
        colors = []
        for scheme in SCHEMES:
            key = f"{model}-{scheme}" if scheme != "Baseline" else model
            if key in data:
                schemes_present.append(scheme)
                values.append(data[key]["tps"])
                colors.append(COLORS[scheme])

        ax.bar(schemes_present, values, color=colors)
        ax.set_title(model, fontsize=11)
        ax.set_ylabel("Tokens/s" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Output Token Throughput by Quantization Scheme", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def plot_ttft_vs_tps(data, output_path="metrics/ttft_vs_tps.png"):
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
            s=100,
            label=f"{scheme} ({'DO' if 'Llama' in model else 'MoE'})",
            edgecolors="black",
            linewidth=0.5,
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")
    ax.set_xlabel("Output Tokens/s")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token vs. Throughput")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def plot_disk_sizes(sizes, output_path="metrics/disk_sizes.png"):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 6), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        schemes_present = []
        values = []
        colors = []
        for scheme in SCHEMES:
            key = f"{model}-{scheme}" if scheme != "Baseline" else model
            if key in sizes:
                schemes_present.append(scheme)
                values.append(sizes[key])
                colors.append(COLORS[scheme])

        ax.bar(schemes_present, values, color=colors)
        ax.set_title(model, fontsize=11)
        ax.set_ylabel("Size (GB)" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Model Size on Disk by Quantization Scheme", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="metrics/guidellm", help="Directory with guidellm JSON results")
    parser.add_argument("--output-dir", default="metrics", help="Directory for output plots")
    parser.add_argument("--model-dir", default="output", help="Directory with quantized models for disk size")
    args = parser.parse_args()

    data = load_guidellm_results(args.results_dir)
    sizes = load_disk_sizes(args.model_dir)

    if data:
        plot_tps(data, os.path.join(args.output_dir, "tps.png"))
        plot_ttft_vs_tps(data, os.path.join(args.output_dir, "ttft_vs_tps.png"))
    if sizes:
        plot_disk_sizes(sizes, os.path.join(args.output_dir, "disk_sizes.png"))
