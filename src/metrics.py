import csv
import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from math import pi

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

        metrics = bench.get("metrics", {})
        if metrics:
            tps_block = metrics.get("output_tokens_per_second", {}).get("successful", {})
            ttft_block = metrics.get("time_to_first_token_ms", {}).get("successful", {})
            itl_block = metrics.get("inter_token_latency_ms", {}).get("successful", {})
            tps = tps_block.get("mean", 0)
            ttft = ttft_block.get("mean", 0) / 1000.0
        else:
            tps_block, ttft_block, itl_block = {}, {}, {}
            tps = bench.get("output_token_throughput", 0)
            ttft = bench.get("ttft_mean", 0) or bench.get("time_to_first_token_mean", 0)

        pct_keys = ["p50", "p75", "p90", "p95", "p99"]
        def _extract_percentiles(block):
            pcts = block.get("percentiles", {})
            return {k: pcts.get(k, 0) for k in pct_keys}

        data[name] = {
            "tps": tps,
            "ttft": ttft,
            "ttft_percentiles": _extract_percentiles(ttft_block),
            "itl_percentiles": _extract_percentiles(itl_block),
            "tps_percentiles": _extract_percentiles(tps_block),
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

        marker = "o" if "Llama" in key else "s"
        ax.scatter(
            vals["tps"],
            vals["ttft"] * 1000,
            c=COLORS.get(scheme, "#999"),
            marker=marker,
            s=120,
            label=f"{scheme} ({'DO' if 'Llama' in key else 'MoE'})",
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


MODEL_SHORT = {
    "Llama-3.2-1B-Instruct": "Llama 1B (DO)",
    "deepseek-moe-16b-chat": "DeepSeek 16B (MoE)",
}

QUANT_SCHEMES = ["W8A8", "W4A16", "FP8", "NVFP4"]


def plot_grouped_bars(perf, quality, sizes, output_dir):
    metrics_to_plot = []

    tps_data = {}
    for model in MODELS:
        vals = []
        for scheme in QUANT_SCHEMES:
            key = _model_scheme_key(model, scheme)
            vals.append(perf[key]["tps"] if key in perf else 0)
        tps_data[model] = vals
    if any(any(v > 0 for v in vals) for vals in tps_data.values()):
        metrics_to_plot.append(("tps", "Tokens/s", "Output Token Throughput", tps_data, ".0f"))

    ttft_data = {}
    for model in MODELS:
        vals = []
        for scheme in QUANT_SCHEMES:
            key = _model_scheme_key(model, scheme)
            vals.append(perf[key]["ttft"] * 1000 if key in perf else 0)
        ttft_data[model] = vals
    if any(any(v > 0 for v in vals) for vals in ttft_data.values()):
        metrics_to_plot.append(("ttft", "TTFT (ms)", "Time to First Token", ttft_data, ".1f"))

    ppl_data = {}
    for model in MODELS:
        vals = []
        for scheme in QUANT_SCHEMES:
            key = _model_scheme_key(model, scheme)
            vals.append(quality[key]["perplexity"] if key in quality else 0)
        ppl_data[model] = vals
    if any(any(v > 0 for v in vals) for vals in ppl_data.values()):
        metrics_to_plot.append(("perplexity", "Perplexity (lower is better)", "Perplexity (wikitext)", ppl_data, ".1f"))

    size_data = {}
    for model in MODELS:
        vals = []
        for scheme in QUANT_SCHEMES:
            key = _model_scheme_key(model, scheme)
            vals.append(sizes.get(key, 0))
        size_data[model] = vals
    if any(any(v > 0 for v in vals) for vals in size_data.values()):
        metrics_to_plot.append(("disk_size", "Size (GB)", "Model Size on Disk", size_data, ".2f"))

    for metric_name, ylabel, title, model_data, fmt in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(QUANT_SCHEMES))
        width = 0.35
        active_models = [m for m in MODELS if any(v > 0 for v in model_data[m])]
        offsets = np.linspace(-width / 2, width / 2, len(active_models)) if len(active_models) > 1 else [0]

        for i, model in enumerate(active_models):
            bars = ax.bar(x + offsets[i], model_data[model], width * 0.9,
                          label=MODEL_SHORT.get(model, model),
                          color=[COLORS[s] for s in QUANT_SCHEMES],
                          edgecolor="white", linewidth=0.5,
                          alpha=1.0 if i == 0 else 0.65,
                          hatch="" if i == 0 else "//")
            for bar, val in zip(bars, model_data[model]):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:{fmt}}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(QUANT_SCHEMES, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"grouped_{metric_name}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


def plot_pct_change(perf, quality, output_dir):
    for metric_name, data_dict, value_key, higher_is_better in [
        ("tps", perf, "tps", True),
        ("perplexity", quality, "perplexity", False),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(QUANT_SCHEMES))
        width = 0.35
        active_models = []
        for model in MODELS:
            baseline_key = _model_scheme_key(model, "Baseline")
            if baseline_key in data_dict:
                active_models.append(model)

        if not active_models:
            plt.close()
            continue

        offsets = np.linspace(-width / 2, width / 2, len(active_models)) if len(active_models) > 1 else [0]

        for i, model in enumerate(active_models):
            baseline_key = _model_scheme_key(model, "Baseline")
            baseline_val = data_dict[baseline_key][value_key]
            pct_changes = []
            for scheme in QUANT_SCHEMES:
                key = _model_scheme_key(model, scheme)
                if key in data_dict:
                    val = data_dict[key][value_key]
                    pct = ((val - baseline_val) / baseline_val) * 100
                    pct_changes.append(pct)
                else:
                    pct_changes.append(0)

            bar_colors = []
            for pct in pct_changes:
                if higher_is_better:
                    bar_colors.append("#4CAF50" if pct >= 0 else "#F44336")
                else:
                    bar_colors.append("#F44336" if pct > 0 else "#4CAF50")

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
        ylabel = "% Change vs Baseline"
        title_metric = "Output Token Throughput" if metric_name == "tps" else "Perplexity"
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title_metric} — % Change vs Baseline", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(QUANT_SCHEMES, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"pct_change_{metric_name}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


def plot_latency_percentiles(perf, output_dir):
    pct_keys = ["p50", "p75", "p90", "p95", "p99"]
    pct_labels = ["P50", "P75", "P90", "P95", "P99"]

    for metric_name, pct_field, ylabel, title_prefix in [
        ("ttft", "ttft_percentiles", "Latency (ms)", "TTFT Percentile Distribution"),
        ("itl", "itl_percentiles", "Latency (ms)", "Inter-Token Latency Percentile Distribution"),
    ]:
        for model in MODELS:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(pct_keys))
            n_schemes = len(QUANT_SCHEMES)
            width = 0.8 / n_schemes
            has_data = False

            for i, scheme in enumerate(QUANT_SCHEMES):
                key = _model_scheme_key(model, scheme)
                if key not in perf or pct_field not in perf[key]:
                    continue
                pcts = perf[key][pct_field]
                values = [pcts.get(k, 0) for k in pct_keys]
                if not any(v > 0 for v in values):
                    continue
                has_data = True
                offset = (i - n_schemes / 2 + 0.5) * width
                bars = ax.bar(x + offset, values, width * 0.9,
                              label=scheme, color=COLORS[scheme],
                              edgecolor="white", linewidth=0.5)
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)

            if not has_data:
                plt.close()
                continue

            label = MODEL_LABELS.get(model, model)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{title_prefix} — {label}", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(pct_labels, fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            path = os.path.join(output_dir, f"{metric_name}_percentiles_{_slug(model)}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"Saved {path}")


def plot_radar(perf, quality, sizes, output_dir):
    axes_config = [
        ("Perplexity", "perplexity", False),
        ("TPS", "tps", True),
        ("TTFT", "ttft", False),
        ("Size (GB)", "size", False),
    ]
    axis_labels = [a[0] for a in axes_config]
    n_axes = len(axes_config)
    angles = [n / n_axes * 2 * pi for n in range(n_axes)]
    angles += angles[:1]

    for model in MODELS:
        raw = {}
        for scheme in QUANT_SCHEMES:
            key = _model_scheme_key(model, scheme)
            vals = []
            vals.append(quality[key]["perplexity"] if key in quality else 0)
            vals.append(perf[key]["tps"] if key in perf else 0)
            vals.append(perf[key]["ttft"] * 1000 if key in perf else 0)
            vals.append(sizes.get(key, 0))
            if any(v > 0 for v in vals):
                raw[scheme] = vals

        if not raw:
            continue

        all_vals = list(raw.values())
        mins = [min(v[i] for v in all_vals) for i in range(n_axes)]
        maxs = [max(v[i] for v in all_vals) for i in range(n_axes)]

        normalized = {}
        for scheme, vals in raw.items():
            norm = []
            for i, (val, cfg) in enumerate(zip(vals, axes_config)):
                span = maxs[i] - mins[i]
                if span == 0:
                    n = 1.0
                else:
                    n = (val - mins[i]) / span
                    if not cfg[2]:
                        n = 1.0 - n
                norm.append(n)
            norm.append(norm[0])
            normalized[scheme] = norm

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", ""], color="grey", size=7)
        plt.ylim(0, 1.1)
        plt.xticks(angles[:-1], axis_labels, fontsize=11)

        for scheme, norm in normalized.items():
            ax.plot(angles, norm, linewidth=2, label=scheme, color=COLORS[scheme])
            ax.fill(angles, norm, alpha=0.1, color=COLORS[scheme])

        label = MODEL_LABELS.get(model, model)
        ax.set_title(f"Quantization Trade-off Profile — {label}",
                     fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.tight_layout()
        path = os.path.join(output_dir, f"radar_{_slug(model)}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


def export_summary_csv(perf, quality, sizes, output_path="metrics/summary.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for model in MODELS:
        arch = "Decoder-Only" if "Llama" in model else "MoE (DeepSeek)"
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
        plot_latency_percentiles(data, args.output_dir)
    if quality:
        plot_quality(quality, args.output_dir)
    if sizes:
        plot_disk_sizes(sizes, args.output_dir)

    if data or quality or sizes:
        plot_grouped_bars(data, quality, sizes, args.output_dir)
    if data and quality:
        plot_pct_change(data, quality, args.output_dir)
    if data and quality and sizes:
        plot_radar(data, quality, sizes, args.output_dir)

    export_summary_csv(data, quality, sizes, os.path.join(args.output_dir, "summary.csv"))
