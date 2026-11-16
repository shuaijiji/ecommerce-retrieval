#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import statistics
from pathlib import Path


def parse_key_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def parse_train_metrics(path: Path) -> dict:
    matches = []
    for line in path.read_text(errors="replace").splitlines():
        if "'train_runtime'" in line:
            matches.append(line[line.index("{") :])
    if not matches:
        raise ValueError(f"train_runtime not found in {path}")
    return ast.literal_eval(matches[-1])


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def parse_gpu_metrics(path: Path, expected_gpus: int, training_samples: int) -> dict:
    by_gpu: dict[int, dict[str, list[float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"].strip())
            stats = by_gpu.setdefault(index, {"util": [], "memory": [], "power": []})
            stats["util"].append(float(row["utilization_gpu_percent"].strip()))
            stats["memory"].append(float(row["memory_used_mib"].strip()))
            stats["power"].append(float(row["power_draw_w"].strip()))
    if len(by_gpu) != expected_gpus:
        raise ValueError(f"expected {expected_gpus} GPUs in {path}, found {sorted(by_gpu)}")
    per_gpu = {}
    for index, stats in sorted(by_gpu.items()):
        stats = {key: values[-training_samples:] for key, values in stats.items()}
        per_gpu[str(index)] = {
            "util_median_percent": round(statistics.median(stats["util"]), 2),
            "util_p10_percent": round(percentile(stats["util"], 0.10), 2),
            "memory_peak_mib": round(max(stats["memory"]), 2),
            "power_median_w": round(statistics.median(stats["power"]), 2),
            "power_peak_w": round(max(stats["power"]), 2),
            "samples": len(stats["util"]),
        }
    return per_gpu


def parse_cpu_metrics(path: Path, training_samples: int) -> dict:
    cpu, rss = [], []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cpu.append(float(row["cpu_percent_sum"]))
            rss.append(float(row["rss_mib"]))
    if not cpu:
        raise ValueError(f"no CPU samples in {path}")
    cpu = cpu[-training_samples:]
    rss = rss[-training_samples:]
    return {
        "cpu_median_percent": round(statistics.median(cpu), 2),
        "cpu_peak_percent": round(max(cpu), 2),
        "rss_peak_mib": round(max(rss), 2),
        "samples": len(cpu),
    }


def summarize_run(run_dir: Path) -> dict:
    info = parse_key_values(run_dir / "run_info.txt")
    train = parse_train_metrics(run_dir / "train.log")
    num_gpus = int(info["num_gpus"])
    max_steps = int(info["max_steps"])
    batch = int(info["per_device_train_batch_size"])
    monitor_interval = float(info["monitor_interval"])
    runtime = float(train["train_runtime"])
    training_samples = max(1, math.ceil(runtime / monitor_interval))
    queries = num_gpus * max_steps * batch
    query_per_second = queries / runtime
    return {
        "run_dir": str(run_dir),
        "num_gpus": num_gpus,
        "max_steps": max_steps,
        "train_runtime_seconds": round(runtime, 4),
        "query_per_second": round(query_per_second, 6),
        "equivalent_pair_per_second": round(query_per_second * 8, 6),
        "gpu": parse_gpu_metrics(run_dir / "gpu_metrics.csv", num_gpus, training_samples),
        "cpu": parse_cpu_metrics(run_dir / "cpu_metrics.csv", training_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=4, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = sorted((summarize_run(path) for path in args.run_dirs), key=lambda row: row["num_gpus"])
    counts = [run["num_gpus"] for run in runs]
    if counts != [1, 2, 4, 8]:
        raise SystemExit(f"expected exactly 1/2/4/8-GPU runs, got {counts}")

    baseline = runs[0]["query_per_second"]
    for run in runs:
        run["speedup_vs_1gpu"] = round(run["query_per_second"] / baseline, 4)

    eight_gpu_speedup = runs[-1]["speedup_vs_1gpu"]
    eight_gpu_util_ok = all(
        gpu["util_median_percent"] >= 60 for gpu in runs[-1]["gpu"].values()
    )
    result = {
        "runs": runs,
        "gate": {
            "eight_gpu_speedup": eight_gpu_speedup,
            "speedup_at_least_4x": eight_gpu_speedup >= 4,
            "eight_gpu_median_util_at_least_60_percent": eight_gpu_util_ok,
            "pass": eight_gpu_speedup >= 4 and eight_gpu_util_ok,
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
