#!/usr/bin/env python3
"""ROCm GPU benchmark collector — runs alongside training in background.

Samples rocm-smi every N seconds, records GPU util%, VRAM, temperature,
and power. Also captures training FPS from the training log.

Usage:
    python benchmark_collect.py --log /tmp/train_output.log --output benchmark/

    # Stop:
    kill $(cat /tmp/benchmark_pid)
"""
import subprocess, time, json, csv, os, argparse, re, signal, sys
from datetime import datetime


def parse_rocm_smi():
    """Parse rocm-smi output to extract GPU metrics."""
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram",
             "--showtemp", "--showpower", "--json"],
            stderr=subprocess.DEVNULL, timeout=10)
        data = json.loads(out)
        card = list(data.values())[0] if data else {}
        return {
            "timestamp": datetime.now().isoformat(),
            "gpu_util_pct": float(card.get("GPU use (%)", 0)),
            "vram_used_mb": float(card.get("VRAM Total Used Memory (B)", 0)) / 1e6,
            "vram_total_mb": float(card.get("VRAM Total Memory (B)", 0)) / 1e6,
            "temp_c": float(card.get("Temperature (Sensor edge) (C)", 0)),
            "power_w": float(card.get("Average Graphics Package Power (W)", 0)),
        }
    except Exception as e:
        return {"timestamp": datetime.now().isoformat(), "error": str(e)}


def parse_training_fps(log_path):
    """Parse latest FPS/steps info from training log."""
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "fps" in line.lower() or "Total time" in line:
                return {"raw_line": line.strip()}
            if "Learning iteration" in line:
                return {"raw_line": line.strip()}
        return None
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to training output log")
    parser.add_argument("--output", default="./benchmark/", help="Output directory")
    parser.add_argument("--interval", type=int, default=5, help="Sample interval (seconds)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "gpu_samples.csv")
    json_path = os.path.join(args.output, "gpu_samples.json")

    with open("/tmp/benchmark_pid", "w") as f:
        f.write(str(os.getpid()))

    print(f"Benchmark collector started (PID={os.getpid()})")
    print(f"  Log: {args.log}")
    print(f"  Output: {csv_path}")
    print(f"  Interval: {args.interval}s")

    samples = []
    csv_file = open(csv_path, "w", newline="")
    fieldnames = ["timestamp", "gpu_util_pct", "vram_used_mb", "vram_total_mb",
                  "temp_c", "power_w", "train_log"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    try:
        while True:
            sample = parse_rocm_smi()
            train_info = parse_training_fps(args.log)
            sample["train_log"] = train_info.get("raw_line", "") if train_info else ""

            writer.writerow(sample)
            csv_file.flush()
            samples.append(sample)
            print(f"  [{sample['timestamp']}] GPU={sample.get('gpu_util_pct', '?')}%  "
                  f"VRAM={sample.get('vram_used_mb', 0):.0f}/{sample.get('vram_total_mb', 0):.0f}MB  "
                  f"Temp={sample.get('temp_c', '?')}C  Pwr={sample.get('power_w', '?')}W",
                  file=sys.stderr)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        csv_file.close()
        with open(json_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"\nCollected {len(samples)} samples. Saved to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
