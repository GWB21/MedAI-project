#!/usr/bin/env python3
"""
Environment & dependency checker.
Run this FIRST before anything else. Reports what's ready and what's missing.

Usage:
    python scripts/setup_check.py
    python scripts/setup_check.py --model llava_v15   # also check model-specific deps
"""

import argparse
import importlib
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("PIL", "Pillow"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("skimage", "scikit-image"),
    ("cv2", "opencv-python"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("yaml", "pyyaml"),
    ("huggingface_hub", "huggingface-hub"),
    ("peft", "peft"),
    ("sentencepiece", "sentencepiece"),
]

MODEL_DEPS = {
    "llava_v15": [("llava", "llava (pip install git+https://github.com/haotian-liu/LLaVA.git)")],
    "huatuogpt": [],  # uses trust_remote_code
    "medvint": [],     # uses cloned repo
}


def check_packages():
    print("[1] Python Packages")
    print("-" * 50)
    all_ok = True
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  [OK]   {pip_name:20s}  {ver}")
        except ImportError:
            print(f"  [MISS] {pip_name:20s}  <- pip install {pip_name}")
            all_ok = False
    return all_ok


def check_gpu():
    print("\n[2] GPU / CUDA")
    print("-" * 50)
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [WARN] CUDA not available -- CPU only")
            return False
        n_gpus = torch.cuda.device_count()
        print(f"  [OK]   CUDA available, {n_gpus} GPU(s)")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_mem / (1024 ** 3)
            print(f"         GPU {i}: {props.name} ({vram_gb:.1f} GB)")
            if vram_gb < 24:
                print(f"         [WARN] GPU {i}: <24GB VRAM, may OOM on 7B models")
        print(f"  [OK]   PyTorch CUDA: {torch.version.cuda}")
        print(f"  [OK]   cuDNN: {torch.backends.cudnn.version()}")
        return True
    except ImportError:
        print("  [MISS] torch not installed")
        return False


def check_disk():
    print("\n[3] Disk Space")
    print("-" * 50)
    usage = shutil.disk_usage(".")
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    print(f"  Free: {free_gb:.1f} GB / {total_gb:.1f} GB")
    if free_gb < 80:
        print(f"  [WARN] < 80GB free. Need ~80GB for model + data + results")
    else:
        print(f"  [OK]   Sufficient disk space")
    return free_gb >= 80


def check_data():
    print("\n[4] PMC-VQA Data")
    print("-" * 50)
    data_dir = "./data/pmc_vqa"
    csv_path = os.path.join(data_dir, "test_clean.csv")
    if not os.path.exists(csv_path):
        print(f"  [MISS] {csv_path}")
        print(f"         Run: bash data/download_data.sh")
        return False

    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"  [OK]   test_clean.csv: {len(df)} samples")

    # Check images
    found = 0
    for _, row in df.iterrows():
        fig = row["Figure_path"]
        for loc in [data_dir, os.path.join(data_dir, "images")]:
            if os.path.exists(os.path.join(loc, fig)) or os.path.exists(os.path.join(loc, os.path.basename(fig))):
                found += 1
                break
    print(f"  [{'OK' if found == len(df) else 'WARN'}]   Images: {found}/{len(df)}")
    return found > 0


def check_config():
    print("\n[5] Experiment Config")
    print("-" * 50)
    import yaml
    config_path = "configs/experiment_config.yaml"
    if not os.path.exists(config_path):
        print(f"  [MISS] {config_path}")
        return False
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check sigmas
    for p in config.get("perturbations", []):
        if p["name"] in ("lpf", "hpf"):
            sigma = p.get("sigma")
            if sigma is None:
                print(f"  [WARN] {p['name']}.sigma is null -- run calibrate_lpf.py first")
            else:
                print(f"  [OK]   {p['name']}.sigma = {sigma}")

    print(f"  [OK]   Config loaded")
    return True


def check_model_deps(model_name: str):
    print(f"\n[6] Model-Specific: {model_name}")
    print("-" * 50)

    if model_name == "llava_v15":
        try:
            import llava
            print("  [OK]   llava library installed")
            return True
        except ImportError:
            print("  [MISS] llava library")
            print("         pip install git+https://github.com/haotian-liu/LLaVA.git")
            return False

    elif model_name == "huatuogpt":
        print("  [OK]   HuatuoGPT uses trust_remote_code (no extra deps)")
        return True

    elif model_name == "medvint":
        repo_path = "/tmp/PMC-VQA"
        if os.path.exists(repo_path):
            print(f"  [OK]   PMC-VQA repo at {repo_path}")
        else:
            print(f"  [MISS] PMC-VQA repo at {repo_path}")
            print(f"         git clone https://github.com/xiaoman-zhang/PMC-VQA.git {repo_path}")
            return False

        ckpt_dir = "./checkpoints/MedVInT-TD"
        if os.path.exists(ckpt_dir):
            print(f"  [OK]   Checkpoint dir: {ckpt_dir}")
        else:
            print(f"  [MISS] Checkpoint: {ckpt_dir}")
            print(f"         huggingface-cli download xmcmic/MedVInT-TD --local-dir {ckpt_dir}")
            return False
        return True

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llava_v15", "huatuogpt", "medvint"], default=None)
    args = parser.parse_args()

    print("=" * 50)
    print("  MedAI-project Environment Check")
    print("=" * 50)

    results = {
        "packages": check_packages(),
        "gpu": check_gpu(),
        "disk": check_disk(),
        "data": check_data(),
        "config": check_config(),
    }

    if args.model:
        results["model_deps"] = check_model_deps(args.model)

    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "NEEDS ATTENTION"
        print(f"  {name:15s}  [{status}]")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All checks passed! Ready to run experiments.")
    else:
        print("\n  Some checks failed. Fix the issues above before running.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
