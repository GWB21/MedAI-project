#!/usr/bin/env python3
"""
Quick verification that each model loads and runs inference correctly.
Uses a dummy image (random noise) -- NOT for actual experiment results.

Usage:
    python scripts/verify_models.py --model llava_v15 --gpu 0
    python scripts/verify_models.py --model huatuogpt --gpu 0
    python scripts/verify_models.py --model medvint --gpu 0
    python scripts/verify_models.py --model all --gpu 0
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DUMMY_PROMPT = """What is shown in this chest X-ray?
A. Pneumonia
B. Normal
C. Cardiomegaly
D. Pleural effusion
Answer with the option's letter from the given choices directly."""


def verify_model(model_name: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  Verifying: {model_name}")
    print(f"{'='*60}")

    from src.models import get_model

    try:
        # 1. Load
        print("[1/4] Loading model...")
        t0 = time.time()
        model = get_model(model_name)
        model.load(device="cuda")
        print(f"  OK - loaded in {time.time()-t0:.1f}s ({model.precision})")

        # 2. Create dummy image (224x224 random)
        print("[2/4] Creating dummy image...")
        dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        print("  OK - shape:", dummy_image.shape)

        # 3. Inference
        print("[3/4] Running inference (max_new_tokens=32)...")
        t0 = time.time()
        output = model.inference(dummy_image, DUMMY_PROMPT, max_new_tokens=32)
        print(f"  OK - {time.time()-t0:.1f}s")
        print(f"  Raw output: {repr(output.raw_text[:100])}")
        print(f"  Parsed: {output.parsed_answer}")
        print(f"  Parse success: {output.parse_success}")

        # 4. Logit extraction
        print("[4/4] Extracting choice logits...")
        logits = output.logits
        print(f"  Logits: A={logits.get('A', 'N/A'):.4f}, B={logits.get('B', 'N/A'):.4f}, "
              f"C={logits.get('C', 'N/A'):.4f}, D={logits.get('D', 'N/A'):.4f}")

        print(f"\n  >>> {model_name}: ALL CHECKS PASSED <<<")
        return True

    except Exception as e:
        print(f"\n  >>> {model_name}: FAILED <<<")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["llava_v15", "huatuogpt", "medvint", "all"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    if args.model == "all":
        models = ["llava_v15", "huatuogpt", "medvint"]
    else:
        models = [args.model]

    results = {}
    for m in models:
        results[m] = verify_model(m)
        # Free GPU memory between models
        import torch
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    for m, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {m:15s}  [{status}]")


if __name__ == "__main__":
    main()
