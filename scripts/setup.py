#!/usr/bin/env python3
"""
팀원용 원클릭 셋업 스크립트.

데이터셋 다운로드 + 본인 담당 모델 다운로드를 한 번에 처리합니다.

Usage:
    # LLaVA-Med 담당자
    python scripts/setup.py --model llava_med --hf_token YOUR_TOKEN

    # HuatuoGPT-Vision 담당자
    python scripts/setup.py --model huatuogpt --hf_token YOUR_TOKEN

    # MedVInT-TD 담당자
    python scripts/setup.py --model medvint --hf_token YOUR_TOKEN

    # 데이터셋만 다운로드 (모델 없이)
    python scripts/setup.py --data_only --hf_token YOUR_TOKEN
"""

import argparse
import os
import sys
import subprocess
import time

# ─── 경로 설정 ───
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pmc_vqa")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
PMC_VQA_REPO_PATH = "/tmp/PMC-VQA"


def download_dataset(hf_token: str):
    """PMC-VQA test_clean 데이터셋 다운로드 (CSV + 이미지)"""
    from huggingface_hub import hf_hub_download, login

    if hf_token:
        login(token=hf_token)

    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. test_clean.csv
    csv_path = os.path.join(DATA_DIR, "test_clean.csv")
    if os.path.exists(csv_path):
        print(f"[SKIP] test_clean.csv 이미 존재: {csv_path}")
    else:
        print("[1/3] test_clean.csv 다운로드 중...")
        hf_hub_download(
            repo_id="xmcmic/PMC-VQA",
            filename="test_clean.csv",
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        print("  완료.")

    # 2. images.zip 다운로드 + 압축 해제
    images_dir = os.path.join(DATA_DIR, "images")
    zip_path = os.path.join(DATA_DIR, "images.zip")

    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 100:
        print(f"[SKIP] images/ 폴더 이미 존재 ({len(os.listdir(images_dir))}개 파일)")
    else:
        if not os.path.exists(zip_path):
            print("[2/3] images.zip 다운로드 중 (~18.9GB, xet 가속 사용)...")
            print("       시간이 걸릴 수 있습니다. 잠시 기다려주세요...")
            t0 = time.time()
            hf_hub_download(
                repo_id="xmcmic/PMC-VQA",
                filename="images.zip",
                repo_type="dataset",
                local_dir=DATA_DIR,
            )
            elapsed = time.time() - t0
            print(f"  완료. ({elapsed/60:.1f}분)")
        else:
            print(f"[SKIP] images.zip 이미 존재")

        # 압축 해제
        if os.path.exists(zip_path):
            print("[3/3] images.zip 압축 해제 중...")
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DATA_DIR)
            print("  완료.")
            # zip 삭제 (디스크 절약)
            os.remove(zip_path)
            print("  images.zip 삭제 (디스크 절약)")

    # 검증
    verify_dataset()


def verify_dataset():
    """다운로드된 데이터셋 검증"""
    import pandas as pd

    csv_path = os.path.join(DATA_DIR, "test_clean.csv")
    if not os.path.exists(csv_path):
        print("[ERROR] test_clean.csv 없음!")
        return False

    df = pd.read_csv(csv_path)
    found = 0
    for _, row in df.iterrows():
        fig = row["Figure_path"]
        for loc in [DATA_DIR, os.path.join(DATA_DIR, "images")]:
            if os.path.exists(os.path.join(loc, fig)) or os.path.exists(
                os.path.join(loc, os.path.basename(fig))
            ):
                found += 1
                break

    print(f"\n[검증] 이미지: {found}/{len(df)}")
    if found == len(df):
        print("[검증] 모든 이미지 확인 완료!")
        return True
    elif found == 0:
        print("[검증] WARNING: 이미지를 찾을 수 없습니다. 디렉토리 구조를 확인하세요.")
        return False
    else:
        print(f"[검증] WARNING: {len(df) - found}개 이미지 누락")
        return False


# ─── 모델 다운로드 함수들 ───


def setup_llava_med(hf_token: str):
    """LLaVA-Med-1.5 셋업"""
    print("\n" + "=" * 50)
    print("  LLaVA-Med-1.5 셋업")
    print("=" * 50)

    # 1. llava 라이브러리 설치 (--no-deps: torch 버전 충돌 방지)
    print("[1/2] llava 라이브러리 설치 중...")
    try:
        import llava
        print("  [SKIP] llava 이미 설치됨")
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
             "git+https://github.com/haotian-liu/LLaVA.git"],
            check=True,
        )
        print("  완료.")

    # 2. 모델 가중치 다운로드 (transformers cache에 자동 저장)
    print("[2/2] LLaVA-Med 모델 가중치 다운로드 중...")
    print("       microsoft/llava-med-v1.5-mistral-7b (~14GB)")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="microsoft/llava-med-v1.5-mistral-7b",
        token=hf_token,
    )
    print("  완료. (HuggingFace cache에 저장됨)")


def setup_huatuogpt(hf_token: str):
    """HuatuoGPT-Vision-7B 셋업"""
    print("\n" + "=" * 50)
    print("  HuatuoGPT-Vision-7B 셋업")
    print("=" * 50)

    HUATUOGPT_REPO = "/tmp/HuatuoGPT-Vision"

    # 1. HuatuoGPT-Vision 레포 클론 (모델 코드 필요)
    print(f"[1/2] HuatuoGPT-Vision 레포 클론 → {HUATUOGPT_REPO}")
    if os.path.exists(HUATUOGPT_REPO):
        print(f"  [SKIP] 이미 존재")
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git",
             HUATUOGPT_REPO],
            check=True,
        )
        print("  완료.")

    # 2. 모델 가중치 다운로드
    print("[2/2] HuatuoGPT-Vision-7B 모델 가중치 다운로드 중...")
    print("       FreedomIntelligence/HuatuoGPT-Vision-7B (~14GB)")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="FreedomIntelligence/HuatuoGPT-Vision-7B",
        token=hf_token,
    )
    print("  완료. (HuggingFace cache에 저장됨)")


def setup_medvint(hf_token: str):
    """MedVInT-TD 셋업"""
    print("\n" + "=" * 50)
    print("  MedVInT-TD 셋업")
    print("=" * 50)

    # 1. PMC-VQA 깃허브 레포 클론 (모델 코드 필요)
    print(f"[1/2] PMC-VQA 레포 클론 중 → {PMC_VQA_REPO_PATH}")
    if os.path.exists(PMC_VQA_REPO_PATH):
        print(f"  [SKIP] 이미 존재: {PMC_VQA_REPO_PATH}")
    else:
        subprocess.run(
            ["git", "clone", "https://github.com/xiaoman-zhang/PMC-VQA.git", PMC_VQA_REPO_PATH],
            check=True,
        )
        print("  완료.")

    # 2. MedVInT-TD 체크포인트 다운로드
    ckpt_dir = os.path.join(CHECKPOINT_DIR, "MedVInT-TD")
    print(f"[2/2] MedVInT-TD 체크포인트 다운로드 중 → {ckpt_dir}")
    if os.path.exists(ckpt_dir) and any(
        f.endswith((".bin", ".pt")) for f in os.listdir(ckpt_dir)
    ):
        print(f"  [SKIP] 체크포인트 이미 존재")
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="xmcmic/MedVInT-TD",
            local_dir=ckpt_dir,
            token=hf_token,
        )
        print("  완료.")

    # PMC-LLaMA 베이스 모델도 필요
    print("[추가] PMC-LLaMA 베이스 모델 다운로드 중...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="chaoyi-wu/PMC_LLAMA_7B",
        token=hf_token,
    )
    print("  완료.")


# ─── 메인 ───


MODEL_SETUP = {
    "llava_med": setup_llava_med,
    "huatuogpt": setup_huatuogpt,
    "medvint": setup_medvint,
}


def main():
    parser = argparse.ArgumentParser(
        description="MedAI-project 원클릭 셋업 (데이터셋 + 모델 다운로드)"
    )
    parser.add_argument(
        "--model",
        choices=["llava_med", "huatuogpt", "medvint"],
        help="담당 모델 (생략하면 데이터셋만 다운로드)",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace 토큰 (없으면 환경변수 HF_TOKEN 사용)",
    )
    parser.add_argument(
        "--data_only",
        action="store_true",
        help="데이터셋만 다운로드 (모델 다운로드 건너뜀)",
    )
    parser.add_argument(
        "--skip_data",
        action="store_true",
        help="데이터셋 다운로드 건너뜀 (모델만 다운로드)",
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
    if not hf_token:
        print("WARNING: HF 토큰이 없습니다. --hf_token 또는 HF_TOKEN 환경변수를 설정하세요.")
        print("         일부 모델 다운로드가 실패할 수 있습니다.\n")

    print("=" * 50)
    print("  MedAI-project Setup")
    print("=" * 50)
    print(f"  프로젝트 루트: {PROJECT_ROOT}")
    print(f"  데이터 경로:   {DATA_DIR}")
    if args.model:
        print(f"  담당 모델:     {args.model}")
    print()

    # 1. 데이터셋 다운로드
    if not args.skip_data:
        print(">>> 데이터셋 다운로드")
        download_dataset(hf_token)

    # 2. 모델 다운로드
    if args.model and not args.data_only:
        MODEL_SETUP[args.model](hf_token)

    # 3. 완료
    print("\n" + "=" * 50)
    print("  셋업 완료!")
    print("=" * 50)
    if not args.skip_data:
        print("  데이터셋: OK")
    if args.model and not args.data_only:
        print(f"  모델({args.model}): OK")
    print()
    print("  다음 단계:")
    print("    1. python scripts/setup_check.py --model {모델명}")
    print("    2. python scripts/verify_models.py --model {모델명}")
    print("    3. python scripts/run_experiment.py --model {모델명} --gpu 0")


if __name__ == "__main__":
    main()
