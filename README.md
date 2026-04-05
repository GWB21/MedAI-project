# Medical VQA Image Perturbation Experiment

의료 VQA 모델이 실제로 이미지를 분석하는지, 아니면 텍스트 prior(질문 내 모달리티/부위 정보, 선택지 통계)에 의존하는지를 **이미지 조작(perturbation) 실험**으로 검증합니다.

## Overview

| 항목 | 설정 |
|------|------|
| **데이터셋** | [PMC-VQA](https://huggingface.co/datasets/xmcmic/PMC-VQA) test_clean (2,000 samples) |
| **모델** | LLaVA-Med-1.5, HuatuoGPT-Vision-7B, MedVInT-TD |
| **이미지 조건** | Original, Black, LPF, HPF, Patch Shuffle (16×16) |
| **디코딩** | Greedy (temperature=0, do_sample=False) |

### Image Conditions

| Condition | Description |
|-----------|-------------|
| **Original** | 원본 이미지 그대로 |
| **Black** | 모든 픽셀 0 (검정 이미지) — 이미지 정보 완전 제거 |
| **LPF** | Gaussian Low-Pass Filter (sigma calibrated independently to SSIM ≈ 0.7) — 고주파 디테일 제거 |
| **HPF** | High-Pass Filter, Original - LPF + 128 (sigma calibrated independently to SSIM ≈ 0.7) — 저주파 구조 제거, 엣지만 보존 |
| **Patch Shuffle** | 16×16 패치 위치를 무작위 셔플 — 전역 구조 파괴, 로컬 텍스처 보존 |

### Diagnostic Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **VRS** | Acc(Original) - Acc(Black) | 양수: 이미지 의존 / 0: 이미지 무시 |
| **IS** | 1 - Acc(PatchShuffle) / Acc(Original) | 높음: 이미지 변화에 민감 |
| **NSR** | 1 - Acc(LPF) / Acc(Original) | 높음: 고주파 정보(디테일)에 의존 |
| **HPF_Drop** | Acc(Original) - Acc(HPF) | 큼: 저주파 정보(전체 구조)가 중요 |

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/GWB21/MedAI-project.git
cd MedAI-project
pip install -r requirements.txt
```

**모델별 추가 설치:**

```bash
# LLaVA-Med 담당자
pip install git+https://github.com/haotian-liu/LLaVA.git

# MedVInT-TD 담당자
git clone https://github.com/xiaoman-zhang/PMC-VQA.git /tmp/PMC-VQA
huggingface-cli download xmcmic/MedVInT-TD --local-dir ./checkpoints/MedVInT-TD
```

### 2. Data Download

```bash
bash data/download_data.sh
```

### 3. LPF Sigma Calibration (팀 리더가 1회 실행)

```bash
python scripts/calibrate_lpf.py --data_dir ./data/pmc_vqa --target_ssim 0.7
# LPF와 HPF 각각의 sigma가 출력됨 → configs/experiment_config.yaml에 기입
```

### 4. Model Verification (각 팀원)

```bash
# 본인 담당 모델이 정상 로드 & 추론되는지 확인
python scripts/verify_models.py --model llava_med --gpu 0
python scripts/verify_models.py --model huatuogpt --gpu 0
python scripts/verify_models.py --model medvint --gpu 0
```

### 5. Run Experiment

```bash
# 각 팀원이 담당 모델로 실행 (5개 조건 자동 순회)
python scripts/run_experiment.py --model llava_med --gpu 0
python scripts/run_experiment.py --model huatuogpt --gpu 0
python scripts/run_experiment.py --model medvint --gpu 0

# 특정 조건만 실행하려면:
python scripts/run_experiment.py --model llava_med --gpu 0 --conditions original black
```

### 6. Analyze Results

```bash
# 3명의 결과 CSV를 모아서 분석
python scripts/analyze_results.py --results results/*_all_*.csv
```

---

## Project Structure

```
MedAI-project/
├── configs/
│   └── experiment_config.yaml     # 실험 파라미터 (모든 팀원 동일하게 사용)
├── data/
│   ├── download_data.sh           # PMC-VQA 다운로드 스크립트
│   └── README.md
├── src/
│   ├── perturbations.py           # 이미지 조작 함수 (Black, LPF, HPF, Patch Shuffle)
│   ├── dataset.py                 # PMC-VQA 데이터 로더
│   ├── parse_answer.py            # 모델 출력에서 A/B/C/D 파싱
│   ├── metrics.py                 # VRS, IS, NSR, HPF_Drop 계산
│   ├── inference.py               # 메인 추론 루프
│   └── models/
│       ├── base_model.py          # 공통 인터페이스 (ABC)
│       ├── llava_med.py           # LLaVA-Med-1.5 wrapper
│       ├── huatuogpt.py           # HuatuoGPT-Vision-7B wrapper
│       └── medvint.py             # MedVInT-TD wrapper
├── scripts/
│   ├── run_experiment.py          # 실험 실행 엔트리포인트
│   ├── calibrate_lpf.py           # LPF sigma calibration
│   ├── analyze_results.py         # 결과 분석 + 테이블 생성
│   └── verify_models.py           # 모델 로드/추론 검증
├── results/                       # 결과 CSV 저장
└── docker/
    └── Dockerfile
```

## Models

| Model | HF Repo | Base | Paper |
|-------|---------|------|-------|
| **LLaVA-Med-1.5** | `microsoft/llava-med-v1.5-mistral-7b` | Mistral-7B + CLIP | [NeurIPS 2023](https://arxiv.org/abs/2306.00890) |
| **HuatuoGPT-Vision-7B** | `FreedomIntelligence/HuatuoGPT-Vision-7B` | Qwen2-7B | [EMNLP 2024](https://arxiv.org/abs/2406.19280) |
| **MedVInT-TD** | `xmcmic/MedVInT-TD` | PMC-CLIP + LLaMA | [arXiv 2305.10415](https://arxiv.org/abs/2305.10415) |

## Output CSV Format

| Column | Description |
|--------|-------------|
| `image_id` | Image identifier |
| `condition` | original / black / lpf / hpf / patch_shuffle |
| `model` | llava_med / huatuogpt / medvint |
| `question` | PMC-VQA original question |
| `choice_A~D` | Answer choices |
| `gt_answer` | Ground truth (A/B/C/D) |
| `pred_answer` | Model prediction (A/B/C/D or PARSE_FAIL) |
| `correct` | 1 if correct, 0 otherwise |
| `parse_success` | 1 if parsing succeeded |
| `raw_output` | Raw model output text |
| `logit_A~D` | Per-choice logit values |

## Hardware Requirements

| | Minimum | Recommended |
|------|---------|-------------|
| **GPU VRAM** | 24GB (RTX 3090/4090) | 48GB (A6000 Ada) |
| **RAM** | 32GB | 64GB |
| **Disk** | 80GB | 120GB+ |

## Git Convention

```
feat: 새 기능 추가
fix: 버그 수정
docs: 문서 수정
exp: 실험 관련 변경
data: 데이터 관련 변경
```

## Team Checklist

### Before Experiment
- [ ] `pip install -r requirements.txt`
- [ ] PMC-VQA test_clean 다운로드 (`bash data/download_data.sh`)
- [ ] 담당 모델 가중치 다운로드
- [ ] `python scripts/verify_models.py --model {your_model}` 통과
- [ ] `configs/experiment_config.yaml`에 LPF sigma 값 확인

### Run Experiment
- [ ] `python scripts/run_experiment.py --model {your_model} --gpu 0`
- [ ] 5개 조건 모두 완료
- [ ] 파싱 실패율 < 10%

### After Experiment
- [ ] 결과 CSV를 `results/`에 공유
- [ ] `python scripts/analyze_results.py --results results/*_all_*.csv`

## References

- [PMC-VQA: Visual Instruction Tuning for Medical VQA](https://arxiv.org/abs/2305.10415)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine](https://arxiv.org/abs/2306.00890)
- [HuatuoGPT-Vision: Injecting Medical Visual Knowledge into Multimodal LLMs](https://arxiv.org/abs/2406.19280)
