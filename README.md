# Medical VQA 이미지 Perturbation 실험

## 1. 프로젝트 개요

의료 VQA(Visual Question Answering) 모델이 실제로 이미지를 분석하는지, 아니면 텍스트 prior(질문 내 모달리티/부위 정보, 선택지 통계)에만 의존하는지를 **이미지 perturbation 실험**으로 검증합니다.

**핵심 아이디어**: 이미지를 다양한 방식으로 훼손(검정, 블러, 엣지만 남기기, 패치 셔플)한 뒤에도 정답률이 크게 떨어지지 않는다면, 모델이 이미지를 제대로 활용하지 않는다는 증거입니다.

### 사용 모델 (3종)

| 모델 | HF Repo | transformers 버전 | 추가 설치 |
|------|---------|-------------------|-----------|
| **LLaVA-v1.5-7B** | `liuhaotian/llava-v1.5-7b` | 4.37.2 | `pip install llava --no-deps` |
| **HuatuoGPT-Vision-7B** | `FreedomIntelligence/HuatuoGPT-Vision-7B` | 4.40.0 | 레포 클론 (setup.py 자동) |
| **MedVInT-TD** | `xmcmic/MedVInT-TD` | 4.40.0 | 레포 클론 + PMC-CLIP (setup.py 자동) |

### 답변 결정 방식

모든 모델에서 `model.generate()`로 텍스트를 생성한 뒤, 출력 텍스트를 **파싱(parsing)**하여 A/B/C/D 답변을 결정합니다. VLM 평가 논문의 표준 방식을 따릅니다.

### 프롬프트 형식

- **LLaVA-v1.5** / **HuatuoGPT**: `dataset.get_prompt()`에서 제공하는 동일한 프롬프트 형식 사용
- **MedVInT**: 자체 프롬프트 형식 사용 (`build_prompt`에 raw choices 전달)

### Baseline 결과

| 모델 | Original Accuracy |
|------|-------------------|
| LLaVA-v1.5-7B | 33.8% |
| HuatuoGPT-Vision-7B | 43.2% |
| MedVInT-TD | 42.7% |

---

## 2. 실험 설계

### 기본 설정

| 항목 | 설정 |
|------|------|
| **데이터셋** | [PMC-VQA](https://huggingface.co/datasets/xmcmic/PMC-VQA) test_clean (2,000 samples) |
| **모델** | LLaVA-v1.5-7B, HuatuoGPT-Vision-7B, MedVInT-TD |
| **이미지 조건** | Original, Black, LPF, HPF, Patch Shuffle (16x16) |
| **디코딩** | Greedy (temperature=0, do_sample=False) |
| **batch_size** | 1 (이미지 해상도가 제각각이므로 VLM 평가 논문 표준에 따라 batch=1 사용) |

### 이미지 조건 (Image Conditions)

| Condition | 설명 |
|-----------|------|
| **Original** | 원본 이미지 그대로 |
| **Black** | 모든 픽셀 0 (검정 이미지) -- 이미지 정보 완전 제거 |
| **LPF** | Gaussian Low-Pass Filter (sigma=3, calibrated) -- 고주파 디테일 제거 |
| **HPF** | High-Pass Filter, Original - LPF + 128 (sigma=25, calibrated) -- 저주파 구조 제거, 엣지만 보존 |
| **Patch Shuffle** | 16x16 패치 위치를 무작위 셔플 -- 전역 구조 파괴, 로컬 텍스처 보존 |

### 진단 지표 (Diagnostic Metrics)

| Metric | Formula | 해석 |
|--------|---------|------|
| **VRS** (Vision Reliance Score) | Acc(Original) - Acc(Black) | 양수: 이미지 의존 / 0: 이미지 무시 |
| **IS** (Image Sensitivity) | 1 - Acc(PatchShuffle) / Acc(Original) | 높음: 이미지 변화에 민감 |
| **NSR** (Noise Sensitivity Ratio) | 1 - Acc(LPF) / Acc(Original) | 높음: 고주파 정보(디테일)에 의존 |
| **HPF_Drop** | Acc(Original) - Acc(HPF) | 큼: 저주파 정보(전체 구조)가 중요 |

---

## 3. 팀원별 셋업 가이드

### 공통 사전 준비

- **GPU**: 24GB VRAM 이상 (RTX 3090 / RTX 4090 / A6000 Ada)
  - LLaVA-v1.5: ~14.8GB VRAM
  - HuatuoGPT: ~21GB VRAM
  - MedVInT: ~27.5GB VRAM (24GB GPU에서도 동작)
- **CUDA**: 12.4+
- **Python**: 3.10+
- **HuggingFace 토큰**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 에서 발급

### 1단계: 레포 클론 & 기본 패키지

```bash
git clone https://github.com/GWB21/MedAI-project.git
cd MedAI-project
pip install -r requirements_base.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2단계: 모델별 셋업 (본인 담당 모델만)

#### LLaVA-v1.5 담당자

```bash
# transformers 버전 설치
pip install -r requirements_llava_v15.txt

# llava 라이브러리 설치 (반드시 --no-deps 옵션 사용)
pip install llava --no-deps

# 데이터셋 + 모델 다운로드
python scripts/setup.py --model llava_v15 --hf_token YOUR_TOKEN
```

> **주의**: LLaVA-v1.5는 `transformers==4.37.2`가 필요합니다. 다른 모델과 같은 환경에서 사용할 수 없으므로 별도 venv를 권장합니다.

#### HuatuoGPT-Vision 담당자

```bash
# transformers 버전 설치
pip install -r requirements_huatuogpt.txt

# 데이터셋 + 모델 다운로드 + 레포 클론 + forward kwargs 패치
python scripts/setup.py --model huatuogpt --hf_token YOUR_TOKEN
```

> **참고**: `setup.py`가 자동으로 `repos/HuatuoGPT-Vision`에 레포를 클론하고, transformers 호환성 패치(`forward()` kwargs)를 적용합니다.

#### MedVInT-TD 담당자

```bash
# transformers 버전 설치
pip install -r requirements_medvint.txt

# 데이터셋 + 모델 + PMC-VQA 레포 클론 + PMC-CLIP 다운로드
python scripts/setup.py --model medvint --hf_token YOUR_TOKEN
```

> **참고**: `setup.py`가 PMC-VQA 레포 클론, MedVInT-TD 체크포인트, PMC-LLaMA, PMC-CLIP을 모두 다운로드합니다.

### 3단계: 실험 실행

```bash
python scripts/run_experiment.py --model {모델명} --gpu 0
```

> **주의**: `repos/` 디렉토리에 클론된 외부 레포(HuatuoGPT-Vision, PMC-VQA)는 **절대 수정하지 마세요**. setup.py의 자동 패치 외에 직접 코드를 변경하면 재현성이 깨집니다.

---

## 4. 실험 실행

```bash
python scripts/run_experiment.py --model {모델명} --gpu 0
```

5개 이미지 조건(original, black, lpf, hpf, patch_shuffle)을 자동으로 순회하며, 각 조건별 CSV와 전체 통합 CSV를 `results/`에 저장합니다.

### 커맨드 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | (필수) | `llava_v15` / `huatuogpt` / `medvint` |
| `--gpu` | `0` | GPU 번호 |
| `--batch_size` | `1` | 배치 크기 (이미지 해상도 차이로 인해 1 권장) |
| `--num_workers` | `8` | CPU 데이터 로더 수 (8 권장) |
| `--conditions` | 전체 | 특정 조건만 실행: `--conditions original black` |
| `--config` | `configs/experiment_config.yaml` | 설정 파일 경로 |

### VRAM 정보

모든 모델은 24GB VRAM GPU에서 동작합니다.

| 모델 | VRAM 사용량 |
|------|-------------|
| LLaVA-v1.5-7B | ~14.8GB |
| HuatuoGPT-Vision-7B | ~21GB |
| MedVInT-TD | ~27.5GB |

> **참고**: `batch_size=1`이 표준입니다. 의료 이미지는 해상도가 제각각이므로 배치로 묶을 수 없으며, 모든 VLM 평가 논문에서도 batch=1을 사용합니다.

### 실행 예시

```bash
# 전체 실험 (5개 조건)
python scripts/run_experiment.py --model llava_v15 --gpu 0

# 특정 조건만
python scripts/run_experiment.py --model huatuogpt --gpu 0 --conditions original black

# 워커 수 조정
python scripts/run_experiment.py --model medvint --gpu 0 --num_workers 4
```

---

## 5. 결과 CSV 형식

| Column | 설명 |
|--------|------|
| `image_id` | 이미지 식별자 |
| `condition` | original / black / lpf / hpf / patch_shuffle |
| `model` | llava_v15 / huatuogpt / medvint |
| `question` | PMC-VQA 원본 질문 |
| `choice_A`~`choice_D` | 선택지 |
| `gt_answer` | 정답 (A/B/C/D) |
| `pred_answer` | 모델 예측 (A/B/C/D, generate 텍스트 파싱 기반) |
| `correct` | 정답 여부 (1/0) |
| `parse_success` | 텍스트 파싱 성공 여부 (분석용) |
| `raw_output` | 모델 생성 텍스트 원본 |

---

## 6. 분석

```bash
# 3명의 결과 CSV를 모아서 분석
python scripts/analyze_results.py --results results/*_all_*.csv
```

분석 스크립트가 생성하는 항목:
- 모델별/조건별 정답률 테이블
- VRS, IS, NSR, HPF_Drop 진단 지표
- 시각화 차트

### 진단 지표 해석 가이드

| 시나리오 | VRS | IS | 해석 |
|----------|-----|-----|------|
| 이미지 활용 잘 함 | 높음 (>0.15) | 높음 (>0.2) | 이미지에 의존하고 변화에 민감 |
| 텍스트만 의존 | 낮음 (~0) | 낮음 (~0) | 이미지 무시, 질문/선택지만으로 답변 |
| 이미지 보지만 둔감 | 중간 | 낮음 | 이미지의 대략적 특성만 활용 |

---

## 7. 프로젝트 구조

```
MedAI-project/
├── configs/
│   └── experiment_config.yaml      # 실험 파라미터 (모든 팀원 동일하게 사용)
├── data/
│   └── pmc_vqa/                    # PMC-VQA 데이터셋 (setup.py로 다운로드)
├── repos/                          # 외부 레포 클론 (수정 금지!)
│   ├── HuatuoGPT-Vision/           # HuatuoGPT 모델 코드
│   └── PMC-VQA/                    # MedVInT 모델 코드
├── src/
│   ├── __init__.py
│   ├── perturbations.py            # 이미지 조작 함수 (Black, LPF, HPF, Patch Shuffle)
│   ├── dataset.py                  # PMC-VQA 데이터 로더 + get_prompt()
│   ├── parse_answer.py             # 모델 출력에서 A/B/C/D 파싱
│   ├── metrics.py                  # VRS, IS, NSR, HPF_Drop 계산
│   ├── inference.py                # 메인 추론 루프 (generate + 텍스트 파싱)
│   └── models/
│       ├── __init__.py
│       ├── base_model.py           # 공통 인터페이스 (ABC)
│       ├── llava_v15.py            # LLaVA-v1.5-7B wrapper
│       ├── huatuogpt.py            # HuatuoGPT-Vision-7B wrapper
│       └── medvint.py              # MedVInT-TD wrapper
├── scripts/
│   ├── run_experiment.py           # 실험 실행 엔트리포인트
│   ├── setup.py                    # 원클릭 셋업 (데이터 + 모델 다운로드)
│   ├── calibrate_lpf.py            # LPF/HPF sigma calibration
│   ├── analyze_results.py          # 결과 분석 + 테이블/차트 생성
│   └── verify_models.py            # 모델 로드/추론 검증
├── results/                        # 결과 CSV 저장 디렉토리
├── requirements_base.txt           # 공통 패키지 (transformers 버전 미포함)
├── requirements_llava_v15.txt      # LLaVA-v1.5 전용 (transformers==4.37.2)
├── requirements_huatuogpt.txt      # HuatuoGPT 전용 (transformers==4.40.0)
├── requirements_medvint.txt        # MedVInT 전용 (transformers==4.40.0)
└── README.md
```

---

## 8. 모델별 참고 정보

| 모델 | HF Repo | GitHub | transformers | 논문 |
|------|---------|--------|-------------|------|
| **LLaVA-v1.5-7B** | `liuhaotian/llava-v1.5-7b` | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | 4.37.2 | [NeurIPS 2023](https://arxiv.org/abs/2304.08485) |
| **HuatuoGPT-Vision-7B** | `FreedomIntelligence/HuatuoGPT-Vision-7B` | [FreedomIntelligence/HuatuoGPT-Vision](https://github.com/FreedomIntelligence/HuatuoGPT-Vision) | 4.40.0 | [EMNLP 2024](https://arxiv.org/abs/2406.19280) |
| **MedVInT-TD** | `xmcmic/MedVInT-TD` | [xiaoman-zhang/PMC-VQA](https://github.com/xiaoman-zhang/PMC-VQA) | 4.40.0 | [arXiv 2305.10415](https://arxiv.org/abs/2305.10415) |

> **중요**: LLaVA-v1.5는 `transformers==4.37.2`가 필요하고, HuatuoGPT/MedVInT는 `transformers==4.40.0`이 필요합니다. 같은 환경에서 모든 모델을 실행할 수 없으므로, 각 팀원이 자기 담당 모델의 requirements만 설치하세요.

---

## 9. Calibration 결과

LPF와 HPF는 각각 독립적으로 SSIM ~ 0.70이 되도록 sigma를 calibration했습니다.

| 필터 | Calibrated Sigma | Target SSIM |
|------|-----------------|-------------|
| **LPF** | sigma=3 | SSIM ~ 0.70 |
| **HPF** | sigma=25 | SSIM ~ 0.70 |

calibration 스크립트:
```bash
python scripts/calibrate_lpf.py --data_dir ./data/pmc_vqa --target_ssim 0.7
```

결과 sigma 값은 `configs/experiment_config.yaml`에 반영되어 있습니다.

---

## 10. Git Convention

커밋 메시지 접두사:

| 접두사 | 용도 |
|--------|------|
| `feat:` | 새 기능 추가 |
| `fix:` | 버그 수정 |
| `docs:` | 문서 수정 |
| `exp:` | 실험 관련 변경 |
| `data:` | 데이터 관련 변경 |

### 브랜치 전략

- `main`: 안정 버전
- 각자 작업 시 feature 브랜치 사용 권장

### 주의사항

- `repos/` 디렉토리의 클론된 레포는 수정 금지 (`.gitignore`에 포함됨)
- `data/`, `checkpoints/`, `.hf_home/`은 각자 로컬에서 `setup.py`로 다운로드
- 결과 CSV는 `results/`에 저장되며 `.gitignore`에 포함됨 (별도 공유)
