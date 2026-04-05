# Medical VQA Image Perturbation Experiment

## 프로젝트 개요

의료 VQA(Visual Question Answering) 모델이 실제로 이미지를 분석하는지, 아니면 텍스트 prior(질문 내 모달리티/부위 정보, 선택지 통계)에만 의존하는지를 **이미지 perturbation 실험**으로 검증합니다.

핵심 아이디어: 이미지를 다양한 방식으로 훼손(검정, 블러, 엣지만 남기기, 패치 셔플)한 뒤에도 정답률이 크게 떨어지지 않는다면, 모델이 이미지를 제대로 활용하지 않는다는 증거입니다.

## 실험 설계

### 기본 설정

| 항목 | 설정 |
|------|------|
| **데이터셋** | [PMC-VQA](https://huggingface.co/datasets/xmcmic/PMC-VQA) test_clean (2,000 samples) |
| **모델** | LLaVA-Med-1.5, HuatuoGPT-Vision-7B, MedVInT-TD |
| **이미지 조건** | Original, Black, LPF, HPF, Patch Shuffle (16x16) |
| **디코딩** | Greedy (temperature=0, do_sample=False) |

### 이미지 조건 (Image Conditions)

| Condition | 설명 |
|-----------|------|
| **Original** | 원본 이미지 그대로 |
| **Black** | 모든 픽셀 0 (검정 이미지) -- 이미지 정보 완전 제거 |
| **LPF** | Gaussian Low-Pass Filter (sigma calibrated to SSIM ~ 0.7) -- 고주파 디테일 제거 |
| **HPF** | High-Pass Filter, Original - LPF + 128 (sigma calibrated to SSIM ~ 0.7) -- 저주파 구조 제거, 엣지만 보존 |
| **Patch Shuffle** | 16x16 패치 위치를 무작위 셔플 -- 전역 구조 파괴, 로컬 텍스처 보존 |

### 진단 지표 (Diagnostic Metrics)

| Metric | Formula | 해석 |
|--------|---------|------|
| **VRS** (Vision Reliance Score) | Acc(Original) - Acc(Black) | 양수: 이미지 의존 / 0: 이미지 무시 |
| **IS** (Image Sensitivity) | 1 - Acc(PatchShuffle) / Acc(Original) | 높음: 이미지 변화에 민감 |
| **NSR** (Noise Sensitivity Ratio) | 1 - Acc(LPF) / Acc(Original) | 높음: 고주파 정보(디테일)에 의존 |
| **HPF_Drop** | Acc(Original) - Acc(HPF) | 큼: 저주파 정보(전체 구조)가 중요 |

---

## 팀원별 셋업 가이드

### 공통 사전 준비

- **GPU**: 24GB+ VRAM (RTX 3090 / RTX 4090 / A6000 Ada)
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

#### LLaVA-Med 담당자

```bash
pip install -r requirements_llava_med.txt
pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps
python scripts/setup.py --model llava_med --hf_token YOUR_TOKEN
```

> **주의**: LLaVA-Med는 transformers==4.37.2가 필요합니다. 다른 모델과 같은 환경에서 사용할 수 없으므로 별도 venv를 권장합니다.

#### HuatuoGPT-Vision 담당자

```bash
pip install -r requirements_huatuogpt.txt
python scripts/setup.py --model huatuogpt --hf_token YOUR_TOKEN
```

> **참고**: setup.py가 자동으로 레포 클론 + transformers 4.44 호환성 패치를 적용합니다.

#### MedVInT-TD 담당자

```bash
pip install -r requirements_medvint.txt
python scripts/setup.py --model medvint --hf_token YOUR_TOKEN
```

> **참고**: setup.py가 PMC-VQA 레포 클론, MedVInT-TD 체크포인트, PMC-LLaMA, PMC-CLIP을 모두 다운로드합니다.

### 3단계: 검증

```bash
python scripts/test_e2e.py --model {모델명}
```

셋업이 정상적으로 완료되었는지 end-to-end 테스트를 실행합니다.

---

## 실험 실행

```bash
python scripts/run_experiment.py --model {모델명} --gpu 0
```

5개 이미지 조건(original, black, lpf, hpf, patch_shuffle)을 자동으로 순회하며, 각 조건별 CSV와 전체 통합 CSV를 `results/`에 저장합니다.

### 커맨드 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | (필수) | `llava_med` / `huatuogpt` / `medvint` |
| `--gpu` | `0` | GPU 번호 |
| `--batch_size` | `1` | 배치 크기 (VRAM에 따라 조정) |
| `--num_workers` | `8` | CPU 데이터 로더 수 (코어 수에 맞게) |
| `--conditions` | 전체 | 특정 조건만 실행: `--conditions original black` |
| `--config` | `configs/experiment_config.yaml` | 설정 파일 경로 |

### VRAM별 권장 설정

| GPU | VRAM | batch_size | num_workers |
|-----|------|-----------|-------------|
| RTX 3090 | 24GB | 1 | 4 |
| RTX 4090 | 24GB | 1 | 8 |
| A5000 | 24GB | 1 | 8 |
| A6000 Ada | 48GB | 2 | 8 |

### 실행 예시

```bash
# 전체 실험 (5개 조건)
python scripts/run_experiment.py --model llava_med --gpu 0

# 특정 조건만
python scripts/run_experiment.py --model huatuogpt --gpu 0 --conditions original black

# 배치/워커 조정
python scripts/run_experiment.py --model medvint --gpu 0 --batch_size 2 --num_workers 4
```

### 예상 소요시간

텍스트 생성(generate) 없이 **forward 1회로 logit만 추출**하므로 매우 빠릅니다.

| GPU | 조건당 (~2,000 samples) | 전체 (5개 조건) |
|-----|----------------------|----------------|
| RTX 3090/4090 | ~2분 | **~10분** |
| A6000 Ada | ~1.5분 | **~8분** |

---

## 답변 결정 방식

모든 모델에서 **logit argmax** 방식을 사용합니다. 텍스트 생성(generate)은 하지 않습니다.

```
입력: 이미지 + 프롬프트 → model.forward() 1회 → 마지막 위치 logit에서 A/B/C/D 비교 → argmax = 답변
```

- `model.generate()` (autoregressive, ~32 forward passes) 대신 `model.forward()` **1회**만 실행
- VRAM 사용량 적음 (KV cache 불필요), 추론 속도 ~30배 빠름
- 텍스트 파싱 실패(PARSE_FAIL)가 원천적으로 발생하지 않음
- 모델의 실제 확률 분포를 직접 반영하여 모든 모델에서 일관된 비교 가능

---

## 결과 CSV 형식

| Column | 설명 |
|--------|------|
| `image_id` | 이미지 식별자 |
| `condition` | original / black / lpf / hpf / patch_shuffle |
| `model` | llava_med / huatuogpt / medvint |
| `question` | PMC-VQA 원본 질문 |
| `choice_A`~`choice_D` | 선택지 |
| `gt_answer` | 정답 (A/B/C/D) |
| `pred_answer` | 모델 예측 (A/B/C/D, logit argmax 기반) |
| `correct` | 정답 여부 (1/0) |
| `parse_success` | 텍스트 파싱 성공 여부 (분석용) |
| `raw_output` | 모델 생성 텍스트 원본 |
| `logit_A`~`logit_D` | 선택지별 logit 값 |

---

## 분석

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

## 프로젝트 구조

```
MedAI-project/
├── configs/
│   └── experiment_config.yaml      # 실험 파라미터 (모든 팀원 동일하게 사용)
├── data/
│   └── pmc_vqa/                    # PMC-VQA 데이터셋 (setup.py로 다운로드)
├── src/
│   ├── __init__.py
│   ├── perturbations.py            # 이미지 조작 함수 (Black, LPF, HPF, Patch Shuffle)
│   ├── dataset.py                  # PMC-VQA 데이터 로더
│   ├── parse_answer.py             # 모델 출력에서 A/B/C/D 파싱
│   ├── metrics.py                  # VRS, IS, NSR, HPF_Drop 계산
│   ├── inference.py                # 메인 추론 루프 (logit argmax 기반)
│   └── models/
│       ├── __init__.py
│       ├── base_model.py           # 공통 인터페이스 (ABC)
│       ├── llava_med.py            # LLaVA-Med-1.5 wrapper
│       ├── huatuogpt.py            # HuatuoGPT-Vision-7B wrapper
│       └── medvint.py              # MedVInT-TD wrapper
├── scripts/
│   ├── run_experiment.py           # 실험 실행 엔트리포인트
│   ├── setup.py                    # 원클릭 셋업 (데이터 + 모델 다운로드)
│   ├── calibrate_lpf.py            # LPF/HPF sigma calibration
│   ├── analyze_results.py          # 결과 분석 + 테이블/차트 생성
│   ├── verify_models.py            # 모델 로드/추론 검증
│   └── test_e2e.py                 # End-to-end 테스트
├── results/                        # 결과 CSV 저장 디렉토리
├── requirements_base.txt           # 공통 패키지
├── requirements_llava_med.txt      # LLaVA-Med 전용
├── requirements_huatuogpt.txt      # HuatuoGPT 전용
├── requirements_medvint.txt        # MedVInT 전용
└── README.md
```

---

## 모델별 참고 정보

| 모델 | HF Repo | GitHub | transformers | 논문 |
|------|---------|--------|-------------|------|
| **LLaVA-Med-1.5** | `microsoft/llava-med-v1.5-mistral-7b` | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | 4.37.2 | [NeurIPS 2023](https://arxiv.org/abs/2306.00890) |
| **HuatuoGPT-Vision-7B** | `FreedomIntelligence/HuatuoGPT-Vision-7B` | [FreedomIntelligence/HuatuoGPT-Vision](https://github.com/FreedomIntelligence/HuatuoGPT-Vision) | 4.44.0 | [EMNLP 2024](https://arxiv.org/abs/2406.19280) |
| **MedVInT-TD** | `xmcmic/MedVInT-TD` | [xiaoman-zhang/PMC-VQA](https://github.com/xiaoman-zhang/PMC-VQA) | 4.44.0 | [arXiv 2305.10415](https://arxiv.org/abs/2305.10415) |

> **중요**: LLaVA-Med는 transformers 4.37이 필요하고, HuatuoGPT/MedVInT는 4.44가 필요합니다. 같은 환경에서 모든 모델을 실행할 수 없으므로, 각 팀원이 자기 담당 모델의 requirements만 설치하세요.

---

## Calibration 결과

LPF와 HPF는 각각 독립적으로 SSIM ~ 0.70이 되도록 sigma를 calibration합니다.

| 필터 | Calibrated Sigma | Target SSIM |
|------|-----------------|-------------|
| **LPF** | sigma=3 | SSIM ~ 0.70 |
| **HPF** | sigma=25 | SSIM ~ 0.70 |

calibration 스크립트:
```bash
python scripts/calibrate_lpf.py --data_dir ./data/pmc_vqa --target_ssim 0.7
```

결과 sigma 값은 `configs/experiment_config.yaml`에 자동 반영됩니다.

---

## Git Convention

커밋 메시지 접두사:

| 접두사 | 용도 |
|--------|------|
| `feat:` | 새 기능 추가 |
| `fix:` | 버그 수정 |
| `docs:` | 문서 수정 |
| `exp:` | 실험 관련 변경 |
| `data:` | 데이터 관련 변경 |

---

## References

- [PMC-VQA: Visual Instruction Tuning for Medical VQA](https://arxiv.org/abs/2305.10415)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine](https://arxiv.org/abs/2306.00890)
- [HuatuoGPT-Vision: Injecting Medical Visual Knowledge into Multimodal LLMs](https://arxiv.org/abs/2406.19280)
