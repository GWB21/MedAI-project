# Data Directory

## PMC-VQA Dataset

After running `bash data/download_data.sh`, this directory will contain:

```
data/
└── pmc_vqa/
    ├── test_clean.csv       # 2,000 manually verified VQA samples
    └── images/              # Medical images (JPG)
        ├── PMC1064097_F1.jpg
        ├── PMC1079854_F10.jpg
        └── ...
```

## CSV Schema (`test_clean.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `Figure_path` | str | Relative image path |
| `Question` | str | VQA question (contains modality/anatomy info) |
| `Answer` | str | Ground-truth answer text |
| `Choice A` | str | Option A |
| `Choice B` | str | Option B |
| `Choice C` | str | Option C |
| `Choice D` | str | Option D |
| `Answer_label` | str | Correct label: A, B, C, or D |

## Source

- HuggingFace: [xmcmic/PMC-VQA](https://huggingface.co/datasets/xmcmic/PMC-VQA)
- Paper: [PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering](https://arxiv.org/abs/2305.10415)
