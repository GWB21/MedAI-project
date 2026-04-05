#!/usr/bin/env python3
"""
End-to-End 테스트: 실제 PMC-VQA 이미지 + 5개 perturbation → 추론 → 파싱 → 결과.
llava 모듈 충돌 방지를 위해 각 모델을 별도 subprocess로 실행.

Usage:
    python scripts/test_e2e.py --model llava_v15
    python scripts/test_e2e.py --model huatuogpt
    python scripts/test_e2e.py --model medvint
    python scripts/test_e2e.py --model all
"""

import argparse
import subprocess
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 각 모델의 테스트 코드 (subprocess로 실행)
TEST_LLAVA_MED = '''
import sys, os, torch, numpy as np, time
sys.path.insert(0, "{project_root}")

from src.dataset import PMCVQADataset
from src.perturbations import apply_perturbation
from src.parse_answer import parse_answer

print("=== LLaVA-v1.5-7B E2E Test ===")
dataset = PMCVQADataset("{project_root}/data/pmc_vqa")
image = dataset.load_image(0)
prompt = dataset.get_prompt(0)
item = dataset[0]
print(f"Image: {{item['image_id']}}, GT: {{item['gt_answer']}}")
print(f"Q: {{item['question'][:80]}}...")

# Load
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from PIL import Image

t0 = time.time()
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path='liuhaotian/llava-v1.5-7b', model_base=None,
    model_name='llava-v1.5-7b', device_map='auto')
model.eval()
print(f"Model loaded: {{time.time()-t0:.1f}}s")

def run_inference(img_np, prompt_text):
    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\\n' + prompt_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    pil_img = Image.fromarray(img_np)
    img_t = process_images([pil_img], image_processor, model.config).to(model.device, dtype=torch.float16)
    ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        out = model.generate(ids, images=img_t, attention_mask=attn, do_sample=False, max_new_tokens=32, num_beams=1)
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        input_text = full_prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw = full_text[full_text.index(input_text)+len(input_text):].strip() if input_text in full_text else full_text.strip()
        outputs = model(input_ids=ids, images=img_t, attention_mask=attn, return_dict=True)
        logits_last = outputs.logits[:, -1, :]
    choice_logits = {{}}
    for c in 'ABCD':
        tid = tokenizer.encode(c, add_special_tokens=False)[0]
        choice_logits[c] = logits_last[0, tid].item()
    return raw, parse_answer(raw), choice_logits

conditions = [
    ("original", {{}}),
    ("black", {{}}),
    ("lpf", {{"sigma": 3}}),
    ("hpf", {{"sigma": 25}}),
    ("patch_shuffle", {{"patch_size": 16, "seed": 42}}),
]
print()
for cond_name, kwargs in conditions:
    perturbed = apply_perturbation(image, cond_name, **kwargs)
    raw, parsed, logits = run_inference(perturbed, prompt)
    status = "OK" if parsed else "FAIL"
    logit_str = ' '.join(f'{{k}}={{vv:.2f}}' for k,vv in logits.items())
    print(f"  {{cond_name:15s}} | pred={{parsed or 'NONE':5s}} | gt={{item['gt_answer']}} | correct={{parsed==item['gt_answer']}} | raw={{repr(raw[:50])}} | {{logit_str}}")

print("\\n>>> LLaVA-v1.5 E2E: PASS <<<")
'''

TEST_HUATUOGPT = '''
import sys, os, torch, numpy as np, time
project_root = "{project_root}"
sys.path.insert(0, os.path.join(project_root, "repos", "HuatuoGPT-Vision"))
sys.path.insert(0, project_root)

from src.dataset import PMCVQADataset
from src.perturbations import apply_perturbation
from src.parse_answer import parse_answer

print("=== HuatuoGPT-Vision-7B E2E Test ===")
dataset = PMCVQADataset(os.path.join(project_root, "data/pmc_vqa"))
image = dataset.load_image(0)
prompt = dataset.get_prompt(0)
item = dataset[0]
print(f"Image: {{item['image_id']}}, GT: {{item['gt_answer']}}")

# Load
from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from llava.constants import IMAGE_TOKEN_INDEX
from huggingface_hub import snapshot_download
from PIL import Image

t0 = time.time()
model_dir = snapshot_download(repo_id='FreedomIntelligence/HuatuoGPT-Vision-7B')
model = LlavaQwen2ForCausalLM.from_pretrained(model_dir, init_vision_encoder_from_ckpt=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token_id = tokenizer.eos_token_id

vt = model.get_vision_tower()
vit_path = os.path.join(model_dir, 'vit', 'clip_vit_large_patch14_336')
vt.vision_tower = CLIPVisionModel.from_pretrained(vit_path)
vt.image_processor = CLIPImageProcessor.from_pretrained(vit_path)
vt.is_loaded = True
vt.vision_tower.requires_grad_(False)
vt.to(dtype=torch.bfloat16, device='cuda')
ip = vt.image_processor
model.eval().to('cuda')
print(f"Model loaded: {{time.time()-t0:.1f}}s")

def expand2square(img, bg):
    w, h = img.size
    if w == h: return img
    sz = max(w, h)
    r = Image.new(img.mode, (sz, sz), bg)
    r.paste(img, ((sz-w)//2, (sz-h)//2))
    return r

def tokenize_with_image(text):
    chunks = [tokenizer(c, add_special_tokens=False).input_ids for c in text.split('<image>')]
    ids = []
    off = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        off = 1; ids.append(chunks[0][0])
    sep_list = [e for sub in zip(chunks, [[IMAGE_TOKEN_INDEX]*(off+1)]*len(chunks)) for e in sub][:-1]
    for x in sep_list: ids.extend(x[off:])
    return torch.tensor(ids, dtype=torch.long)

def run_inference(img_np, prompt_text):
    pil = Image.fromarray(img_np)
    sq = expand2square(pil, tuple(int(x*255) for x in ip.image_mean))
    it = ip.preprocess(sq, return_tensors='pt')['pixel_values'][0].unsqueeze(0).to('cuda', dtype=torch.bfloat16)
    fp = f'<image>\\n<|user|>\\n{{prompt_text}}\\n<|assistant|>\\n'
    ids = tokenize_with_image(fp).unsqueeze(0).to('cuda')
    with torch.no_grad():
        out = model.generate(ids, images=it, do_sample=False, max_new_tokens=32, num_beams=1, use_cache=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        input_text = fp.replace('<image>', '').strip()
        raw = full_text[full_text.index(input_text)+len(input_text):].strip() if input_text in full_text else full_text.strip()
        outputs = model(input_ids=ids, images=it, return_dict=True)
        ll = outputs.logits[:, -1, :]
    cl = {{}}
    for c in 'ABCD':
        tid = tokenizer.encode(c, add_special_tokens=False)[0]
        cl[c] = ll[0, tid].item()
    return raw, parse_answer(raw), cl

conditions = [("original",{{}}),("black",{{}}),("lpf",{{"sigma":3}}),("hpf",{{"sigma":25}}),("patch_shuffle",{{"patch_size":16,"seed":42}})]
print()
for cn, kw in conditions:
    p = apply_perturbation(image, cn, **kw)
    raw, parsed, logits = run_inference(p, prompt)
    logit_str = ' '.join(f'{{k}}={{vv:.2f}}' for k,vv in logits.items())
    print(f"  {{cn:15s}} | pred={{parsed or 'NONE':5s}} | gt={{item['gt_answer']}} | correct={{parsed==item['gt_answer']}} | raw={{repr(raw[:50])}} | {{logit_str}}")

print("\\n>>> HuatuoGPT-Vision E2E: PASS <<<")
'''

TEST_MEDVINT = '''
import sys, os, torch, numpy as np, time
project_root = "{project_root}"
medvint_src = os.path.join(project_root, "repos", "PMC-VQA", "src", "MedVInT_TD")
sys.path.insert(0, medvint_src)
sys.path.insert(0, project_root)

from src.dataset import PMCVQADataset
from src.perturbations import apply_perturbation
from src.parse_answer import parse_answer

# PyTorch 2.6+: torch.load defaults to weights_only=True, but PMC-CLIP checkpoint needs False
import functools
_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)

print("=== MedVInT-TD E2E Test ===")
dataset = PMCVQADataset(os.path.join(project_root, "data/pmc_vqa"))
image = dataset.load_image(0)
prompt = dataset.get_prompt(0)
item = dataset[0]
print(f"Image: {{item['image_id']}}, GT: {{item['gt_answer']}}")

from huggingface_hub import snapshot_download
from transformers import LlamaTokenizerFast as LlamaTokenizer
from PIL import Image
from torchvision import transforms

t0 = time.time()
llama_path = snapshot_download(repo_id='chaoyi-wu/PMC_LLAMA_7B')
ckpt_path = os.path.join(project_root, "checkpoints/MedVInT-TD/VQA_lora_PMC_LLaMA_PMCCLIP/choice/checkpoint-4000/pytorch_model.bin")

from dataclasses import dataclass
@dataclass
class Args:
    model_path: str = llama_path
    checkpointing: bool = False
    N: int = 12; H: int = 8; img_token_num: int = 32
    voc_size: int = 32000; hidden_dim: int = 4096
    Vision_module: str = "PMC-CLIP"
    visual_model_path: str = os.path.join(project_root, "checkpoints/PMC-CLIP/checkpoint.pt")
    is_lora: bool = True; peft_mode: str = "lora"; lora_rank: int = 8

from models.QA_model import QA_model
model = QA_model(Args())

ckpt = torch.load(ckpt_path, map_location='cpu')
fixed = {{}}
for k, v in ckpt.items():
    if 'self_attn.q_proj.weight' in k and 'vision_model' not in k:
        k = k.replace('self_attn.q_proj.weight','self_attn.q_proj.base_layer.weight')
    if 'self_attn.v_proj.weight' in k and 'vision_model' not in k:
        k = k.replace('self_attn.v_proj.weight','self_attn.v_proj.base_layer.weight')
    if 'lora_A' in k and 'lora_A.default' not in k:
        k = k.replace('lora_A','lora_A.default')
    if 'lora_B' in k and 'lora_B.default' not in k:
        k = k.replace('lora_B','lora_B.default')
    fixed[k] = v
model.load_state_dict(fixed, strict=False)
model.to('cuda').eval()
# Fix PMC-LLaMA's broken tokenizer_config (empty special tokens)
import json, shutil, tempfile
_tmp_tok = tempfile.mkdtemp()
for f in os.listdir(llama_path):
    if 'tokenizer' in f or 'special' in f:
        shutil.copy2(os.path.join(llama_path, f), _tmp_tok)
cfg_path = os.path.join(_tmp_tok, 'tokenizer_config.json')
with open(cfg_path) as f:
    tcfg = json.load(f)
tcfg['bos_token'] = '<s>'
tcfg['eos_token'] = '</s>'
tcfg['unk_token'] = '<unk>'
with open(cfg_path, 'w') as f:
    json.dump(tcfg, f)
tokenizer = LlamaTokenizer.from_pretrained(_tmp_tok)
print(f"Model loaded: {{time.time()-t0:.1f}}s")

tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711])
])

def run_inference(img_np, prompt_text):
    pil = Image.fromarray(img_np)
    it = tf(pil.convert('RGB')).unsqueeze(0).to('cuda')
    ids = tokenizer(prompt_text, return_tensors='pt')['input_ids'].to('cuda')
    with torch.no_grad():
        gen = model.generate(ids, it)
        pred_ids = gen.argmax(-1)
        raw = tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
        out = model(ids, it)
        ll = out.logits[:, -1, :]
    cl = {{}}
    for c in 'ABCD':
        tid = tokenizer.encode(c, add_special_tokens=False)[-1]
        cl[c] = ll[0, tid].item()
    parsed = parse_answer(raw)
    if not parsed and raw:
        lc = raw[-1].upper()
        if lc in 'ABCD': parsed = lc
    return raw, parsed, cl

conditions = [("original",{{}}),("black",{{}}),("lpf",{{"sigma":3}}),("hpf",{{"sigma":25}}),("patch_shuffle",{{"patch_size":16,"seed":42}})]
print()
for cn, kw in conditions:
    p = apply_perturbation(image, cn, **kw)
    raw, parsed, logits = run_inference(p, prompt)
    logit_str = ' '.join(f'{{k}}={{vv:.2f}}' for k,vv in logits.items())
    print(f"  {{cn:15s}} | pred={{parsed or 'NONE':5s}} | gt={{item['gt_answer']}} | correct={{parsed==item['gt_answer']}} | raw={{repr(raw[:50])}} | {{logit_str}}")

print("\\n>>> MedVInT-TD E2E: PASS <<<")
'''

TESTS = {
    "llava_v15": TEST_LLAVA_MED,
    "huatuogpt": TEST_HUATUOGPT,
    "medvint": TEST_MEDVINT,
}


def run_test(model_name: str):
    code = TESTS[model_name].format(project_root=PROJECT_ROOT)
    print(f"\n{'='*60}")
    print(f"  Testing: {model_name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        timeout=600,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["llava_v15", "huatuogpt", "medvint", "all"])
    args = parser.parse_args()

    models = ["llava_v15", "huatuogpt", "medvint"] if args.model == "all" else [args.model]

    results = {}
    for m in models:
        results[m] = run_test(m)

    print(f"\n{'='*60}")
    print("  E2E TEST SUMMARY")
    print(f"{'='*60}")
    for m, ok in results.items():
        print(f"  {m:15s}  [{'PASS' if ok else 'FAIL'}]")


if __name__ == "__main__":
    main()
