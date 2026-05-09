# 📊 TableVLM

> Efficient multimodal table understanding and reasoning with **DiSCo** and **Table-GLS**.

![Conference](https://img.shields.io/badge/ICML%202026-Spotlight-blue)
![Task](https://img.shields.io/badge/Task-Multimodal%20Table%20Reasoning-green)

Official code for **Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance**.

TableVLM provides a lightweight framework for multimodal table understanding and reasoning. It combines efficient table alignment with structure-aware inference, aiming to improve LVLM performance on complex table images without external tools or heavy reasoning-specific supervision.

This repository includes two complementary components:

- 🧩 **DiSCo**: disentangled structure-content alignment for multimodal table understanding.
- 🔎 **Table-GLS**: global-to-local structure-guided table reasoning without external tools.


## ✨ News

- 🎉 Initial release of DiSCo and Table-GLS inference/evaluation code.
- 🏆 Our paper has been accepted to **ICML 2026** as a **Spotlight paper**.

## 🗂️ Repository Structure

```text
TableVLM_Code/
├── DiSCo/
│   ├── alignment_data/        # DiSCo alignment data indices
│   ├── scripts/               # launch scripts
│   ├── src/                   # MMTab understanding inference code
│   └── test_data/             # MMTab understanding test split
├── TableGLS/
│   ├── scripts/               # launch scripts for DA/CoT and Table-GLS stages
│   ├── src/                   # Table-GLS inference code
│   └── test_data/             # MMTab reasoning test split
└── evaluation/
    ├── MMTab_evaluation.py    # MMTab evaluation script
    └── eval.sh                # example evaluation command
```

## ⚙️ Requirements

Please prepare a Python environment with the following libraries. Exact versions may depend on the base model and CUDA setup you use.

- `python`
- `torch`
- `torchvision`
- `transformers`
- `accelerate`
- `Pillow`
- `tqdm`
- `sacrebleu`
- `vllm`
- `huggingface_hub`
- `LLaMA-Factory`

**Additional notes:**

- `TableGLS` uses `vllm` for Qwen3-VL and Gemma3n inference.
- `DiSCo/src/eval_mmtab_understanding.py` uses Hugging Face `transformers`.
- DiSCo alignment training is performed with `LLaMA-Factory`.
- Please install model-specific dependencies according to the checkpoint you use.

## 📦 Data Preparation

Before training or testing, download [MMTab](https://huggingface.co/datasets/SpursgoZmy/MMTab) images and evaluation files from Huggingface🤗.

You need at least:

- `MMTab-instruct_table_images_82K.zip`
- `MMTab-pre_table_images_part_2_16K.zip`
- `MMTab-eval_table_images_23K.zip`
- `MMTab-eval_test_data_49K.json`
- `MMTab-eval_test_tables_23K.json`

After downloading the image archives, unzip them in `MMTab`. You should obtain two image folders:

```text
MMTab/IID_train_image        # For alignment training
MMTab/table_pretrain_part_2  # For alignment training
MMTab/all_test_image         # For understanding and reasoning inference
```

## 🧩 DiSCo: Disentangled Structure–Content Alignment

### 🏋️ Alignment Training

DiSCo alignment data indices are provided in:

```
DiSCo/alignment_data/disco_alignment_data_5K_images.json
DiSCo/alignment_data/disco_alignment_data_10K_images.json
DiSCo/alignment_data/disco_alignment_data_15K_images.json
DiSCo/alignment_data/disco_alignment_data_20K_images.json
```

These files are already formatted in the ShareGPT style, and can be used with [LlamaFactory](https://github.com/hiyouga/LlamaFactory) for multimodal alignment training according to your chosen data scale and base LVLM. More information please refer to [LlamaFactory](https://github.com/hiyouga/LlamaFactory).

### 🚀 Table Understanding Inference

Run MMTab understanding inference:

```bash
cd DiSCo

MODEL_PATH=/path/to/model \
MODEL_TYPE=qwen3vl \
IMG_PATH=/path/to/MMTab/all_test_image \
bash scripts/run_mmtab_understanding.sh
```

Supported `MODEL_TYPE` values:

```text
qwen3vl
gemma3
gemma3n
llava-v1.6
```

Optional environment variables:

```text
MAX_NEW_TOKENS
EVAL_FILE_PATH
IMG_PATH
ANSWERS_FILE
```

Example with Gemma3n:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/gemma3n-checkpoint \
MODEL_TYPE=gemma3n \
IMG_PATH=/path/to/MMTab/all_test_image \
bash scripts/run_mmtab_understanding.sh
```

For Gemma3n, we recommend exposing a single GPU during inference (e.g., `CUDA_VISIBLE_DEVICES=0`) to avoid device mismatch caused by automatic multi-GPU model dispatch.

## 🔎 Table-GLS: Global-to-Local Structure-Guided Reasoning

### Table-GLS performs global-to-local reasoning in three stages.

```bash
cd TableGLS
```

**Stage 1: global structure exploration.**

```bash
MODEL_PATH=/path/to/model \
MODEL_TYPE=qwen3vl \
IMG_PATH=/path/to/MMTab/all_test_image \
bash scripts/run_tablegls_stage1.sh
```

**Stage 2: self-refined sub-table extraction.**

```bash
MODEL_PATH=/path/to/model \
MODEL_TYPE=qwen3vl \
STAGE1_FILE=/path/to/stage1_file \
IMG_PATH=/path/to/MMTab/all_test_image \
bash scripts/run_tablegls_stage2.sh
```

**Stage 3: evidence-grounded reasoning.**

```bash
MODEL_PATH=/path/to/model \
MODEL_TYPE=qwen3vl \
STAGE2_FILE=/path/to/stage2_file \
IMG_PATH=/path/to/MMTab/all_test_image \
bash scripts/run_tablegls_stage3.sh
```

By default, stage outputs are saved under:

```text
TableGLS/outputs/${MODEL_TYPE}/
```

Optional environment variables:

```text
EVAL_FILE_PATH
IMG_PATH
STAGE1_FILE
STAGE2_FILE
ANSWERS_FILE
TEMPERATURE
```

---
### 💬 Direct Answer / CoT Baselines

The direct-answer and chain-of-thought baseline script is:

```bash
cd TableGLS

MODEL_PATH=/path/to/model \
MODEL_TYPE=qwen3vl \
MODE=cot \
IMG_PATH=/path/to/MMTab/all_test_image \
bash scripts/run_mmtab_da_cot.sh
```

Use `MODE=da` for direct answer prompting.

---
Table-GLS and baselines inference is based on `vllm`. When using Gemma3n with `vllm`, we also recommend exposing a single GPU (e.g., `CUDA_VISIBLE_DEVICES=0`) for more stable inference.


## 📏 Evaluation

The same evaluation script is shared by both understanding and reasoning prediction results.

```bash
cd evaluation

python MMTab_evaluation.py \
  --prediction_file /path/to/predictions.jsonl \
  --eval_data_file /path/to/MMTab-eval_test_data_49K.json \
  --eval_tables_file /path/to/MMTab-eval_test_tables_23K.json
```


## 📝 Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{zhu2026tablevlm,
  title = {Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance},
  author = {Yingjie Zhu, Xuefeng Bai, Kehai Chen, Yang Xiang, Youcheng Pan, Xiaoqiang Zhou and Min Zhang},
  booktitle = {Forty-third International Conference on Machine Learning},
  year = {2026}
}
```

## 🙏 Acknowledgement

This project builds on open-source LVLM ecosystems and the MMTab benchmark. Please also follow the licenses of the base models and datasets you use.
