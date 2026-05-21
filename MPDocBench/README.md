English | [简体中文](./README_zh-CN.md)

# MPDocBench-Parse

Benchmarking Practical Multi-page Document Parsing

[![arXiv](https://img.shields.io/badge/arXiv-2605.03904-b31b1b.svg)](https://arxiv.org/abs/2605.03903) [![GitHub](https://img.shields.io/badge/GitHub-Repository-181717.svg)](https://github.com/Tongyi-Zhiwen/Qwen-Doc) [![ModelScope](https://img.shields.io/badge/ModelScope-Dataset-624AFF.svg)](https://huggingface.co/datasets/Eioss/CC-OCR-V2) [![License](https://img.shields.io/badge/License-Apache_2.0-4CAF50.svg)](LICENSE)

**MPDocBench-Parse** is a comprehensive evaluation toolkit for multi-page document parsing. It provides a unified framework to assess the quality of document parsing systems across multiple dimensions including text extraction, table recognition, formula recognition, document structure, and multi-modal information grounding.

> **Note on the open-sourced dataset.** As described in the original paper, the benchmark contains a total of **433 documents**. After an internal compliance review, **13 documents** involving copyright concern are removed. As a result, the publicly released version of the dataset contains **420 PDF documents**.

<p align="center">
  <img src="assets/figures/overview.png" alt="MPDocBench-Parse overview" width="90%">
</p>

---

## Features

**1. Multi-page, End-to-End Evaluation**
Takes full multi-page documents (Pages 1, 2 … N) as input and evaluates parsing results end-to-end against ground truth, covering all content types in a single pass.

**2. Broad Document Diversity**
Spans 15 domains with documents in both English and Chinese, covering a wide range of layouts — from dense academic papers to multi-column magazines and presentation slides.

**3. Parsing Content Fidelity**
Evaluates the completeness and accuracy of extracted content across four element types:
- **Text** — normalized edit distance (Edit_dist), BLEU, METEOR
- **Formula** — Character Detection Matching (CDM)
- **Table** — Tree-Edit-Distance-based Similarity (TEDS)
- **Figure** — figure detection F1 (FigureF1)

Also assesses **semantic continuity** for cross-page truncated elements (truncated text merging, truncated table merging) via Relation F1.

**4. Logical Structural Correctness**
Evaluates the logical organization of parsed output:
- **Reading Order** — edit distance on the sequence of content blocks
- **Hierarchical Structure** — HeadTEDS for document heading hierarchy

---

## Installation

### Prerequisites

- Python 3.10+
- Node.js, ImageMagick, LaTeX *(optional — only required for CDM formula evaluation)*

### Setup

```bash
conda create -n mpdocbench python=3.10 -y
conda activate mpdocbench
pip install -r requirements.txt
```

### CDM Environment *(Optional)*

If you need formula evaluation with CDM, please refer to the [CDM Installation Guide](./metrics/cdm/README.md) for additional dependencies (Node.js, ImageMagick, TeX Live).

---

## Data Download

### 1. Download the MPDocBench Dataset

Visit the following link to download the dataset, then extract it into the `MPDocBench/` directory:

```bash
wget https://www.modelscope.cn/datasets/zhoubb/MPDocBench/file/view/master/MPDocBench_data.zip -O MPDocBench_data.zip
unzip MPDocBench_data.zip -d ./
rm MPDocBench_data.zip
```

### 2. Download SlideVQA Documents

Part of the benchmark relies on documents from the [SlideVQA](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA) dataset. Use the provided script to fetch the required subset:

```bash
python ./tools/download_data_from_slidevqa.py
```

> The script will download the necessary SlideVQA files into the appropriate location under `MPDocBench/images`.

---

## Quick Start

### 1. Prepare Your Data

| Type | Description |
|---|---|
| Ground Truth | A JSON file containing annotated document structure (default `MPDocBench.json`) |
| Prediction | A directory containing model output markdown files (one `.md` per PDF document) |

You can refer to the inference scripts under [`tools/model_infer/`](tools/model_infer/) to run inference with various models (e.g. Chandra, Dolphin, GLM-OCR, MinerU, MonkeyOCR-Pro, PP-OCR-VL, Qwen-VL, etc.). After obtaining the raw model outputs, run the corresponding notebook under [`tools/model_infer/prediction_to_md/`](tools/model_infer/prediction_to_md/) to convert the predictions into the markdown format expected by the evaluator.

### 2. Run Evaluation

```bash
python pdf_validation.py --config ./configs/end2end.yaml
```

Results will be saved to the `result/` directory.

### Supported Metrics

| Metric | Task | Description |
|:---|:---|:---|
| Edit_dist | Text / Reading Order / Truncated Text Merging | Normalized edit distance |
| BLEU | Text | Bilingual Evaluation Understudy |
| METEOR | Text | Metric for Evaluation of Translation with Explicit Ordering |
| TEDS | Table / Truncated Table Merging | Tree-Edit-Distance-based Similarity |
| CDM | Formula | Character Detection Matching |
| Relation_F1 | Table/Text Relations | F1 score for truncated merging relations |
| HeadTEDS | Heading | Tree-edit distance for heading structure |
| FigureF1 | Figure | F1 score for figure detection |

---

## Evaluation Result

> An interactive leaderboard with sorting and filtering is available at [leaderboard](https://bang123-box.github.io/MPDocBench-Parsing-Leaderboard/).

Models evaluated on **MPDocBench-Parse**, including general-purpose VLMs and specialized OCR/parsing models across all supported metrics.

| Model | Overall | Text Edit | Truncated Text Edit | Formula CDM | Table TEDS | Truncated Table TEDS | Figure F1 | Read Order | Heading TEDS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pipeline-based Specialized VLMs** | | | | | | | | | |
| GLM-OCR | 75.01 | 0.062 | 0.313 | 87.62 | 82.69 | 63.15 | 71.63 | 0.126 | 44.95 |
| PaddleOCR-VL-1.5 | **80.70** | 0.048 | **0.152** | 87.14 | 83.87 | 83.09 | **74.99** | 0.106 | 47.11 |
| MinerU2.5 | 77.30 | 0.060 | 0.326 | 85.42 | 86.18 | 88.02 | 72.32 | 0.120 | 37.01 |
| MinerU2.5 pro | 79.77 | 0.077 | 0.191 | 88.00 | **89.23** | **91.44** | 72.77 | 0.125 | 36.10 |
| Youtu-Parsing | 74.34 | 0.091 | 0.343 | 86.87 | 85.14 | 63.95 | 71.62 | 0.130 | 43.55 |
| MonkeyOCR-pro-3B | 74.41 | 0.055 | 0.302 | 88.53 | 78.76 | 61.22 | 73.69 | 0.119 | 40.65 |
| Dolphin-v2 | 73.07 | 0.106 | 0.333 | 79.58 | 84.16 | 64.31 | 63.98 | 0.138 | 50.23 |
| **End-to-End Specialized VLMs** | | | | | | | | | |
| dots.mocr | 72.94 | 0.070 | 0.300 | 86.03 | 81.94 | 62.21 | 67.85 | 0.113 | 33.70 |
| FireRed-OCR | 69.29 | **0.042** | 0.179 | **89.68** | 81.88 | 62.63 | 0.00 | 0.087 | **50.84** |
| dots.ocr | 74.19 | 0.074 | 0.305 | 85.88 | 83.46 | 60.71 | 67.59 | 0.114 | 45.21 |
| DeepSeek-OCR2 | 76.43 | 0.068 | 0.256 | 86.62 | 81.25 | 63.02 | 73.46 | 0.104 | 49.93 |
| OCRVerse | 64.47 | 0.104 | 0.293 | 86.89 | 84.14 | 63.99 | 0.00 | 0.152 | 35.63 |
| Logics-Parsing-v2 | 74.57 | 0.047 | 0.313 | 86.67 | 83.95 | 63.88 | 71.71 | **0.085** | 34.72 |
| Qianfan-OCR | 71.89 | 0.095 | 0.467 | 88.74 | 83.18 | 62.46 | 58.44 | 0.110 | 49.48 |
| ChandraOCR 2 | 74.62 | 0.097 | 0.294 | 86.67 | 84.74 | 64.70 | 64.66 | 0.134 | 48.69 |
| **General VLMs** | | | | | | | | | |
| Gemini-3.1-pro-preview | 71.94 | 0.070 | 0.223 | 88.37 | 81.99 | 61.30 | 58.93 | 0.127 | 26.90 |
| ChatGPT-5.2-2025-12-11 | 65.47 | 0.111 | 0.387 | 84.33 | 79.31 | 58.85 | 30.90 | 0.170 | 37.23 |
| Qwen3.6-plus | 71.95 | 0.095 | 0.260 | 88.77 | 83.33 | 60.53 | 64.77 | 0.182 | 31.94 |
| Qwen3-VL-235B | 74.00 | 0.088 | 0.187 | 84.71 | 81.64 | 61.28 | 63.20 | 0.138 | 42.41 |
| InternVL-3.5-38B | 57.18 | 0.131 | 0.502 | 84.30 | 69.93 | 51.63 | 8.02 | 0.198 | 26.78 |

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

## Copyright Statement

The PDF documents included in this benchmark are sourced from publicly accessible internet resources as well as voluntary contributions from the open-source community. Any material that is not permitted for redistribution has been carefully filtered out before release. This dataset is intended **solely for academic and research purposes** and must not be used for any commercial activity.

If any content in this benchmark raises copyright concerns, please reach out to us and we will address the issue promptly.

---

## Acknowledgments

- [OmniDocBench](https://github.com/opendatalab/OmniDocBench) 
- [CDM](https://github.com/opendatalab/UniMERNet/tree/main/cdm) 
- [READoc](https://github.com/icip-cas/READoc) 

---

## Citation

```bibtex
@misc{ouyang2024omnidocbenchbenchmarkingdiversepdf,
      title={OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations},
      author={Linke Ouyang and Yuan Qu and Hongbin Zhou and Jiawei Zhu and Rui Zhang and Qunshu Lin and Bin Wang and Zhiyuan Zhao and Man Jiang and Xiaomeng Zhao and Jin Shi and Fan Wu and Pei Chu and Minghao Liu and Zhenxiang Li and Chao Xu and Bo Zhang and Botian Shi and Zhongying Tu and Conghui He},
      year={2024},
      eprint={2412.07626},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.07626},
}
```
