<p align="center" width="100%">
  <h1 align="center">Qwen-Doc</h1>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
    <a href="https://github.com/Tongyi-Zhiwen/Qwen-Doc"><img src="https://img.shields.io/badge/GitHub-Qwen--Doc-4b32c3?logo=github"></a>
    <a href=""><img src="https://img.shields.io/badge/Maintained%20by-Tongyi--Zhiwen-orange"></a>
</p>

<p align="center">
    <i>An Open-Source Collection of Projects on Document Understanding, Parsing, and Agents</i>
</p>

## 📖 Introduction

**Qwen-Doc** is an open-source repository dedicated to Document AI, developed and maintained by the Tongyi-Zhiwen team.

This repository aims to bring together a series of explorations and practices centered on cutting-edge technologies such as long-context understanding, document parsing, and document-based intelligent agents. We are committed to enhancing the capabilities of Large Language Models in processing and comprehending complex documents, and we open-source our models, data, and methodologies to foster community growth.

## 🎉 News

- **Dec 15, 2025:** 🔥 We released the **QwenLong-L1.5** project! It provides a complete post-training recipe for long-context reasoning and memory management. The corresponding model and technical report have also been released.
- **Dec 15, 2025:** 🔥 We released the code implementation of **SPELL**, which is a self-play reinforcement learning framework designed to improve long-context reasoning abilities in LLMs.
- **May 28, 2025:** 🔥 The **QwenLong-L1** project released `QwenLong-L1-32B-AWQ`, a version processed with AWQ int4 quantization.
- **May 26, 2025:** 🔥 We officially open-sourced the **QwenLong-L1** project, the industry's first large model trained for long-context reasoning using reinforcement learning. We also released the accompanying `QwenLong-L1-32B` model and the `DocQA-RL-1.6K` training dataset.

## 📂 Project List

This repository currently includes the following projects:

### 1. [QwenLong-L1](./QwenLong-L1)

- **Description:** A framework designed to generalize Large Models from short-context proficiency to robust long-context reasoning capabilities using Reinforcement Learning. This project explores mechanisms like curriculum learning and difficulty-aware sampling, and releases the **QwenLong-L1-32B** model trained on this framework, which has achieved state-of-the-art performance on multiple long-context document question answering (DocQA) benchmarks.

### 2. [QwenLong-L1.5](./QwenLong-L1.5)

- **Description:** A complete "Post-Training Recipe" for long-context reasoning and memory management. This project features three core contributions: a synthesis pipeline for generating complex reasoning data, the Adaptive Entropy-Controlled Policy Optimization (AEPO) algorithm optimized for long-context training, and a memory management framework that extends operation beyond the model's physical context window. Based on this recipe, we introduce the **QwenLong-L1.5-30B-A3B** model.

### 3. [SPELL](./SPELL)

- **Description:** A self-play reinforcement learning framework designed to improve long-context reasoning abilities in LLMs. SPELL cycles a single LLM through three roles—questioner, responder, and verifier—to autonomously generate training data and rewards, without requiring external supervision. Extensive experiments across 12 models and 6 benchmarks demonstrate consistent improvements. Notably, SPELL provides a potential path for elevating the performance ceiling of models surpassing human performance.


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tongyi-Zhiwen/Qwen-Doc&type=Timeline)](https://star-history.com/#Tongyi-Zhiwen/Qwen-Doc&Timeline)

## 📝 Citation

If you find our work helpful in your research, please consider citing our papers:

```bibtex
@article{wan2025qwenlongl1,
  title={QwenLong-L1: : Towards Long-Context Large Reasoning Models with Reinforcement Learning},
  author={Fanqi Wan, Weizhou Shen, Shengyi Liao, Yingcheng Shi, Chenliang Li, Ziyi Yang, Ji Zhang, Fei Huang, Jingren Zhou, Ming Yan},
  journal={arXiv preprint arXiv:2505.17667},
  year={2025}
}
```

```bibtex
@article{shen2025qwenlongl15,
  title={QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management},
  author={Weizhou Shen, Ziyi Yang, Chenliang Li, Zhiyuan Lu, Miao Peng, Huashan Sun, Yingcheng Shi, Shengyi Liao, Shaopeng Lai, Bo Zhang, Dayiheng Liu, Fei Huang, Jingren Zhou, Ming Yan},
  year={2025}
}
```

```bibtex
@article{yang2025spell,
    title={SPELL: Self-Play Reinforcement Learning for evolving Long-Context Language Models},
    author={Ziyi Yang, Weizhou Shen, Ruijun Chen, Chenliang Li, Fanqi Wan, Ming Yan, Xiaojun Quan, Fei Huang},
    journal={arXiv preprint arXiv:2509.23863},
    year={2025}
}
```