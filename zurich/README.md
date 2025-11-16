<div align="center"><h1>Watermarking Diffusion Language Models</h1></div>

This repository contains the implementation of *Watermarking Diffusion Language Models*.
Our watermark is the first watermark tailored for Diffusion Language Models.

## Overview

We present the first watermark tailored for Diffusion Language Models. Our watermark extends Red-Green watermarks, originally designed for Autoregressive Language Models, by applying them in expectation over the context hashes and, leveraging the capabilities of Diffusion Language Models, biasing tokens that lead to hashes making other tokens green. 

## Installation

### Prerequisites
- CUDA-compatible GPU with CUDA 12.9

### Setup

We recommend using `uv` to install the environment.

0. **Create a virtual environment:**
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

1. **Install the dependencies:**
```bash
uv pip install -r requirements.txt --torch-backend="auto"
```

2. **Install the main package:**
```bash
uv pip install -e .
```

### Optional Dependencies

For the [KTH](https://arxiv.org/abs/2307.15593) watermark baseline, we rely on a custom rust-based implementation of the detector.
It can be installed with:
```bash
uv pip install additional/levenshtein_rust-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl
```
For more information, refer to `additional/README.md`.

## Quick Usage 

While this repository contains all the code needed to reproduce our experiments, our watermark specific implementation is in `src/dlm_watermark/watermarks/diffusion_watermark.py`. 
To quickly evaluate our watermark, run
```bash
python scripts/run_config.py --config configs/main/Llada/ourWatermark_llada8b_instruct.yaml
```

Specifically, we configure the model and watermark through `.yaml` configuration files.
You can find examples of such configuration in `configs`.
For more information, please refer to `src/dlm_watermark/configs.py`.

## Project Structure

- `src/dlm_watermark/`: Python package with the watermark implementations, model wrappers, and helpers powering the experiments.
  - `watermarks/`: all watermark algorithms, including the diffusion watermark at `diffusion_watermark.py` and baselines for comparison.
  - `models/`: lightweight adapters around diffusion language models (e.g., Llada, Dream, DreamOn) and shared generation utilities.
  - `quality_evaluations/`: judges and metrics used to measure watermark impact (perplexity, quality scores, etc.).
  - `configs.py`: dataclasses describing the YAML configuration schema used throughout the project.
- `configs/`: ready-to-run YAML configs covering main experiments and ablations.
- `scripts/`: entrypoints for launching experiments, ablations, and evaluation pipelines; bash wrappers in `scripts/bash/` reproduce the paper main results.
- `data/`: small reference datasets (e.g., WaterBench, infilling prompts) needed for evaluating the watermark.
- `additional/`: optional Rust-based Levenshtein detector for the KTH baseline plus build instructions.

## Evaluation

We provide bash scripts in `scripts/bash` to reproduce our main experiments and all needed scripts are in the `scripts` folder.

## Citation

```
@misc{gloaguen2025watermarkingdiffusionlanguagemodels,
      title={Watermarking Diffusion Language Models}, 
      author={Thibaud Gloaguen and Robin Staab and Nikola Jovanović and Martin Vechev},
      year={2025},
      eprint={2509.24368},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24368}, 
}
```
