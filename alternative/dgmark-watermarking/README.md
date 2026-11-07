# dgMARK: Decoding-Guided Watermarking for Diffusion Language Models

Implementation of **dgMARK** on [LLaDA](https://github.com/ML-GSAI/LLaDA), which embeds watermarks by guiding the decoding order to prioritize parity-matching tokens.

For details, please refer to our [project page](https://dgmark-watermarking.github.io) and [paper](https://arxiv.org/abs/2601.22985).

Questions and feedback are welcome—please contact the authors.



## Getting Started
The following sections provide step-by-step instructions for setup and running the code.

### Setup

**Install dependencies**:
   ```bash
   # Install PyTorch with CUDA support (example)
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

   # Install other dependencies
   pip install -r requirements.txt
   ```


## Dataset Setup

This project uses the C4 validation dataset from HuggingFace. You have two options for dataset setup:

### Option 1: Automatic Download

The dataset will be downloaded automatically on first run:

```bash
python scripts/generate.py --method original --num_samples 10 --dataset_url https://huggingface.co/datasets/allenai/c4/resolve/main/realnewslike/c4-validation.00000-of-00001.json.gz
```

This will download `c4-validation.00000-of-00001.json.gz` (~364MB) to the current directory.

### Option 2: Manual Download

If you prefer to download manually or want to specify a custom location:

1. **Download the dataset**:
   ```bash
   wget https://huggingface.co/datasets/allenai/c4/resolve/main/realnewslike/c4-validation.00000-of-00001.json.gz
   ```

2. **Use custom dataset path**:
   ```bash
   python scripts/generate.py --dataset_path /path/c4-validation.00000-of-00001.json.gz --method original
   ```

## Quick Start

```bash
python scripts/generate.py --method original --num_samples 10
python scripts/generate.py --method watermark --num_samples 10 --sampling_strategy multinomial
python scripts/detect.py --watermarked generated_results_watermark.csv --original generated_results_original.csv --plot detection_results.png
```

### 1. Generate Baseline (Non-Watermarked) Text

```bash
python scripts/generate.py --method original --num_samples 50 --output_prefix baseline
```


### 2. Generate Watermarked Text

```bash
python scripts/generate.py --method watermark --num_samples 50 --sampling_strategy multinomial
```

### 3. Run Detection Analysis

```bash
python scripts/detect.py \
  --watermarked generated_results_watermark.csv \
  --original generated_results_original.csv \
  --plot detection_results.png
```

## Generation Methods

This repository supports the following generation methods:

### 1. Non-Watermark (Greedy or Multinomial Sampling)
Basic LLaDA generation (default: greedy). Multinomial sampling is also supported:
```bash
python scripts/generate.py --method original --sampling_strategy greedy
python scripts/generate.py --method original --sampling_strategy multinomial --top_k 3 
```

### 2. Watermark (Greedy or Multinomial Sampling)
```bash
python scripts/generate.py --method watermark --sampling_strategy greedy 
python scripts/generate.py --method watermark --sampling_strategy multinomial --top_k 3 
```

### 3. Beam Search  

**With greedy sampling:**
```bash
python scripts/generate.py --method beam --beam_size 3 --sampling_strategy greedy
```

**With multinomial sampling:**
```bash
python scripts/generate.py --method beam --beam_size 3 --sampling_strategy multinomial --top_k 3
```

Setting `--beam_size 1` falls back to standard watermarking.

## Detection Analysis

### Statistical Detection

The detection system uses z-score analysis to distinguish watermarked from non-watermarked text:

```bash
python scripts/detect.py \
  --watermarked results_watermarked.csv \
  --original results_original.csv \
  --threshold_z 4.0 \
  --min_length 200 \
  --plot detection_plot.png
```

### Key Parameters

- `--threshold_z`: Z-score threshold for detection (default: 4.0)
- `--min_length`: Minimum sequence length to include (default: 200)
- `--plot`: Save detection visualization plot


## Robust Detection Workflow


**1. Analyze with robust detection:**
```bash
# Analyze original text
python scripts/robust_detection.py --mode analyze \
  --input_csv generated_results_original.csv \
  --output_scores ./original_z_scores.txt \
  --window_size 8

# Analyze watermarked text
python scripts/robust_detection.py --mode analyze \
  --input_csv generated_results_watermark.csv \
  --output_scores ./watermarked_z_scores.txt \
  --window_size 8
```

**Optional private key.** If desired, add `--private_key` to both `scripts/generate.py` and `scripts/robust_detection.py --mode analyze` to randomize the parity mapping using a private-key seed. The same key must be used for generation and analysis. We recommend the default configuration for most use cases.



**2. Compute AUC value:**
```bash
python scripts/robust_detection.py --mode auc \
  --original original_z_scores.txt \
  --watermark watermarked_z_scores.txt
```


## Citation

```bibtex
@article{hong2026dgmark,
  title={dgMARK: Decoding-Guided Watermarking for Diffusion Language Models}, 
  author={Pyo Min Hong and Albert No},
  journal={arXiv preprint arXiv:2601.22985},
  year={2026}
}
```