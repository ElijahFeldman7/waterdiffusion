"""
LLaDA Model Runner Script (cluster-friendly)

Quick examples:

1) Config + single prompt
   python scripts/run.py --config configs/llada_8b_instruct.yaml --prompt "Hello, world!"

2) Batch prompts and JSONL output
   python scripts/run.py --config configs/llada_8b_instruct.yaml \
     --prompts_file data/prompts.txt --num_generations 3 \
     --output_dir runs --run_name llada_infer --jsonl --csv

3) Local files only and fixed seed (reproducible)
   python scripts/run.py --model_path GSAI-ML/LLaDA-8B-Instruct --seed 42 \
     --local_files_only --gen_length 256 --steps 128

4) Watermark + detection
   python scripts/run.py --prompt "Tell me a story" --watermark_key "secret" --detect_watermark
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.llada import LLaDAGenerator
from watermarks.simple_remask_watermark import SimpleRemaskWatermark


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaDA model generation")
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["llada"],
        default="llada",
        help="Model type to use (currently only 'llada' supported)"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="HuggingFace model path (overrides config)"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for model loading (e.g., auto)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch dtype override: bfloat16, float16, float32"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to a text file with one prompt per line"
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., allenai/c4)"
    )
    parser.add_argument(
        "--hf_subset",
        type=str,
        default=None,
        help="Dataset config/subset (e.g., en.noblocklist)"
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="validation",
        help="Dataset split (e.g., train, validation)"
    )
    parser.add_argument(
        "--dataset_text_field",
        type=str,
        default="text",
        help="Field to read as prompt from dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to process from dataset"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream dataset instead of full download"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset"
    )
    parser.add_argument(
        "--filter_min_len",
        type=int,
        default=None,
        help="Filter dataset examples shorter than this length"
    )
    parser.add_argument(
        "--filter_max_len",
        type=int,
        default=None,
        help="Filter dataset examples longer than this length"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=1,
        help="Number of generations per prompt"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of diffusion steps (overrides config)"
    )
    
    parser.add_argument(
        "--gen_length",
        type=int,
        default=None,
        help="Length of text to generate in tokens (overrides config)"
    )
    
    parser.add_argument(
        "--block_length",
        type=int,
        default=32,
        help="Block length for generation"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config)"
    )
    
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale"
    )
    
    parser.add_argument(
        "--remasking",
        type=str,
        choices=["low_confidence", "random"],
        default="low_confidence",
        help="Remasking strategy"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu, overrides config)"
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load models/tokenizers from local cache"
    )
    
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat template formatting"
    )
    
    parser.add_argument(
        "--watermark_key",
        type=str,
        default=None,
        help="Secret key for watermarking (enables watermark if provided)"
    )
    
    parser.add_argument(
        "--watermark_delta",
        type=float,
        default=2.0,
        help="Delta value for watermark confidence boost (default: 2.0)"
    )
    
    parser.add_argument(
        "--detect_watermark",
        action="store_true",
        help="Run watermark detection on generated text"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs",
        help="Base directory to write run outputs"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name; otherwise timestamped"
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write results to results.jsonl"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write results to results.csv"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output files if present"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce stdout logging (cluster-friendly)"
    )
    parser.add_argument(
        "--save_full_sequence",
        action="store_true",
        help="Save full token sequence tensor index ranges"
    )
    
    return parser.parse_args()


def setup_run_dirs_and_logging(args):
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    auto_name_bits = [timestamp]
    if slurm_job_id:
        auto_name_bits.append(f"job{slurm_job_id}")
    if args.run_name:
        auto_name = f"{args.run_name}_{'_'.join(auto_name_bits)}"
    else:
        auto_name = "_".join(auto_name_bits)
    run_dir = base / auto_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.WARNING if args.quiet else logging.INFO
    log_file = run_dir / "run.log"
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Persist args and optional config path
    meta = {"args": vars(args)}
    if args.config and Path(args.config).is_file():
        try:
            cfg_text = Path(args.config).read_text()
        except Exception:
            cfg_text = None
        meta["config_path"] = str(args.config)
        if cfg_text is not None:
            (run_dir / "config.yaml").write_text(cfg_text)
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return run_dir


def set_global_seed(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(args):
    watermark = None
    if args.watermark_key:
        logging.info(f"Initializing watermark with key: {args.watermark_key[:10]}...")
        watermark = SimpleRemaskWatermark(
            config={},
            key=args.watermark_key,
            delta=args.watermark_delta
        )
    
    if args.model == "llada":
        if args.config:
            logging.info(f"Loading LLaDA model from config: {args.config}")
            generator = LLaDAGenerator(
                config=args.config,
                model_path=args.model_path,
                device=args.device,
                dtype=args.dtype,
                watermark=watermark,
                local_files_only=args.local_files_only,
                device_map=args.device_map
            )
        else:
            logging.info(f"Loading LLaDA model with explicit parameters")
            model_path = args.model_path or "GSAI-ML/LLaDA-8B-Instruct"
            generator = LLaDAGenerator(
                model_path=model_path,
                device=args.device or "cuda",
                dtype=args.dtype,
                watermark=watermark,
                local_files_only=args.local_files_only,
                device_map=args.device_map
            )
        return generator, watermark
    else:
        raise ValueError(f"Model type '{args.model}' not supported")


def main():
    args = parse_args()
    run_dir = setup_run_dirs_and_logging(args)

    if args.seed is not None:
        set_global_seed(args.seed)
        logging.info(f"Seed set to {args.seed}")

    logging.info("LLaDA Model Runner starting")
    generator, watermark = load_model(args)
    logging.info(f"Model: {generator.model_path}")
    logging.info(f"Device: {generator.device}")
    logging.info(f"Defaults -> length:{generator.default_gen_length} steps:{generator.default_steps} temp:{generator.default_temperature}")

    def dataset_prompt_iter():
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, args.hf_subset, split=args.hf_split, streaming=args.streaming)
        if not args.streaming and args.shuffle:
            ds = ds.shuffle(seed=args.seed or 0)
        count = 0
        for ex in (ds.shuffle(seed=args.seed or 0, buffer_size=10000) if args.streaming and args.shuffle else ds):
            text = ex.get(args.dataset_text_field) if isinstance(ex, dict) else ex[args.dataset_text_field]
            if text is None:
                continue
            s = text.strip()
            if args.filter_min_len is not None and len(s) < args.filter_min_len:
                continue
            if args.filter_max_len is not None and len(s) > args.filter_max_len:
                continue
            yield s
            count += 1
            if args.max_samples is not None and count >= args.max_samples:
                break

    if args.hf_dataset:
        prompt_source = dataset_prompt_iter()
    elif args.prompts_file:
        p = Path(args.prompts_file)
        if not p.exists():
            raise FileNotFoundError(f"prompts_file not found: {p}")
        prompts_list = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        prompt_source = iter(prompts_list)
    else:
        prompt_source = iter([args.prompt])

    # Prepare outputs
    jsonl_path = run_dir / "results.jsonl" if args.jsonl else None
    csv_path = run_dir / "results.csv" if args.csv else None
    jsonl_file = open(jsonl_path, "a" if args.append else "w") if jsonl_path else None
    csv_file = open(csv_path, "a" if args.append else "w") if csv_path else None
    if csv_file and (not args.append or csv_file.tell() == 0):
        csv_file.write("prompt_id,generation_id,prompt,response,is_watermarked,score,threshold\n")

    try:
        for prompt_id, prompt in enumerate(prompt_source):
            for gen_id in range(args.num_generations):
                logging.info(f"Generating [prompt={prompt_id}] [gen={gen_id}] ...")
                if args.detect_watermark and generator.watermark:
                    response, full_sequence = generator.generate_text(
                        prompt=prompt,
                        steps=args.steps,
                        gen_length=args.gen_length,
                        block_length=args.block_length,
                        temperature=args.temperature,
                        cfg_scale=args.cfg_scale,
                        remasking=args.remasking,
                        use_chat_template=not args.no_chat_template,
                        return_full_sequence=True
                    )
                else:
                    response = generator.generate_text(
                        prompt=prompt,
                        steps=args.steps,
                        gen_length=args.gen_length,
                        block_length=args.block_length,
                        temperature=args.temperature,
                        cfg_scale=args.cfg_scale,
                        remasking=args.remasking,
                        use_chat_template=not args.no_chat_template
                    )

                detection_result = None
                if args.detect_watermark and generator.watermark:
                    detection_result = generator.watermark.detect(
                        text=response,
                        tokenizer=generator.tokenizer,
                        model=generator.model
                    )

                # Stdout minimal
                if not args.quiet:
                    print("\n" + "="*60)
                    print(f"Prompt[{prompt_id}] Gen[{gen_id}] Response:")
                    print("="*60)
                    print(response)
                    print("="*60)
                
                record = {
                    "prompt_id": prompt_id,
                    "generation_id": gen_id,
                    "prompt": prompt,
                    "response": response,
                }
                if detection_result:
                    record.update({
                        "is_watermarked": detection_result.get("is_watermarked"),
                        "score": detection_result.get("score"),
                        "threshold": detection_result.get("threshold"),
                        "t_statistic": detection_result.get("t_statistic"),
                        "green_mean": detection_result.get("green_mean"),
                        "red_mean": detection_result.get("red_mean"),
                        "green_count": detection_result.get("green_count"),
                        "red_count": detection_result.get("red_count"),
                    })

                if jsonl_file:
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_file.flush()
                if csv_file:
                    csv_file.write(
                        f"{prompt_id},{gen_id}," +
                        json.dumps(prompt).strip('"') + "," +
                        json.dumps(response).strip('"') + "," +
                        (str(record.get("is_watermarked")) if detection_result else "") + "," +
                        (f"{record.get('score'):.6f}" if detection_result and record.get('score') is not None else "") + "," +
                        (str(record.get("threshold")) if detection_result else "") +
                        "\n"
                    )
                    csv_file.flush()
                
                if args.save_full_sequence and args.detect_watermark and 'full_sequence' in locals():
                    seq_path = run_dir / f"full_sequence_{prompt_id}_{gen_id}.pt"
                    try:
                        import torch
                        torch.save(full_sequence.cpu(), seq_path)
                    except Exception as e:
                        logging.warning(f"Failed saving full sequence: {e}")
    finally:
        if jsonl_file:
            jsonl_file.close()
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
