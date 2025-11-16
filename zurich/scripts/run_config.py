from dlm_watermark.configs import MainConfiguration
from dlm_watermark.watermarks.watermark_factory import load_watermark_from_config
from dlm_watermark.watermark_eval import Evaluator
from dlm_watermark.models.model_factory import load_model
import yaml
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description="Ablate generation parameters")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--output_path", type=str, help="Path to save the output results", default="outputs")
    parser.add_argument("--num_samples", type=int, default=None, help="Overwrite number of samples to evaluate.")
        
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()
    config = MainConfiguration(**yaml.safe_load(open(args.config, "r")))
    
    config.evaluation_config.save_path = f"{args.output_path}/results.json"
    print(config.short_summary())

    additional_info = {
        "model_name": config.model_configuration.model_name,
    }

    if args.num_samples is not None:
        config.evaluation_config.num_samples = args.num_samples
    evaluator = Evaluator(config=config.evaluation_config)
    
    model, tokenizer = load_model(config.model_configuration, tokenizer_only=False)
    watermark = load_watermark_from_config(config=config.watermark_config, tokenizer=tokenizer, watermark_type=config.watermark_type)
    evaluator.evaluate_watermark(model,tokenizer,watermark, additional_info=additional_info)


if __name__ == "__main__":
    main()