from dlm_watermark.configs import MainConfiguration
from dlm_watermark.watermarks.watermark_factory import load_watermark_from_config
from dlm_watermark.watermark_eval import Evaluator
from dlm_watermark.models.model_factory import load_model
import yaml
import argparse
import json

def parse_args():
    
    parser = argparse.ArgumentParser(description="Ablate generation parameters")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--name", type=str, help="Name of the ablation experiment")
    parser.add_argument("--delta", type=float, default=None, help="Delta value for watermark")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma value for watermark")
    parser.add_argument("--kernel", type=str, default=None, help="Kernel type for convolution")
    parser.add_argument("--topk", type=int, default=None, help="Top-k value for watermark")
    parser.add_argument("--topk_greenify", type=int, default=None, help="Top-k value for greenify")
    parser.add_argument("--topk_hashes", type=int, default=None, help="Top-k value for hashes")
    parser.add_argument("--n_iter", type=int, default=None)
    parser.add_argument("--seeding_scheme", type=str, default=None, help="Seeding scheme for watermark")
    parser.add_argument("--disable_generation", action="store_true", help="Disable generation during evaluation")
    parser.add_argument("--ppl", action="store_true", help="Evaluate perplexity")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    
    # OptimalKGWGeneral specific parameters
    parser.add_argument("--enforce_kl", action="store_true", help="Enforce KL divergence constraint in the optimization")
    parser.add_argument("--no_enforce_kl", action="store_true", help="Disable KL divergence constraint in the optimization")
    parser.add_argument("--greenlist_type", type=str, default=None, help="Type of greenlist to use. Options: 'bernoulli', 'gaussian'")
    parser.add_argument("--greenlist_params", type=str, default=None, help="Parameters for the greenlist as JSON string (e.g., '{\"gamma\": 0.25}')")
    
    # Generation specific parameters
    parser.add_argument("--temperature", type=float, default=None, help="Temperature parameter")
        
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    delta = args.delta
    kernel = eval(args.kernel) if args.kernel is not None else None
    topk = args.topk
    
    default_config = args.config
    config = MainConfiguration(**yaml.safe_load(open(default_config, "r")))
    
    config.evaluation_config.save_path = f"output/{args.name}/watermark_ablation.jsonl"
    config.evaluation_config.num_samples = args.num_samples if args.num_samples is not None else config.evaluation_config.num_samples
    if delta==0:
        config.evaluation_config.save_path = f"output/{args.name}/watermark_ablation_no_watermark.jsonl"
    
    if config.watermark_type.value == "None":
        print("No watermark type specified -- evaluating without watermark.")
    elif config.watermark_type.value == "KGW":
        config.watermark_config.delta = delta if delta is not None else config.watermark_config.delta
        config.watermark_config.convolution_kernel = kernel if kernel is not None else config.watermark_config.convolution_kernel
        config.watermark_config.seeding_scheme = args.seeding_scheme if args.seeding_scheme is not None else config.watermark_config.seeding_scheme
    elif config.watermark_type.value == "Ours":
        config.watermark_config.delta = delta if delta is not None else config.watermark_config.delta
        config.watermark_config.convolution_kernel = kernel if kernel is not None else config.watermark_config.convolution_kernel
        config.watermark_config.topk = topk if topk is not None else config.watermark_config.topk
        config.watermark_config.n_iter = args.n_iter if args.n_iter is not None else config.watermark_config.n_iter
        config.watermark_config.seeding_scheme = args.seeding_scheme if args.seeding_scheme is not None else config.watermark_config.seeding_scheme
        
        # Handle enforce_kl parameter
        if args.enforce_kl:
            config.watermark_config.enforce_kl = True
        elif args.no_enforce_kl:
            config.watermark_config.enforce_kl = False
        
        # Handle greenlist_type parameter
        if args.greenlist_type is not None:
            config.watermark_config.greenlist_type = args.greenlist_type
        
        # Handle greenlist_params parameter
        if args.greenlist_params is not None:
            try:
                config.watermark_config.greenlist_params = json.loads(args.greenlist_params)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON for greenlist_params: {args.greenlist_params}")
                print("Using default greenlist_params")
    elif config.watermark_type.value == "AAR":
        config.watermark_config.convolution_kernel = kernel if kernel is not None else config.watermark_config.convolution_kernel
    else:
        config.watermark_config.delta = delta if delta is not None else config.watermark_config.delta
        config.watermark_config.convolution_kernel = kernel if kernel is not None else config.watermark_config.convolution_kernel
        config.watermark_config.seeding_scheme = args.seeding_scheme if args.seeding_scheme is not None else config.watermark_config.seeding_scheme
    
    print(config.short_summary())

    additional_info = {
        "model_name": config.model_configuration.model_name,
    }

    evaluator = Evaluator(config=config.evaluation_config)
    
    if not args.disable_generation:   
        print(f"Evaluating with watermark: {delta}, kernel: {kernel}, topk: {topk}")
        model, tokenizer = load_model(config.model_configuration, tokenizer_only=False)    

        # Generation parametters
        model.config.temperature = args.temperature if args.temperature is not None else model.config.temperature

        if delta==0:
            watermark = None
        else:
            watermark = load_watermark_from_config(config=config.watermark_config, tokenizer=tokenizer, watermark_type=config.watermark_type)
        evaluator.evaluate_watermark(model,tokenizer,watermark, additional_info=additional_info)
        
    if args.ppl:
        evaluator.evaluate_ppl()


if __name__ == "__main__":
    main()
