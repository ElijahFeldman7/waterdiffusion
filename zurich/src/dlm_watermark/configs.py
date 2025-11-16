from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import StrEnum

class WatermarkType(StrEnum):
    NONE = "None"
    OURS = "Ours"
    KGW = "KGW"
    KTH = "KTH"
    AAR = "AAR"
    ORDER_AGNOSTIC = "OrderAgnostic"
    UNIGRAM = "Unigram" 

class NoWatermarkConfiguration(BaseModel):
    """
    Configuration for no watermarking. Useful to get PPL of the base model.
    """

    def short_summary(self):
        """
        Print a short summary of the no watermark configuration
        """
        print("No watermarking is applied.")

class OurWatermarkConfiguration(BaseModel):
    delta: float = Field(default=2, description="Maximum distortion allowed.")
    enforce_kl: bool = Field(
        default=True,
        description="Whether to enforce KL divergence constraint in the optimization.",
    )
    convolution_kernel: List[int] = Field(default_factory=lambda: [-1])
    topk: int = Field(default=100, description="Top-k tokens used in the energy.")
    n_iter: int = Field(
        default=1, description="Number of fixed-point iterations."
    )
    seeding_scheme: str = Field(
        default="sumhash",
        description="Seeding scheme for the watermark. Options: 'sumhash', 'minhash'.",
    )
    greenlist_type: str = Field(
        default="greenify",
        description="Type of greenlist to use. Options: 'bernoulli', 'gaussian' and 'lognormal'.",
    )
    greenlist_params: dict = Field(
        default_factory=lambda: {"gamma": 0.25},
        description="Parameters for the greenlist. For 'bernoulli', use {'gamma': float}. For 'gaussian' or 'lognormal', use {}.",
    )
    greenify_only: bool = Field(
        default=False,
        description="Whether to use the predictive bias term only.",
    )
    booster_only: bool = Field(
        default=False,
        description="Whether to use the expectation boost term only.",
    )

    def short_summary(self):
        """
        Print a short summary of the watermark configuration
        """
        print(f"Delta: {self.delta}")
        print(f"Enforce KL: {self.enforce_kl}")
        print(f"Convolution Kernel: {self.convolution_kernel}")
        print(f"Seeding Scheme: {self.seeding_scheme}")
        print(f"Top-k: {self.topk}")
        print(f"n_iter: {self.n_iter}")
        print(f"Greenlist Type: {self.greenlist_type}")
        print(f"Greenlist Parameters: {self.greenlist_params}")
        print(f"Predictive Bias Only: {self.greenify_only}")
        print(f"Expectation Boost Only: {self.booster_only}")

class UnigramConfiguration(BaseModel):
    delta: float = Field(default=2.0, description="Logits boosting factor.")
    gamma: float = Field(default=0.25, description="Percentage of green tokens.")
    seed: int = Field(default=42, description="Random seed for greenlist generation.")

    def short_summary(self):
        """
        Print a short summary of the Unigram watermark configuration
        """
        print(f"Delta: {self.delta}")
        print(f"Gamma: {self.gamma}")
        print(f"Seed: {self.seed}")

class OrderAgnosticConfiguration(BaseModel):
    delta: float = Field(default=2.0, description="Logits boosting factor.")
    l: int = Field(default=2, description="Number of colors.")
    pattern_length: int = Field(default=4, description="Length of each pattern.")
    patterns: List[List[int]] = Field(
        default_factory=lambda: [[0, 1, 0, 1], [1, 0, 1, 0]],
        description="List of patterns to use for the watermark.",
    )
    transition_matrix: Optional[List[List[float]]] = Field(
        default_factory=lambda: [[0, 0.5], [0.5, 0]],
        description="Transition matrix for the Markov chain.",
    )
    initial_state: Optional[List[float]] = Field(
        default_factory=lambda: [0.5, 0.5],
        description="Initial state distribution for the Markov chain.",
    )

    def short_summary(self):
        """
        Print a short summary of the OrderAgnostic watermark configuration
        """
        print(f"Delta: {self.delta}")
        print(f"Number of colors (l): {self.l}")
        print(f"Pattern length: {self.pattern_length}")
        print(f"Patterns: {self.patterns}")
        print(f"Transition Matrix: {self.transition_matrix}")
        print(f"Initial State: {self.initial_state}")

class KGWConfiguration(BaseModel):
    """
    Hyperparameters for the KGW baseline
    """

    # Watermark parameters
    gamma: float = Field(default=0.25, description="Percentage of green tokens.")
    delta: float = Field(default=2.0, description="Logits boosting factor.")
    convolution_kernel: List[int] = Field(default_factory=lambda: [-2, -1])
    seeding_scheme: str = Field(
        default="sumhash",
        description="Seeding scheme for the watermark. Options: 'sumhash', 'minhash'.",
    )

    def short_summary(self):
        """
        Print a short summary of the watermark configuration
        """
        print(f"Gamma: {self.gamma}")
        print(f"Delta: {self.delta}")
        print(f"Convolution Kernel: {self.convolution_kernel}")
        print(f"Seeding Scheme: {self.seeding_scheme}")

class KTHConfiguration(BaseModel):
    key_len: int = Field(default=16, description="Length of the key for KTH watermark.")
    seed: int = Field(default=42, description="Seed for KTH watermark generation.")

    def short_summary(self):
        """
        Print a short summary of the KTH watermark configuration
        """
        print(f"Key Length: {self.key_len}")
        print(f"Seed: {self.seed}")

class AARConfiguration(BaseModel):
    """
    Hyperparameters for AAR watermark
    """
    convolution_kernel: List[int] = Field(default_factory=lambda: [-2, -1])

    def short_summary(self):
        """
        Print a short summary of the AAR watermark configuration
        """
        print(f"Convolution Kernel: {self.convolution_kernel}")


class EvaluationDataset(BaseModel):
    path: str = Field(..., description="Path to HF dataset.")
    name: Optional[str] = Field(default=None, description="Name of the dataset.")
    split: str = Field(default="train", description="Split of the dataset.")
    streaming: bool = Field(default=True, description="Whether to stream the dataset.")
    chat: bool = Field(default=True, description="Whether to use chat template.")
    max_length: int = Field(
        default=512, description="Maximum length of the input sequence."
    )
    min_length: int = Field(
        default=0, description="Minimum length of the input sequence."
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for chat datasets."
    )
    user_prompt: str = Field(
        default="{text}", description="User prompt for chat datasets."
    )

    def short_summary(self):
        """
        Print a short summary of the evaluation dataset configuration
        """
        print(f"Path: {self.path}")
        print(f"Chat: {self.chat}")
        print(f"Max Length: {self.max_length}")


class EvaluationConfiguration(BaseModel):
    """
    Evaluation configuration class
    """

    # Evaluation parameters
    num_samples: int = Field(default=1000, description="Number of samples to evaluate.")
    batch_size: int = Field(default=1, description="Batch size for evaluation.")

    # Datasets
    evaluation_datasets: List[EvaluationDataset] = Field(
        default_factory=list, description="List of evaluation datasets."
    )

    # PPL evaluation
    ppl_model_name: Optional[str] = Field(
        default="Qwen/Qwen2.5-32B-Instruct",
        description="Model name for perplexity evaluation.",
    )
    ppl_batch_size: int = Field(
        default=10, description="Batch size for perplexity evaluation."
    )

    # IO
    save_path: str = Field(
        default="watermark_results.jsonl",
        description="Path to save the evaluation results",
    )
    skip_if_exists: bool = Field(
        default=True,
        description="Skip generation if results already exist. Only applies for evaluate_watermark.",
    )

    # Others
    tqdm: bool = Field(default=True, description="Whether to use tqdm for progress bar.")

    def short_summary(self):
        """
        Print a short summary of the evaluation configuration
        """
        print(f"Num Samples: {self.num_samples}")
        print(f"Save Path: {self.save_path}")
        print("Evaluation Dataset:")
        for dataset in self.evaluation_datasets:
            dataset.short_summary()


class ModelConfiguration(BaseModel):
    # Model loading parameters
    model_name: str = Field(..., description="Name of the model.")
    tokenizer_name: str = Field(..., description="Name of the tokenizer.")
    torch_dtype: str = Field(default="bfloat16", description="Torch data type.")
    device_map: str = Field(default="cuda", description="Device map for model loading.")
    trust_remote_code: bool = Field(
        default=True, description="Trust remote code for model loading."
    )

    # Model generation parameters
    steps: int = Field(default=128, description="Number of diffusion steps.")
    gen_length: int = Field(default=128, description="Length of the generated sequence.")
    temperature: float = Field(default=1.0, description="Temperature for sampling.")
    remasking: str = Field(default="low_confidence", description="Remasking strategy. Options: 'low_confidence', 'random', 'ar'.")

    # For model-specific arguments, refer to models/<model type>.py to see the available options
    model_specific_arguments: Optional[dict] = {
        "cfg_scale": 0.0,
        "block_length": 128,
        "mask_id": 126336,
    }

    def short_summary(self):
        """
        Print a short summary of the model configuration
        """
        print("Model Name:", self.model_name)
        print("Tokenizer Name:", self.tokenizer_name)
        print("Torch Data Type:", self.torch_dtype)
        print("Device Map:", self.device_map)
        print("Trust Remote Code:", self.trust_remote_code)


class MainConfiguration(BaseModel):
    """
    Main configuration class
    """

    model_configuration: ModelConfiguration = Field(
        default_factory=ModelConfiguration,
        description="Model configuration.",
    )
    evaluation_config: Optional[EvaluationConfiguration] = Field(
        default_factory=EvaluationConfiguration,
        description="Evaluation configuration.",
    )
    watermark_type: WatermarkType = Field(
        ...,
        description="Type of watermarking to use.",
    )
    watermark_config: Union[
        OurWatermarkConfiguration,
        KGWConfiguration,
        KTHConfiguration,
        NoWatermarkConfiguration,
        AARConfiguration,
        OrderAgnosticConfiguration,
        UnigramConfiguration,
    ] = Field(..., description="Parameters for the chosen watermark algorithm.")

    @field_validator("watermark_config", mode="before")
    @classmethod
    def _dispatch_watermark_config(cls, raw, info):
        """
        Before validating `watermark_config`, pick the right subclass
        based on the already‐parsed `watermark_type` in info.data.
        """
        wt = info.data.get("watermark_type")
        # map each enum to its config class
        dispatch_map = {
            WatermarkType.OURS: OurWatermarkConfiguration,
            WatermarkType.KGW: KGWConfiguration,
            WatermarkType.KTH: KTHConfiguration,
            WatermarkType.NONE: NoWatermarkConfiguration,
            WatermarkType.AAR: AARConfiguration,
            WatermarkType.ORDER_AGNOSTIC: OrderAgnosticConfiguration,
            WatermarkType.UNIGRAM: UnigramConfiguration,
        }
        ModelCls = dispatch_map.get(wt)
        if ModelCls is None:
            raise ValueError(f"Unsupported watermark_type: {wt}")
        # if the user didn’t provide any nested config, raw might be None
        data = raw or {}
        # use the v2 entry‐point to parse
        return ModelCls.model_validate(data)

    def short_summary(self):
        """
        Print a short summary of the configuration
        """
        print("Model Configuration:")
        self.model_configuration.short_summary()
        print("\nEvaluation Configuration:")
        self.evaluation_config.short_summary()
        print("Watermark Type:", self.watermark_type.value)
        print("\nWatermark Configuration:")
        self.watermark_config.short_summary()
