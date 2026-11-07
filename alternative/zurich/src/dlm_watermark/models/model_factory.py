from ..configs import ModelConfiguration
from .llada import LLaDA
from .dream import Dream
from .dreamon import DreamOn
from transformers import AutoModel, AutoTokenizer
from ..constants import DTYPE_MAP
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from .model import DiffusionLM

def _load_model(model_config: ModelConfiguration, tokenizer_only: bool = False):
    """
    Load the model and tokenizer
    """
    torch_dtype = DTYPE_MAP[model_config.torch_dtype]

    if tokenizer_only:
        model = None
    else:
        model = AutoModel.from_pretrained(
            model_config.model_name,
            torch_dtype=torch_dtype,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name,
        trust_remote_code=model_config.trust_remote_code,
        padding_side='left'
    )
    return model, tokenizer

def load_model(model_config: ModelConfiguration, tokenizer_only: bool = False) -> Tuple['DiffusionLM', AutoTokenizer]:
    
    model, tokenizer = _load_model(model_config, tokenizer_only=tokenizer_only)
    
    if "LLaDA" in model_config.model_name:
        model = LLaDA(config=model_config, hf_model=model)
    elif "DreamOn" in model_config.model_name:
        model = DreamOn(config=model_config, hf_model=model)
    elif "Dream" in model_config.model_name:
        model = Dream(config=model_config, hf_model=model)
    else:
        raise NotImplementedError(f"Model {model_config.model_name} is not implemented.")
    return model, tokenizer
