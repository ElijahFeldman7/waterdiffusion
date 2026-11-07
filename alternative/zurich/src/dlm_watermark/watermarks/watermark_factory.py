from .kgw import KGWWatermarkAR
from .diffusion_watermark import OurWatermark
from .aar import AARWatermark
from .order_agnostic import OrderAgnosticWatermark
from .unigram import UnigramWatermark
from ..configs import (
    WatermarkType,
    KGWConfiguration,
    KTHConfiguration,
    OurWatermarkConfiguration,
    AARConfiguration,
    OrderAgnosticConfiguration,
    UnigramConfiguration,
)
from transformers import AutoTokenizer
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .watermark_interface import Watermark


def load_watermark_from_config(
    config: Union[
        KGWConfiguration,
        KTHConfiguration,
        AARConfiguration,
        OrderAgnosticConfiguration,
        OurWatermarkConfiguration,
        UnigramConfiguration
    ],
    tokenizer: AutoTokenizer,
    watermark_type: WatermarkType,
) -> "Watermark":

    if watermark_type.value == "KGW":
        return KGWWatermarkAR(
            gamma=config.gamma,
            delta=config.delta,
            conv_kernel=config.convolution_kernel,
            tokenizer=tokenizer,
            seeding_scheme=config.seeding_scheme,
        )

    elif watermark_type.value == "KTH":
        from .kth import (
            KTHWatermark,
        )  # This is to avoid import KTH if not used

        return KTHWatermark(
            vocab_size=tokenizer.vocab_size,
            key_len=config.key_len,
            seed=config.seed,
        )

    elif watermark_type.value == "Ours":
        return OurWatermark(
            delta=config.delta,
            enforce_kl=config.enforce_kl,
            convolution_kernel=config.convolution_kernel,
            greenlist_type=config.greenlist_type,
            greenlist_params=config.greenlist_params,
            topk=config.topk,
            n_iter=config.n_iter,
            seeding_scheme=config.seeding_scheme,
            tokenizer=tokenizer,
            booster_only= config.booster_only,
            greenify_only=config.greenify_only,
        )
    elif watermark_type.value == "None":
        return None
    elif watermark_type.value == "AAR":
        return AARWatermark(
            vocab_size=tokenizer.vocab_size,
            conv_kernel=config.convolution_kernel,
        )
    elif watermark_type.value == "OrderAgnostic":

        return OrderAgnosticWatermark(
            l=config.l,
            transition_matrix=config.transition_matrix,
            initial_state=config.initial_state,
            delta=config.delta,
            tokenizer=tokenizer,
            patterns=config.patterns,
            pattern_length=config.pattern_length,
        )
    elif watermark_type.value == "Unigram":
        return UnigramWatermark(
            delta=config.delta,
            gamma=config.gamma,
            tokenizer=tokenizer,
            seed=config.seed,
        )
    else:
        raise NotImplementedError(
            f"Watermark type {watermark_type} is not implemented."
        )
