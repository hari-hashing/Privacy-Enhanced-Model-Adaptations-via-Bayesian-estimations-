"""Load Qwen3 in 4-bit and prepare its trainable LoRA adapters."""

from __future__ import annotations

import os
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_qwen_model(
    config: dict[str, Any], *, adapter_path: Path | None = None, trainable: bool = True
) -> tuple[torch.nn.Module, Any]:
    """Load the 4-bit base model and attach or restore its causal-LM adapter."""

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 4-bit loading requires CUDA and bitsandbytes.")
    if find_spec("bitsandbytes") is None:
        raise RuntimeError("Install the project dependencies, including bitsandbytes.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    compute_dtype = getattr(torch, config.get("compute_dtype", "bfloat16"))
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if trainable:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        if adapter_path is None:
            adapter_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(config["rank"]),
                lora_alpha=int(config["alpha"]),
                lora_dropout=float(config.get("lora_dropout", 0.0)),
                target_modules=list(config["target_modules"]),
                bias="none",
            )
            model = get_peft_model(model, adapter_config)
        else:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    elif adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    model.config.use_cache = False
    return model, tokenizer


def wrap_fsdp(model: torch.nn.Module, config: dict[str, Any]) -> torch.nn.Module:
    """Shard Qwen decoder blocks across the launched GPUs with CPU offload."""

    import torch.distributed as distributed
    from torch.distributed.fsdp import (
        CPUOffload,
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    if not distributed.is_initialized():
        raise RuntimeError("Initialize torch.distributed before wrapping the model with FSDP.")
    world_size = distributed.get_world_size()
    expected_world_size = int(config.get("fsdp_world_size", 2))
    if world_size != expected_world_size:
        raise RuntimeError(f"Expected {expected_world_size} FSDP ranks, received {world_size}.")

    base_model = model.get_base_model()
    decoder_layers = getattr(getattr(base_model, "model", None), "layers", None)
    if decoder_layers is None or len(decoder_layers) == 0:
        raise RuntimeError("Could not find Qwen decoder layers for the FSDP auto-wrap policy.")
    layer_type = type(decoder_layers[0])
    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_type})
    dtype = getattr(torch, config.get("compute_dtype", "bfloat16"))
    return FullyShardedDataParallel(
        model,
        auto_wrap_policy=wrap_policy,
        cpu_offload=CPUOffload(offload_params=bool(config.get("cpu_offload", True))),
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=dtype,
        ),
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        limit_all_gathers=True,
    )