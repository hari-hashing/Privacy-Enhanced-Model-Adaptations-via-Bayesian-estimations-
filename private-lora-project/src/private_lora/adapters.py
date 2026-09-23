"""LoRA adapters that keep the trainable surface small for DP-SGD."""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model

def add_dp_lora(model, rank: int, alpha: float):
    """Attach PEFT LoRA modules; DP-SGD is applied by the training loop."""

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_lin", "v_lin", "query", "value"],
    )
    return get_peft_model(model, config)
