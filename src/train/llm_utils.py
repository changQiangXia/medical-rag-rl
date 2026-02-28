from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prefer_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def get_default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if prefer_bf16() else torch.float16
    return torch.float32


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def infer_lora_targets(model) -> list[str]:
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = set()
    for name, _ in model.named_modules():
        last = name.split(".")[-1]
        if last in candidates:
            found.add(last)
    return sorted(found) if found else ["q_proj", "v_proj"]


def load_tokenizer(model_name_or_path: str, padding_side: str = "right"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = padding_side
    return tokenizer


def load_base_model(model_name_or_path: str, use_qlora: bool, gradient_checkpointing: bool):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = get_default_dtype()
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    if use_qlora and not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires CUDA GPU. Set use_qlora=false for CPU.")

    if use_qlora:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["quantization_config"] = bnb_cfg

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.config.use_cache = False

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    return model


def attach_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    targets = cfg.get("lora_target_modules")
    if not targets:
        targets = infer_lora_targets(model)

    if cfg.get("use_qlora", True):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        )

    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 64)),
        lora_alpha=int(cfg.get("lora_alpha", 128)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(targets),
    )
    model = get_peft_model(model, lora_config)
    return model


def load_adapter(model, adapter_name_or_path: str):
    from peft import PeftModel

    adapter_path = Path(adapter_name_or_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)


def trainable_parameter_stats(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def save_training_state(path: str | Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def chunked_iterable(items: list, chunk_size: int) -> Iterable[list]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]
