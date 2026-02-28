#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.common.config import load_config
from src.common.io_utils import read_jsonl
from src.train.llm_utils import (
    attach_lora,
    get_model_device,
    load_base_model,
    load_tokenizer,
    save_training_state,
    set_global_seed,
    to_device,
    trainable_parameter_stats,
)


PROMPT_TEMPLATE = (
    "You are a medical assistant. Answer using only retrieved evidence.\n"
    "If evidence is insufficient, explicitly state that evidence is insufficient.\n\n"
    "Question:\n{instruction}\n\n"
    "Retrieved Evidence:\n{input}\n\n"
    "Answer:\n"
)
PROMPT_NO_INPUT_TEMPLATE = (
    "You are a medical assistant. Answer using only retrieved evidence.\n"
    "If evidence is insufficient, explicitly state that evidence is insufficient.\n\n"
    "Question:\n{instruction}\n\n"
    "Answer:\n"
)


def format_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return PROMPT_TEMPLATE.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT_TEMPLATE.format(instruction=instruction)


@dataclass
class SFTExample:
    input_ids: list[int]
    labels: list[int]


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_seq_length: int, mask_prompt: bool):
        self.examples: list[SFTExample] = []
        self.skipped_all_masked = 0
        eos = tokenizer.eos_token or ""

        for row in rows:
            instruction = row.get("instruction", "").strip()
            output = row.get("output", "").strip()
            if not instruction or not output:
                continue

            input_text = row.get("input", "")
            prompt = format_prompt(instruction, input_text)
            full_text = prompt + output + eos

            full_ids = tokenizer(
                full_text,
                truncation=True,
                max_length=max_seq_length,
                add_special_tokens=False,
            )["input_ids"]
            if not full_ids:
                continue

            labels = list(full_ids)
            if mask_prompt:
                prompt_ids = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_seq_length,
                    add_special_tokens=False,
                )["input_ids"]
                prompt_len = min(len(prompt_ids), len(labels))
                labels[:prompt_len] = [-100] * prompt_len

            # If prompt consumed full sequence after truncation, there is no target token left.
            if not any(x != -100 for x in labels):
                self.skipped_all_masked += 1
                continue

            self.examples.append(SFTExample(input_ids=full_ids, labels=labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        ex = self.examples[idx]
        return {
            "input_ids": ex.input_ids,
            "labels": ex.labels,
        }


class SFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, f in enumerate(features):
            seq_len = len(f["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(f["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "sft.yaml"))
    parser.add_argument("--data", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def save_checkpoint(model, tokenizer, output_dir: Path, name: str, state: dict) -> None:
    ckpt_dir = output_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    save_training_state(ckpt_dir / "trainer_state.json", state)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    data_path = Path(args.data or cfg.get("data_path", ROOT / "data" / "synthetic" / "sft_train.jsonl"))
    model_name_or_path = args.model or cfg.get("model_name_or_path", "Qwen/Qwen2-7B-Instruct")
    output_dir = Path(args.output_dir or cfg.get("output_dir", ROOT / "outputs" / "raft-sft"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Missing data file: {data_path}")
        return 1

    rows = list(read_jsonl(data_path))
    max_samples = int(cfg.get("max_samples", 0)) if args.max_samples < 0 else args.max_samples
    if max_samples > 0:
        rows = rows[:max_samples]

    if not rows:
        print("No SFT samples found.")
        return 1

    tokenizer = load_tokenizer(model_name_or_path)
    model = load_base_model(
        model_name_or_path=model_name_or_path,
        use_qlora=bool(cfg.get("use_qlora", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
    )
    model = attach_lora(model, cfg)

    trainable, total = trainable_parameter_stats(model)
    print(f"[INFO] trainable params: {trainable} / {total} ({100.0 * trainable / max(total, 1):.4f}%)")

    dataset = SFTDataset(
        rows=rows,
        tokenizer=tokenizer,
        max_seq_length=int(cfg.get("max_seq_length", 1024)),
        mask_prompt=bool(cfg.get("mask_prompt", True)),
    )
    if len(dataset) == 0:
        print("No valid SFT samples after tokenization.")
        return 1
    print(f"[INFO] skipped_all_masked={dataset.skipped_all_masked}")

    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=SFTCollator(pad_token_id=tokenizer.pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )

    grad_accum = int(cfg.get("gradient_accumulation_steps", 16))
    num_epochs = int(cfg.get("num_train_epochs", 1))
    max_steps = int(cfg.get("max_steps", 0))
    logging_steps = int(cfg.get("logging_steps", 10))
    save_steps = int(cfg.get("save_steps", 200))

    updates_per_epoch = math.ceil(len(dataloader) / grad_accum)
    total_updates = max_steps if max_steps > 0 else updates_per_epoch * num_epochs

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    warmup_steps = int(total_updates * float(cfg.get("warmup_ratio", 0.03)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    device = get_model_device(model)
    print(
        f"[INFO] samples={len(dataset)} batch={dataloader.batch_size} grad_accum={grad_accum} "
        f"epochs={num_epochs} total_updates={total_updates}"
    )

    global_step = 0
    running_loss = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"SFT Epoch {epoch+1}/{num_epochs}",
            unit="batch",
        )
        for step_idx, batch in pbar:
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at epoch={epoch+1} step_idx={step_idx+1}; skip batch")
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({"step": global_step, "loss": "nan-skip"})
                continue
            (loss / grad_accum).backward()
            running_loss += float(loss.item())

            should_step = ((step_idx + 1) % grad_accum == 0) or ((step_idx + 1) == len(dataloader))
            if not should_step:
                pbar.set_postfix({"step": global_step, "loss": f"{float(loss.item()):.4f}"})
                continue

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            avg_loss = running_loss / max(1, logging_steps)
            pbar.set_postfix(
                {
                    "u_step": global_step,
                    "loss": f"{float(loss.item()):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if global_step % logging_steps == 0:
                print(
                    f"[SFT] epoch={epoch+1}/{num_epochs} step={global_step}/{total_updates} "
                    f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )
                running_loss = 0.0

            if global_step % save_steps == 0:
                save_checkpoint(
                    model,
                    tokenizer,
                    output_dir,
                    name=f"checkpoint-{global_step}",
                    state={
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "total_updates": total_updates,
                    },
                )

            if global_step >= total_updates:
                break
        pbar.close()

        if global_step >= total_updates:
            break

    save_checkpoint(
        model,
        tokenizer,
        output_dir,
        name="final",
        state={
            "global_step": global_step,
            "total_updates": total_updates,
            "seed": seed,
            "data_path": str(data_path),
            "model_name_or_path": model_name_or_path,
        },
    )
    print(f"[OK] SFT training complete. Final adapter: {output_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
