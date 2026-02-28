#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
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
    load_adapter,
    load_base_model,
    load_tokenizer,
    save_training_state,
    set_global_seed,
    to_device,
    trainable_parameter_stats,
)


@dataclass
class PairExample:
    chosen_input_ids: list[int]
    chosen_labels: list[int]
    rejected_input_ids: list[int]
    rejected_labels: list[int]


class DPOPairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        max_seq_length: int,
        max_prompt_length: int,
        max_target_length: int,
    ):
        self.examples: list[PairExample] = []
        eos_id = tokenizer.eos_token_id

        for row in rows:
            prompt = row.get("prompt", "").strip()
            chosen = row.get("chosen", "").strip()
            rejected = row.get("rejected", "").strip()
            if not prompt or not chosen or not rejected:
                continue

            prompt_ids = tokenizer(
                prompt,
                truncation=True,
                max_length=max_prompt_length,
                add_special_tokens=False,
            )["input_ids"]
            if not prompt_ids:
                continue

            chosen_ids = tokenizer(
                chosen,
                truncation=True,
                max_length=max_target_length,
                add_special_tokens=False,
            )["input_ids"]
            rejected_ids = tokenizer(
                rejected,
                truncation=True,
                max_length=max_target_length,
                add_special_tokens=False,
            )["input_ids"]

            if eos_id is not None:
                chosen_ids = chosen_ids + [eos_id]
                rejected_ids = rejected_ids + [eos_id]

            ch_input = (prompt_ids + chosen_ids)[:max_seq_length]
            rj_input = (prompt_ids + rejected_ids)[:max_seq_length]
            ch_labels = ([-100] * len(prompt_ids) + chosen_ids)[:max_seq_length]
            rj_labels = ([-100] * len(prompt_ids) + rejected_ids)[:max_seq_length]

            if all(x == -100 for x in ch_labels) or all(x == -100 for x in rj_labels):
                continue

            self.examples.append(
                PairExample(
                    chosen_input_ids=ch_input,
                    chosen_labels=ch_labels,
                    rejected_input_ids=rj_input,
                    rejected_labels=rj_labels,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        ex = self.examples[idx]
        return {
            "chosen_input_ids": ex.chosen_input_ids,
            "chosen_labels": ex.chosen_labels,
            "rejected_input_ids": ex.rejected_input_ids,
            "rejected_labels": ex.rejected_labels,
        }


class DPOCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    @staticmethod
    def _pad(seqs: list[list[int]], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
        bs = len(seqs)
        max_len = max(len(s) for s in seqs)
        out = torch.full((bs, max_len), pad_value, dtype=torch.long)
        mask = torch.zeros((bs, max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            l = len(s)
            out[i, :l] = torch.tensor(s, dtype=torch.long)
            mask[i, :l] = 1
        return out, mask

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        ch_ids, ch_mask = self._pad([x["chosen_input_ids"] for x in features], self.pad_token_id)
        ch_labels, _ = self._pad([x["chosen_labels"] for x in features], -100)
        rj_ids, rj_mask = self._pad([x["rejected_input_ids"] for x in features], self.pad_token_id)
        rj_labels, _ = self._pad([x["rejected_labels"] for x in features], -100)

        return {
            "chosen_input_ids": ch_ids,
            "chosen_attention_mask": ch_mask,
            "chosen_labels": ch_labels,
            "rejected_input_ids": rj_ids,
            "rejected_attention_mask": rj_mask,
            "rejected_labels": rj_labels,
        }


def sequence_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]

    target_ids = input_ids[:, 1:]
    shift_labels = labels[:, 1:]
    label_mask = (shift_labels != -100).float()

    log_probs = torch.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    token_logps = token_logps * label_mask
    seq_logps = token_logps.sum(dim=-1) / label_mask.sum(dim=-1).clamp(min=1.0)
    return seq_logps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "dpo.yaml"))
    parser.add_argument("--data", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--adapter", default="")
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

    data_path = Path(args.data or cfg.get("data_path", ROOT / "data" / "preference" / "dpo_train.jsonl"))
    model_name_or_path = args.model or cfg.get("model_name_or_path", "Qwen/Qwen2-7B-Instruct")
    adapter_path = args.adapter or str(cfg.get("adapter_name_or_path", "")).strip()
    output_dir = Path(args.output_dir or cfg.get("output_dir", ROOT / "outputs" / "raft-dpo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Missing data file: {data_path}")
        return 1

    rows = list(read_jsonl(data_path))
    max_samples = int(cfg.get("max_samples", 0)) if args.max_samples < 0 else args.max_samples
    if max_samples > 0:
        rows = rows[:max_samples]
    if not rows:
        print("No DPO samples found.")
        return 1

    tokenizer = load_tokenizer(model_name_or_path)
    model = load_base_model(
        model_name_or_path=model_name_or_path,
        use_qlora=bool(cfg.get("use_qlora", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
    )

    if adapter_path:
        model = load_adapter(model, adapter_path)
    else:
        model = attach_lora(model, cfg)

    # Required for stable backward with k-bit + gradient checkpointing.
    if bool(cfg.get("gradient_checkpointing", True)) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable, total = trainable_parameter_stats(model)
    print(f"[INFO] trainable params: {trainable} / {total} ({100.0 * trainable / max(total, 1):.4f}%)")

    reference_free = bool(cfg.get("reference_free", True))
    ref_model = None
    if not reference_free:
        ref_model = load_base_model(
            model_name_or_path=model_name_or_path,
            use_qlora=bool(cfg.get("use_qlora", True)),
            gradient_checkpointing=False,
        )
        if adapter_path:
            ref_model = load_adapter(ref_model, adapter_path)
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()

    dataset = DPOPairDataset(
        rows=rows,
        tokenizer=tokenizer,
        max_seq_length=int(cfg.get("max_seq_length", 1024)),
        max_prompt_length=int(cfg.get("max_prompt_length", 512)),
        max_target_length=int(cfg.get("max_target_length", 384)),
    )
    if len(dataset) == 0:
        print("No valid DPO samples after tokenization.")
        return 1

    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=DPOCollator(pad_token_id=tokenizer.pad_token_id),
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
        lr=float(cfg.get("learning_rate", 1e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    warmup_steps = int(total_updates * float(cfg.get("warmup_ratio", 0.03)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    beta = float(cfg.get("beta", 0.1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    device = get_model_device(model)
    if ref_model is not None:
        ref_model.to(device)

    print(
        f"[INFO] samples={len(dataset)} batch={dataloader.batch_size} grad_accum={grad_accum} "
        f"epochs={num_epochs} total_updates={total_updates} reference_free={reference_free}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    running_loss = 0.0
    running_acc = 0.0

    for epoch in range(num_epochs):
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"DPO Epoch {epoch+1}/{num_epochs}",
            unit="batch",
        )
        for step_idx, batch in pbar:
            ch_batch = to_device(
                {
                    "input_ids": batch["chosen_input_ids"],
                    "attention_mask": batch["chosen_attention_mask"],
                    "labels": batch["chosen_labels"],
                },
                device,
            )
            rj_batch = to_device(
                {
                    "input_ids": batch["rejected_input_ids"],
                    "attention_mask": batch["rejected_attention_mask"],
                    "labels": batch["rejected_labels"],
                },
                device,
            )

            chosen_logps = sequence_logps(
                model,
                input_ids=ch_batch["input_ids"],
                attention_mask=ch_batch["attention_mask"],
                labels=ch_batch["labels"],
            )
            rejected_logps = sequence_logps(
                model,
                input_ids=rj_batch["input_ids"],
                attention_mask=rj_batch["attention_mask"],
                labels=rj_batch["labels"],
            )

            pi_logratios = chosen_logps - rejected_logps

            if ref_model is not None:
                with torch.no_grad():
                    ref_chosen_logps = sequence_logps(
                        ref_model,
                        input_ids=ch_batch["input_ids"],
                        attention_mask=ch_batch["attention_mask"],
                        labels=ch_batch["labels"],
                    )
                    ref_rejected_logps = sequence_logps(
                        ref_model,
                        input_ids=rj_batch["input_ids"],
                        attention_mask=rj_batch["attention_mask"],
                        labels=rj_batch["labels"],
                    )
                ref_logratios = ref_chosen_logps - ref_rejected_logps
            else:
                ref_logratios = torch.zeros_like(pi_logratios)

            logits = beta * (pi_logratios - ref_logratios)
            loss = -F.logsigmoid(logits).mean()
            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at epoch={epoch+1} step_idx={step_idx+1}; skip batch")
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({"step": global_step, "loss": "nan-skip"})
                continue
            (loss / grad_accum).backward()

            batch_acc = (pi_logratios > 0).float().mean().item()
            running_loss += float(loss.item())

            should_step = ((step_idx + 1) % grad_accum == 0) or ((step_idx + 1) == len(dataloader))
            if not should_step:
                pbar.set_postfix(
                    {
                        "step": global_step,
                        "loss": f"{float(loss.item()):.4f}",
                        "acc": f"{batch_acc:.4f}",
                    }
                )
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_acc += batch_acc
            pbar.set_postfix(
                {
                    "u_step": global_step,
                    "loss": f"{float(loss.item()):.4f}",
                    "acc": f"{batch_acc:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if global_step % logging_steps == 0:
                print(
                    f"[DPO] epoch={epoch+1}/{num_epochs} step={global_step}/{total_updates} "
                    f"loss={running_loss / logging_steps:.4f} "
                    f"acc={running_acc / logging_steps:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )
                running_loss = 0.0
                running_acc = 0.0

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
                        "beta": beta,
                        "reference_free": reference_free,
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
            "adapter_name_or_path": adapter_path,
            "reference_free": reference_free,
        },
    )
    print(f"[OK] DPO training complete. Final adapter: {output_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
