#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=r"Special tokens have been added in the vocabulary.*")
warnings.filterwarnings(
    "ignore",
    message=r"Could not find a config file in .* will assume that the vocabulary was not modified\.",
)

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
    trainable_parameter_stats,
)
from src.train.rlvr_reward import compute_rlvr_reward


EN_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")
DOC_RE = re.compile(r"(?:Document\s*(\d+)\s*:|\[Doc\s*(\d+)\])", re.IGNORECASE)
CIT_RE = re.compile(r"\[Doc\s*(\d+)\]", re.IGNORECASE)
DOC_MARKER_RE = re.compile(r"Document\s*(\d+)\s*:", re.IGNORECASE)
ANSWER_SPLIT_RE = re.compile(r"\$Answer\$\s*[:：]?\s*", re.IGNORECASE)


def token_set(text: str) -> set[str]:
    en = {w.lower() for w in EN_TOKEN_RE.findall(text or "")}
    cjk = {ch for ch in (text or "") if "\u4e00" <= ch <= "\u9fff"}
    return en | cjk


def extract_answer(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if "$Answer$" not in raw:
        return raw
    parts = ANSWER_SPLIT_RE.split(raw, maxsplit=1)
    if len(parts) >= 2 and parts[1].strip():
        return parts[1].strip()
    return raw


def ngram_repeat_ratio(text: str, n: int = 3) -> float:
    toks = [x for x in (text or "").lower().split() if x]
    if len(toks) < n + 2:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def extract_prompt_doc_ids(prompt: str) -> set[str]:
    ids: set[str] = set()
    for a, b in DOC_RE.findall(prompt or ""):
        if a:
            ids.add(a)
        if b:
            ids.add(b)
    return ids


def build_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = DOC_MARKER_RE.sub(r"[Doc \1]", input_text or "").strip()
    return (
        "You are a medical assistant. Use only provided evidence and cite with [Doc i].\n\n"
        f"Question:\n{instruction}\n\n"
        f"Retrieved Evidence:\n{input_text}\n\n"
        "Answer:\n"
    )


def score_response(prompt: str, response: str, weights: dict) -> tuple[float, dict[str, float]]:
    evidence_tokens = token_set(prompt.split("Answer:", 1)[0])
    response_clean = CIT_RE.sub(" ", response or "").strip()
    response_tokens = token_set(response_clean)

    overlap = 0.0
    if response_tokens:
        overlap = len(response_tokens & evidence_tokens) / max(1, len(response_tokens))

    groundedness = min(1.0, overlap * 3.0)

    valid_doc_ids = extract_prompt_doc_ids(prompt)
    cited_list = CIT_RE.findall(response or "")
    cited_unique = set(cited_list)
    valid_cited = cited_unique & valid_doc_ids
    if cited_unique:
        citation_precision = len(valid_cited) / max(1, len(cited_unique))
        citation_recall = len(valid_cited) / max(1, len(valid_doc_ids))
        citation = 0.7 * citation_precision + 0.3 * citation_recall
    else:
        citation = 0.0

    cite_repeat = 0.0
    if cited_list:
        cite_repeat = (len(cited_list) - len(cited_unique)) / len(cited_list)
    text_repeat = ngram_repeat_ratio(response_clean, n=3)
    repetition_penalty = min(1.0, 0.6 * cite_repeat + 0.4 * text_repeat)

    content_ok = len(response_clean) >= 20
    evidence_hit = 1.0 if (citation > 0.2 and groundedness >= 0.08 and content_ok) else 0.0

    low = response.lower()
    refusal = any(
        x in low
        for x in (
            "insufficient evidence",
            "cannot answer safely",
            "not enough evidence",
            "consult a doctor",
            "证据不足",
            "无法安全回答",
        )
    )
    if refusal:
        safety = 1.0
    elif citation > 0.0 and groundedness > 0.15:
        safety = 0.9
    elif len(response.strip()) < 10:
        safety = 0.2
    else:
        safety = 0.5

    reward = compute_rlvr_reward(
        evidence_hit=evidence_hit,
        groundedness=groundedness,
        citation=citation,
        safety=safety,
        repetition_penalty=repetition_penalty,
        weights=weights,
    )
    components = {
        "evidence_hit": evidence_hit,
        "groundedness": groundedness,
        "citation": citation,
        "safety": safety,
        "repetition_penalty": repetition_penalty,
    }
    return float(reward), components


class PromptDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.samples: list[dict] = []
        for row in rows:
            instruction = row.get("instruction", "").strip()
            input_text = row.get("input", "").strip()
            if not instruction:
                continue
            self.samples.append(
                {
                    "prompt": build_prompt(instruction, input_text),
                    "raw_instruction": instruction,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.samples[idx]


class PromptCollator:
    def __call__(self, features: list[dict[str, str]]) -> dict[str, list[str]]:
        return {
            "prompts": [x["prompt"] for x in features],
            "raw_instruction": [x["raw_instruction"] for x in features],
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
    parser.add_argument("--config", default=str(ROOT / "configs" / "rlvr.yaml"))
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

    data_path = Path(args.data or cfg.get("data_path", ROOT / "data" / "synthetic" / "sft_train.jsonl"))
    model_name_or_path = args.model or cfg.get("model_name_or_path", "Qwen/Qwen2-7B-Instruct")
    adapter_path = args.adapter or str(cfg.get("adapter_name_or_path", "")).strip()
    output_dir = Path(args.output_dir or cfg.get("output_dir", ROOT / "outputs" / "raft-rlvr"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Missing data file: {data_path}")
        return 1

    rows = list(read_jsonl(data_path))
    max_samples = int(cfg.get("max_samples", 0)) if args.max_samples < 0 else args.max_samples
    if max_samples > 0:
        rows = rows[:max_samples]
    if not rows:
        print("No RLVR samples found.")
        return 1

    tokenizer = load_tokenizer(model_name_or_path, padding_side="left")
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

    kl_coef = float(cfg.get("kl_coef", 0.0))
    ref_model = None
    if kl_coef > 0:
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

    dataset = PromptDataset(rows)
    if len(dataset) == 0:
        print("No valid RLVR prompts.")
        return 1

    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=PromptCollator(),
        pin_memory=torch.cuda.is_available(),
    )

    grad_accum = int(cfg.get("gradient_accumulation_steps", 8))
    num_epochs = int(cfg.get("num_train_epochs", 1))
    max_steps = int(cfg.get("max_steps", 0))
    logging_steps = int(cfg.get("logging_steps", 10))
    save_steps = int(cfg.get("save_steps", 100))

    updates_per_epoch = math.ceil(len(dataloader) / grad_accum)
    total_updates = max_steps if max_steps > 0 else updates_per_epoch * num_epochs

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=float(cfg.get("learning_rate", 5e-6)),
        weight_decay=0.0,
    )
    warmup_steps = int(total_updates * 0.03)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    device = get_model_device(model)
    if ref_model is not None:
        ref_model.to(device)

    max_prompt_len = int(cfg.get("max_prompt_len", 512))
    max_new_tokens = int(cfg.get("max_new_tokens", 192))
    temperature = float(cfg.get("temperature", 0.8))
    top_p = float(cfg.get("top_p", 0.9))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    weights = cfg.get("reward_weights", None)

    print(
        f"[INFO] samples={len(dataset)} batch={dataloader.batch_size} grad_accum={grad_accum} "
        f"epochs={num_epochs} total_updates={total_updates} algorithm={cfg.get('algorithm', 'reinforce')}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    baseline = 0.0
    running_loss_weighted = 0.0
    running_reward_sum = 0.0
    running_eh_sum = 0.0
    running_gr_sum = 0.0
    running_ci_sum = 0.0
    running_sa_sum = 0.0
    running_rp_sum = 0.0
    running_sample_count = 0
    oom_batches = 0

    for epoch in range(num_epochs):
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"RLVR Epoch {epoch+1}/{num_epochs}",
            unit="batch",
        )
        for step_idx, batch in pbar:
            prompts = batch["prompts"]
            tokenized = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            input_len = int(tokenized["input_ids"].shape[1])

            was_gc_enabled = bool(getattr(model, "is_gradient_checkpointing", False))
            if was_gc_enabled and hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()

            prev_mode_training = model.training
            model.eval()
            use_cache_prev = bool(getattr(model.config, "use_cache", False))
            model.config.use_cache = True
            with torch.no_grad():
                generated = model.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            model.config.use_cache = use_cache_prev
            if was_gc_enabled and hasattr(model, "gradient_checkpointing_enable"):
                try:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    model.gradient_checkpointing_enable()
            if prev_mode_training:
                model.train()

            responses: list[str] = []
            for i in range(generated.size(0)):
                start = input_len
                response_ids = generated[i, start:]
                responses.append(tokenizer.decode(response_ids, skip_special_tokens=True).strip())

            rewards = []
            components = []
            for p, r in zip(prompts, responses):
                reward, comp = score_response(p, r, weights=weights)
                rewards.append(reward)
                components.append(comp)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            baseline = 0.9 * baseline + 0.1 * float(rewards_t.mean().item())
            advantages = rewards_t - baseline

            if tokenizer.pad_token_id is None:
                attn_gen = torch.ones_like(generated, dtype=torch.long, device=device)
            else:
                attn_gen = (generated != tokenizer.pad_token_id).long()

            labels = generated.clone()
            labels[:, :input_len] = -100
            labels = labels.masked_fill(attn_gen == 0, -100)

            try:
                seq_logps = sequence_logps(model, input_ids=generated, attention_mask=attn_gen, labels=labels)
                policy_loss = -(advantages.detach() * seq_logps).mean()

                if ref_model is not None:
                    with torch.no_grad():
                        ref_logps = sequence_logps(ref_model, input_ids=generated, attention_mask=attn_gen, labels=labels)
                    kl = (seq_logps - ref_logps).mean()
                    loss = policy_loss + kl_coef * kl
                else:
                    loss = policy_loss
            except torch.cuda.OutOfMemoryError:
                oom_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[WARN] OOM at epoch={epoch+1} step_idx={step_idx+1}; skip batch (oom_batches={oom_batches})")
                pbar.set_postfix({"step": global_step, "loss": "oom-skip"})
                continue

            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at epoch={epoch+1} step_idx={step_idx+1}; skip batch")
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({"step": global_step, "loss": "nan-skip"})
                continue

            (loss / grad_accum).backward()

            batch_size = max(1, len(components))
            running_loss_weighted += float(loss.item()) * batch_size
            running_reward_sum += float(rewards_t.sum().item())
            running_eh_sum += sum(c["evidence_hit"] for c in components)
            running_gr_sum += sum(c["groundedness"] for c in components)
            running_ci_sum += sum(c["citation"] for c in components)
            running_sa_sum += sum(c["safety"] for c in components)
            running_rp_sum += sum(c["repetition_penalty"] for c in components)
            running_sample_count += batch_size

            should_step = ((step_idx + 1) % grad_accum == 0) or ((step_idx + 1) == len(dataloader))
            if not should_step:
                batch_reward = float(rewards_t.mean().item()) if rewards else 0.0
                pbar.set_postfix(
                    {
                        "step": global_step,
                        "loss": f"{float(loss.item()):.4f}",
                        "reward": f"{batch_reward:.4f}",
                    }
                )
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            batch_reward = float(rewards_t.mean().item()) if rewards else 0.0
            pbar.set_postfix(
                {
                    "u_step": global_step,
                    "loss": f"{float(loss.item()):.4f}",
                    "reward": f"{batch_reward:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if global_step % logging_steps == 0:
                denom = max(1, running_sample_count)
                print(
                    f"[RLVR] epoch={epoch+1}/{num_epochs} step={global_step}/{total_updates} "
                    f"loss={running_loss_weighted / denom:.4f} reward={running_reward_sum / denom:.4f} "
                    f"eh={running_eh_sum / denom:.3f} gr={running_gr_sum / denom:.3f} "
                    f"ci={running_ci_sum / denom:.3f} sa={running_sa_sum / denom:.3f} "
                    f"rp={running_rp_sum / denom:.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
                running_loss_weighted = 0.0
                running_reward_sum = 0.0
                running_eh_sum = 0.0
                running_gr_sum = 0.0
                running_ci_sum = 0.0
                running_sa_sum = 0.0
                running_rp_sum = 0.0
                running_sample_count = 0

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
                        "baseline": baseline,
                        "oom_batches": oom_batches,
                        "algorithm": cfg.get("algorithm", "reinforce"),
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
            "oom_batches": oom_batches,
            "algorithm": cfg.get("algorithm", "reinforce"),
        },
    )

    print(f"[OK] RLVR training complete. Final adapter: {output_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
