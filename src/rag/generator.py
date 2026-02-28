from __future__ import annotations

import re
from typing import Any


DOC_SPAM_RE = re.compile(r"(?:\s*\[Doc\s*\d+\]\s*){4,}", re.IGNORECASE)
DOC_TAG_RE = re.compile(r"\[\s*Doc\s*([0-9\uFF10-\uFF19]+)\s*\]", re.IGNORECASE)
DOC_LINK_RE = re.compile(r"\[(Doc\s*[0-9\uFF10-\uFF19]+)\]\s*\([^)]*\)", re.IGNORECASE)
DOC_ANY_TAG_RE = re.compile(r"\[([^\]]*Doc[^\]]*)\]", re.IGNORECASE)
CHAT_MARKER_RE = re.compile(r"(?:Human|User|Assistant|System)\s*[:：]", re.IGNORECASE)
TRAILING_CHAT_WORD_RE = re.compile(r"(?:Human|User|Assistant|System)\s*$", re.IGNORECASE)
TAIL_INCOMPLETE_DOC_RE = re.compile(r"(?:\[\s*Doc\s*[0-9\uFF10-\uFF19]*\s*$|\[[^\]]*$)", re.IGNORECASE)
SIGNATURE_NOISE_RE = re.compile(r"\(\s*由\s*@?[A-Za-z0-9_]+\s*科学校验\s*\)", re.IGNORECASE)
TRAILING_FILLER_RE = re.compile(r"(?:您还有其他问题吗[？?]?|以上内容仅供参考[。.]?)\s*$")
PROMPT_RESIDUE_RES = [
    re.compile(r"(?:^|[\s(（])(?:请总结(?:这篇文献)?|请用中文回答|请总结回答以上问题)[:：]?", re.IGNORECASE),
    re.compile(r"(?:^|[\s(（])(?:Please summarize|Answer in Chinese|Summarize the paper)[:：]?", re.IGNORECASE),
]
FULLWIDTH_TO_ASCII = str.maketrans({chr(0xFF10 + i): str(i) for i in range(10)})
SUBSCRIPT_TO_ASCII = str.maketrans({chr(0x2080 + i): str(i) for i in range(10)})


class Generator:
    def __init__(self, model_name_or_path: str, device: str = "cuda", adapter_name_or_path: str = ""):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.adapter_name_or_path = adapter_name_or_path
        self.backend = "template"
        self.tokenizer: Any = None
        self.model: Any = None

        try:
            import warnings
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            warnings.filterwarnings("ignore", message=r"Special tokens have been added in the vocabulary.*")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if hasattr(self.model, "generation_config"):
                # Keep deterministic defaults clean when do_sample=False to avoid noisy warnings.
                self.model.generation_config.temperature = 1.0
                self.model.generation_config.top_p = 1.0
                self.model.generation_config.top_k = 50
            if adapter_name_or_path:
                self.model = PeftModel.from_pretrained(self.model, adapter_name_or_path, is_trainable=False)
            self.backend = "transformers"
        except Exception:
            self.backend = "template"

    @staticmethod
    def _is_cjk(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))

    @staticmethod
    def cleanup_generation(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return ""

        marker = CHAT_MARKER_RE.search(out)
        if marker:
            out = out[: marker.start()].rstrip()

        out = DOC_LINK_RE.sub(r"[\1]", out)

        def _norm_doc_tag(m: re.Match[str]) -> str:
            num = m.group(1).translate(FULLWIDTH_TO_ASCII)
            num = "".join(ch for ch in num if ch.isdigit())
            return f"[Doc {num}]" if num else "[Doc 1]"

        def _norm_doc_tag_loose(m: re.Match[str]) -> str:
            raw = m.group(1).translate(FULLWIDTH_TO_ASCII).translate(SUBSCRIPT_TO_ASCII)
            nums = re.findall(r"\d+", raw)
            if nums:
                return f"[Doc {nums[0]}]"
            return ""

        out = DOC_TAG_RE.sub(_norm_doc_tag, out)
        out = DOC_ANY_TAG_RE.sub(_norm_doc_tag_loose, out)
        out = DOC_SPAM_RE.sub(" [Doc 1] [Doc 2] [Doc 3] ", out)
        out = SIGNATURE_NOISE_RE.sub("", out)

        # Strip instruction residue that occasionally leaks from SFT/DPO traces.
        for pat in PROMPT_RESIDUE_RES:
            while True:
                m = pat.search(out)
                if not m:
                    break
                # Mid/late instruction residue: keep prefix and truncate tail.
                if m.start() >= 20:
                    out = out[: m.start()].rstrip()
                    break
                # Early residue: remove marker itself and keep useful suffix.
                out = (out[: m.start()] + " " + out[m.end() :]).strip()

        # Remove repeated lines while preserving first occurrence order.
        raw_lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if raw_lines:
            seen: set[str] = set()
            deduped: list[str] = []
            for ln in raw_lines:
                key = re.sub(r"\s+", " ", ln)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(ln)
            out = " ".join(deduped)

        out = TAIL_INCOMPLETE_DOC_RE.sub("", out).strip()
        out = TRAILING_CHAT_WORD_RE.sub("", out).strip()
        out = TRAILING_FILLER_RE.sub("", out).strip()
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def translate_query_for_retrieval(self, query: str) -> str:
        if not query or not self._is_cjk(query) or self.backend != "transformers":
            return query

        import torch

        prompt = (
            "Translate the following Chinese medical question into concise English for retrieval.\n"
            "Output only the English translation.\n\n"
            f"Question:\n{query}\n\n"
            "English:\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            try:
                model_device = self.model.device
            except Exception:
                model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

        input_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        translated = self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
        return translated or query

    def generate(
        self,
        prompt: str,
        docs: list[dict],
        max_new_tokens: int = 192,
        temperature: float = 0.2,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 4,
    ) -> str:
        if self.backend == "transformers":
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                try:
                    model_device = self.model.device
                except Exception:
                    model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

            input_len = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "repetition_penalty": max(1.0, float(repetition_penalty)),
                    "no_repeat_ngram_size": max(0, int(no_repeat_ngram_size)),
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                output = self.model.generate(**inputs, **gen_kwargs)
            text = self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
            return self.cleanup_generation(text)

        if not docs:
            return "Insufficient evidence to answer safely."

        first = docs[0]["text"][:400]
        return f"Based on retrieved evidence, key finding: {first} [Doc 1]"
