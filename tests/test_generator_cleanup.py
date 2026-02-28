from __future__ import annotations

from src.rag.generator import Generator


def test_cleanup_removes_chat_marker_tail() -> None:
    text = "有效回答 [Doc 1]\nAssistant: extra"
    out = Generator.cleanup_generation(text)
    assert "Assistant:" not in out
    assert out.endswith("[Doc 1]")


def test_cleanup_normalizes_doc_tag_and_link() -> None:
    text = "结论见 [Doc ２](http://x.com) 与 [doc₃]"
    out = Generator.cleanup_generation(text)
    assert "[Doc 2]" in out
    assert "[Doc 3]" in out


def test_cleanup_cuts_prompt_residue_and_incomplete_doc_tail() -> None:
    text = "这是主体答案，包含关键结论 [Doc 1]。 请总结这篇文献: ... [Doc"
    out = Generator.cleanup_generation(text)
    assert "请总结这篇文献" not in out
    assert not out.endswith("[Doc")
    assert "主体答案" in out


def test_cleanup_removes_early_prompt_residue_marker() -> None:
    text = "研究中选择低剂量方案(1 请用中文回答：这会影响外部有效性 [Doc 1]"
    out = Generator.cleanup_generation(text)
    assert "请用中文回答" not in out
    assert "[Doc 1]" in out
