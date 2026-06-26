"""Допоміжні функції для роботи з прикладами train.jsonl/val.jsonl
(plain-слово + бінарна маска наголосу), словником символів і кешем
найчастіших варіантів наголосу на слово."""
import json
from pathlib import Path

from .constants import ACUTE

PAD, UNK = "<pad>", "<unk>"


def mask_to_word(plain: str, mask: list[int]) -> str:
    """('рука', [0,0,0,1]) -> 'рука́'."""
    return "".join(ch + ACUTE if m else ch for ch, m in zip(plain, mask))


def strip_accents(word: str) -> tuple[str, list[int]]:
    """'рука́' -> ('рука', [0,0,0,1])."""
    plain, mask = [], []
    i, n = 0, len(word)
    while i < n:
        stressed = i + 1 < n and word[i + 1] == ACUTE
        plain.append(word[i])
        mask.append(1 if stressed else 0)
        i += 2 if stressed else 1
    return "".join(plain), mask


def load_vocab(data_dir: Path) -> dict:
    return json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))


def load_word_patterns(data_dir: Path) -> dict[str, dict]:
    """{слово: {"best": [...], "all": [[...], ...]}} - усі спостережені
    варіанти наголосу на слово (best - найчастіший, для лагідної оцінки -
    "all"). Не обов'язковий файл - порожній кеш, якщо відсутній."""
    path = data_dir / "word_patterns.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
