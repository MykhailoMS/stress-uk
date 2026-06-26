"""Цикл навчання StressTagger на train.jsonl, добір кращого чекпоінта по
word-level accuracy на val.jsonl.

Запуск (після `pip install stress-uk` або з src/ у режимі розробки):
    python -m stress_uk.train
    python -m stress_uk.train --full-data --epochs 25 --emb-dim 128 --hidden-dim 256 --num-layers 2

Пакет несе data/train.jsonl і data/val.jsonl - невелику ілюстративну
вибірку, не повний навчальний набір. Заміни їх своїми даними того самого
формату (один JSON-обʼєкт на рядок: {"word": "рука", "stress": [0,0,0,1]})
і поповни data/vocab.json/data/word_patterns.json відповідно, якщо
додаєш нові символи чи слова - або передай свій --data-dir.

Чекпоінти зберігаються у ./models/ (поточна робоча директорія, НЕ всередину
встановленого пакета) - передай --checkpoint, щоб вказати інший шлях.

--full-data: окремий режим для ФІНАЛЬНОГО прод-чекпоінта (не для звітності
й порівняння - для цього лишається звичайний запуск). Згортає val.jsonl у
train, тож модель бачить усі приклади, а не лише train.jsonl. Без
легітимного held-out сету для early stopping - тренуємось фіксовану
кількість епох (передай число з найкращої епохи звичайного прогону) і
зберігаємо чекпоінт ПІСЛЯ останньої епохи, в ОКРЕМИЙ файл
(models/stress_tagger_full.pt), щоб не перезаписати вже пробенчмарчений
основний чекпоінт."""
import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data_utils import load_vocab, load_word_patterns, mask_to_word
from .model import PAD_IDX, StressTagger, decode_mask, vowel_mask

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path.cwd() / "models"


class StressDataset(Dataset):
    def __init__(self, paths: Path | list[Path]):
        if isinstance(paths, Path):
            paths = [paths]
        self.examples = []
        for path in paths:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    ex = json.loads(line)
                    self.examples.append((ex["word"], ex["stress"]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def collate(batch, vocab: dict):
    """Функція модульного рівня (не closure) - щоб DataLoader(num_workers>0)
    міг її пікл'ювати для дочірніх процесів на spawn-платформах (Windows)."""
    words = [w for w, _ in batch]
    stresses = [s for _, s in batch]
    lengths = torch.tensor([len(w) for w in words], dtype=torch.long)
    max_len = int(lengths.max())

    char_ids = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    target = torch.zeros((len(batch), max_len), dtype=torch.float)
    for i, (w, s) in enumerate(zip(words, stresses)):
        for j, ch in enumerate(w):
            char_ids[i, j] = vocab.get(ch, vocab["<unk>"])
        for j, v in enumerate(s):
            target[i, j] = v

    v_mask = vowel_mask(words, max_len)
    return char_ids, lengths, target, v_mask, words


def make_collate(vocab: dict):
    return partial(collate, vocab=vocab)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             word_patterns: dict[str, dict]) -> float:
    """word-level exact-match через ту саму decode_mask(), що й у проді.
    Зараховуємо збіг з БУДЬ-ЯКИМ реально спостереженим варіантом слова
    (word_patterns.json), а не лише з тим, що випадково потрапив у цей
    рядок val.jsonl - інакше для слів зі справжньою граматичною омонімією
    відбір чекпоінта несправедливо штрафував би модель за вибір іншого,
    теж правильного варіанта."""
    model.eval()
    correct = 0
    total = 0
    for char_ids, lengths, _, _, words in loader:
        char_ids, lengths = char_ids.to(device), lengths.to(device)
        probs = torch.sigmoid(model(char_ids, lengths)).cpu()
        for i, word in enumerate(words):
            n = len(word)
            pred_mask = decode_mask(word, probs[i, :n].tolist())
            pred_word = mask_to_word(word, pred_mask)
            valid = {mask_to_word(word, p) for p in word_patterns[word]["all"]}
            correct += int(pred_word in valid)
        total += len(words)
    return correct / total if total else 0.0


def train(epochs: int = 40, batch_size: int = 256, lr: float = 1e-3,
          weight_decay: float = 1e-4, dropout: float = 0.3, patience: int = 6,
          device: str | None = None, num_workers: int = 4,
          emb_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2,
          data_dir: Path = DATA_DIR, checkpoint_path: Path | None = None,
          full_data: bool = False) -> None:
    """full_data=True: val.jsonl згортається У train - немає легітимного
    held-out сету, тож early stopping/відбір "найкращого" чекпоінта за
    val_word_acc вимикається повністю. Натомість тренуємось РІВНО `epochs`
    епох (передай те число, де попередній звичайний прогін зупинився
    найкращим) і зберігаємо чекпоінт після останньої епохи."""
    device = torch.device(
        device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    )
    pin_memory = device.type == "cuda"
    default_name = "stress_tagger_full.pt" if full_data else "stress_tagger.pt"
    checkpoint_path = checkpoint_path or (MODELS_DIR / default_name)

    vocab = load_vocab(data_dir)
    word_patterns = load_word_patterns(data_dir)

    train_paths = [data_dir / "train.jsonl"]
    if full_data:
        train_paths += [data_dir / "val.jsonl"]
    train_ds = StressDataset(train_paths)
    val_ds = StressDataset(data_dir / "val.jsonl")
    collate = make_collate(vocab)
    loader_kwargs = dict(
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate, **loader_kwargs)

    arch = dict(emb_dim=emb_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    model = StressTagger(vocab_size=len(vocab), dropout=dropout, **arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        n_examples = 0
        for char_ids, lengths, target, v_mask, _ in train_loader:
            char_ids = char_ids.to(device, non_blocking=pin_memory)
            lengths = lengths.to(device, non_blocking=pin_memory)
            target = target.to(device, non_blocking=pin_memory)
            v_mask = v_mask.to(device, non_blocking=pin_memory)
            logits = model(char_ids, lengths)
            loss = masked_bce(logits, target, v_mask.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * char_ids.size(0)
            n_examples += char_ids.size(0)

        val_acc = evaluate(model, val_loader, device, word_patterns)
        avg_loss = (total_loss / n_examples).item()
        label = "train_word_acc(contaminated)" if full_data else "val_word_acc"
        print(f"epoch {epoch:2d}  train_loss={avg_loss:.4f}  {label}={val_acc:.4f}")

        if full_data:
            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                torch.save(
                    {"model_state": model.state_dict(), "vocab": vocab, "best_val_acc": None,
                     "arch": arch},
                    checkpoint_path,
                )
            continue

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {"model_state": model.state_dict(), "vocab": vocab, "best_val_acc": best_acc,
                 "arch": arch},
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Немає поліпшення {patience} епох підряд - зупиняюсь (early stopping).")
                break

    if full_data:
        print(f"Готово ({epochs} епох, без early stopping, train+val). "
              f"Збережено чекпоінт із найкращим train_loss={best_train_loss:.4f}. -> {checkpoint_path}")
    else:
        print(f"Готово. Найкраща val_word_acc={best_acc:.4f} -> {checkpoint_path}")


def _arch_from_existing_checkpoint() -> dict | None:
    path = MODELS_DIR / "stress_tagger.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("arch")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--full-data", action="store_true",
                    help="згорнути val.jsonl у train, без early stopping - "
                         "фінальний прод-чекпоінт, не для звітності/бенчмарків")
    p.add_argument("--epochs", type=int, default=40,
                    help="для --full-data: рівно стільки епох без early stopping - "
                         "передай число з найкращої епохи попереднього звичайного прогону")
    p.add_argument("--checkpoint", type=Path, default=None,
                    help="перевизначити шлях чекпоінта (дефолт: models/stress_tagger.pt, "
                         "або models/stress_tagger_full.pt для --full-data)")
    p.add_argument("--device", default=None, help="cpu/cuda/mps (дефолт - автовизначення)")
    p.add_argument("--emb-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR,
                    help="директорія з train.jsonl/val.jsonl/vocab.json/word_patterns.json "
                         "(дефолт - приклад-вибірка, що йде з пакетом)")
    args = p.parse_args()

    arch_overrides = {k: v for k, v in (
        ("emb_dim", args.emb_dim), ("hidden_dim", args.hidden_dim), ("num_layers", args.num_layers),
    ) if v is not None}

    if len(arch_overrides) < 3:
        existing_arch = _arch_from_existing_checkpoint()
        if existing_arch:
            print(f"Архітектура з models/stress_tagger.pt: {existing_arch} "
                  f"(передай --emb-dim/--hidden-dim/--num-layers, щоб перевизначити)")
            arch_overrides = {**existing_arch, **arch_overrides}
        else:
            print("models/stress_tagger.pt не знайдено - використовую дефолти train() "
                  "(emb_dim=64, hidden_dim=128, num_layers=2)")

    train(epochs=args.epochs, full_data=args.full_data, data_dir=args.data_dir,
          checkpoint_path=args.checkpoint, device=args.device, **arch_overrides)


if __name__ == "__main__":
    main()
