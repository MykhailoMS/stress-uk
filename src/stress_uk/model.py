"""Char-level BiLSTM: для кожної букви слова передбачає логіт "ця буква
наголошена". Передбачення маскується лише на голосні (приголосна ніколи не
несе наголосу) — і на вхід (через ембединг), і явно при інференсі/лоссі."""
import torch
from torch import nn

from .constants import UK_VOWELS

PAD_IDX = 0


class StressTagger(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, char_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """char_ids: (B, T) -> логіти (B, T), padding-позиції зайві (маскуються лоссом)."""
        emb = self.emb_dropout(self.embedding(char_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=char_ids.size(1)
        )
        out = self.out_dropout(out)
        return self.head(out).squeeze(-1)


def vowel_mask(words: list[str], max_len: int) -> torch.Tensor:
    """(B, T) булева маска: True там, де символ слова — голосна."""
    mask = torch.zeros(len(words), max_len, dtype=torch.bool)
    for i, w in enumerate(words):
        for j, ch in enumerate(w):
            if ch.lower() in UK_VOWELS:
                mask[i, j] = True
    return mask


def hyphen_segments(word: str) -> list[tuple[int, int]]:
    """'кас-салон' -> [(0, 3), (4, 9)] (діапазони індексів без самого дефіса)."""
    segments = []
    start = 0
    for i, ch in enumerate(word):
        if ch == "-":
            if i > start:
                segments.append((start, i))
            start = i + 1
    if start < len(word):
        segments.append((start, len(word)))
    return segments


def decode_mask(word: str, probs: list[float]) -> list[int]:
    """Єдина точка, де ймовірності моделі перетворюються на бінарну маску
    наголосу - і на інференсі, і при виборі найкращого чекпоінта під час
    навчання, щоб ці два місця фізично не могли розійтися.

    Дефісні композити ('маркетинг-план') можуть мати 0, 1 чи 2 наголоси на
    різні частини - не можна силою ставити рівно один на все слово. Для
    звичайного (без дефіса) слова - рівно один наголос, на найвірогіднішу
    голосну."""
    segments = hyphen_segments(word)
    single_segment = len(segments) <= 1
    mask = [0] * len(word)
    for start, end in segments:
        vowel_positions = [i for i in range(start, end) if word[i].lower() in UK_VOWELS]
        if not vowel_positions:
            continue
        best = max(vowel_positions, key=lambda i: probs[i])
        if single_segment or probs[best] > 0.5:
            mask[best] = 1
    return mask
