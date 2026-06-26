"""Інференс: stressify(word) / stressify_text(text).

Кеш точних збігів (O(1), опційний, не обов'язковий файл) + нейромережа як
фолбек/основний шлях для будь-якого слова, бачив модель чи ні.
"""
import re
from functools import lru_cache
from pathlib import Path

import torch

from .data_utils import load_word_patterns, mask_to_word
from .embeddings import SENSE_EXAMPLES, embedding_disambiguate
from .heteronyms import disambiguate
from .hub import get_checkpoint_path
from .model import StressTagger, decode_mask

DATA_DIR = Path(__file__).resolve().parent / "data"

WORD_RE = re.compile(r"[а-щьюяєіїґА-ЩЬЮЯЄІЇҐ'’ʼ-]+")
# слів по кожен бік - але не перетинаючи пунктуацію (stressify_text сам
# обрізає вікно по найближчій комі/крапці/тире тощо з кожного боку)
HETERONYM_CONTEXT_WINDOW = 8

# Текст із буфера обміну (телеграм-бот, копіювання з PDF/Word) іноді містить
# невидимі символи форматування - вони НЕ входять до WORD_RE, тож розбивають
# слово навпіл ("Ві‌н" -> "Ві" + "н" як два окремі "слова"), і кожна частина
# отримує власний наголос: на виході виглядає як "Ві́‌н" - подвійний наголос
# в одному (на вигляд) слові, хоча символ між ними просто невидимий. Тому
# знімаємо їх ще ДО пошуку слів regex'ом, а не лишаємо як "розділювач".
INVISIBLE_CHARS_RE = re.compile("[​‌‍﻿­]")

# Односкладові прийменники, сполучники й частки - у зв'язному тексті завжди
# проклітики/енклітики (не несуть власного наголосу, зливаються з сусіднім
# словом в одну акцентну групу: "на платфо́рму", не "на́ платфо́рму"; "не
# дба́ли", не "не́ дба́ли"). НЕ включені "що"/"як"/"чи" - вони можуть бути й
# питальними словами з власним наголосом ("Що́ ти кажеш?"), а не лише
# сполучниками - там, де однозначно завжди проклітика/енклітика, ризик
# хибного "виправлення" вищий, ніж користь.
UNSTRESSED_FUNCTION_WORDS = frozenset({
    # прийменники
    "в", "у", "з", "із", "зі", "ізо",
    "на", "по", "до", "від", "за", "над", "під", "при", "про",
    "без", "для", "крізь", "між",
    # сполучники й частки - однозначно завжди без власного наголосу
    "і", "й", "та", "а", "не", "б", "би", "ж", "же",
})


class Stressifier:
    def __init__(self, checkpoint_paths: Path | list[Path] | None = None,
                 data_dir: Path = DATA_DIR, device: str | None = None):
        """checkpoint_paths=None (дефолт) - підвантажити готовий чекпоінт із
        HuggingFace Hub при першому використанні (кешується locally, не
        качається повторно). Передай свій шлях (або список шляхів - для
        ансамблю кількох чекпоінтів), щоб використати власну модель."""
        if checkpoint_paths is None:
            checkpoint_paths = [get_checkpoint_path()]
        elif isinstance(checkpoint_paths, Path):
            checkpoint_paths = [checkpoint_paths]
        self.device = torch.device(device or "cpu")
        self.vocab: dict | None = None
        self.models: list[StressTagger] = []
        for path in checkpoint_paths:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if self.vocab is None:
                self.vocab = ckpt["vocab"]
            elif ckpt["vocab"] != self.vocab:
                raise ValueError(
                    f"{path}: словник символів не збігається з іншими чекпоінтами "
                    f"ансамблю - всі мають бути натреновані на тому самому vocab.json"
                )
            model = StressTagger(vocab_size=len(self.vocab), **ckpt.get("arch", {})).to(self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self.models.append(model)
        self.cache = self._build_cache(data_dir)

    @staticmethod
    def _build_cache(data_dir: Path) -> dict[str, str]:
        patterns = load_word_patterns(data_dir)
        return {word: mask_to_word(word, info["best"]) for word, info in patterns.items()}

    @torch.no_grad()
    def _predict(self, plain_word: str) -> str:
        char_ids = torch.tensor(
            [[self.vocab.get(ch, self.vocab["<unk>"]) for ch in plain_word]],
            dtype=torch.long,
        )
        lengths = torch.tensor([len(plain_word)], dtype=torch.long)
        # Ансамбль: усереднюємо sigmoid-ймовірності кількох незалежно
        # натренованих чекпоінтів ПЕРЕД decode_mask (а не голосуємо за вже
        # прийняті рішення). З одним чекпоінтом (типовий випадок) це
        # тотожно звичайній поведінці.
        probs_sum = None
        for model in self.models:
            p = torch.sigmoid(model(char_ids, lengths)[0])
            probs_sum = p if probs_sum is None else probs_sum + p
        probs = (probs_sum / len(self.models)).tolist()
        mask = decode_mask(plain_word, probs)
        return mask_to_word(plain_word, mask)

    def stressify_word(self, word: str) -> str:
        lower = word.lower()
        cached = self.cache.get(lower)
        if cached is not None:
            return cached if word.islower() else cached.capitalize()
        result = self._predict(lower)
        return result if word.islower() else result.capitalize()

    def stressify_text(self, text: str) -> str:
        text = INVISIBLE_CHARS_RE.sub("", text)
        matches = list(WORD_RE.finditer(text))
        words_lower = [m.group(0).lower() for m in matches]

        # Межа контексту - БУДЬ-ЯКА пунктуація між словами (не лише межа
        # речення): кома, крапка з комою, тире, дужка, лапки... Те саме
        # слово в одному реченні, але в іншій, відмежованій комами частині -
        # семантично інший контекст, тригери звідти не надійніші за тригери
        # з іншого речення. WORD_RE сам ковтає апостроф/дефіс ВСЕРЕДИНІ
        # слова, тож будь-який непробільний символ у проміжку між матчами -
        # то й є пунктуація, без окремого regex на конкретні символи.
        segment_id = [0] * len(matches)
        for i in range(1, len(matches)):
            gap = text[matches[i - 1].end():matches[i].start()]
            segment_id[i] = segment_id[i - 1] + (1 if gap.strip() else 0)

        out = []
        last_end = 0
        for i, m in enumerate(matches):
            out.append(text[last_end:m.start()])
            word = m.group(0)

            if words_lower[i] in UNSTRESSED_FUNCTION_WORDS:
                out.append(word)
                last_end = m.end()
                continue

            lo = max(0, i - HETERONYM_CONTEXT_WINDOW)
            hi = min(len(matches), i + HETERONYM_CONTEXT_WINDOW + 1)
            sid = segment_id[i]
            # слово -> відстань у словах до цілі (1 = сусіднє); якщо те
            # саме слово трапляється і ближче, і далі - лишаємо МЕНШУ
            # відстань (найсильніший доказ із усіх його появ у вікні).
            context: dict[str, int] = {}
            for j in range(lo, i):
                if segment_id[j] == sid:
                    w, d = words_lower[j], i - j
                    if d < context.get(w, d + 1):
                        context[w] = d
            for j in range(i + 1, hi):
                if segment_id[j] == sid:
                    w, d = words_lower[j], j - i
                    if d < context.get(w, d + 1):
                        context[w] = d
            override = disambiguate(words_lower[i], context)

            # Стем-тригери не дали відповіді (0 чи ≥2 матчі) - якщо для
            # цього слова є приклади речень (embeddings.SENSE_EXAMPLES,
            # зараз ~47 слів, не повний словник), пробуємо few-shot
            # резолвінг через ембединги того самого контекстного вікна
            # як суцільного тексту. Помітно дорожче (~десятки мс замість
            # мікросекунд) - тому ЛИШЕ фолбек, не заміна тригерів.
            if override is None and words_lower[i] in SENSE_EXAMPLES:
                in_segment = [j for j in range(lo, hi) if segment_id[j] == sid]
                seg_text = text[matches[in_segment[0]].start():matches[in_segment[-1]].end()]
                override = embedding_disambiguate(words_lower[i], seg_text)

            if override is not None:
                out.append(override if word.islower() else override.capitalize())
            else:
                out.append(self.stressify_word(word))
            last_end = m.end()
        out.append(text[last_end:])
        return "".join(out)


@lru_cache(maxsize=1)
def get_default_stressifier() -> Stressifier:
    return Stressifier()


def stressify(word: str) -> str:
    return get_default_stressifier().stressify_word(word)


def stressify_text(text: str) -> str:
    return get_default_stressifier().stressify_text(text)
