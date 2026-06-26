"""Завантаження ваг моделі з HuggingFace Hub при першому використанні
(кешується locally самим huggingface_hub - ~/.cache/huggingface/, повторні
запуски нічого не качають). Сам пакет не несе 9.5MB чекпоінта - лише код +
текстовий словник гетеронімів."""
import os
from pathlib import Path

# TODO: замінити на реальний repo_id після публікації на HuggingFace Hub
# (формат "організація-або-юзернейм/stress-uk").
HF_REPO_ID = "Mikhailo/stress-uk"
HF_FILENAME = "stress_tagger_full.pt"

# Якщо задано - беремо чекпоінт звідси напряму, без звернення до HF Hub
# (зручно для локальної розробки/CI без мережі).
LOCAL_OVERRIDE_ENV = "STRESS_UK_CHECKPOINT"


def get_checkpoint_path() -> Path:
    override = os.environ.get(LOCAL_OVERRIDE_ENV)
    if override:
        return Path(override)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Для автозавантаження моделі з HuggingFace Hub потрібен пакет "
            "huggingface_hub: pip install huggingface_hub. Або встанови "
            f"змінну середовища {LOCAL_OVERRIDE_ENV}=шлях/до/checkpoint.pt, "
            "щоб використати власний файл напряму."
        ) from e

    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME))
