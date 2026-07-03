"""Dataclass-based TOML configuration with sensible defaults."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "echoflow"
CONFIG_PATH = CONFIG_DIR / "config.toml"


@dataclass
class AudioConfig:
    device: str = ""
    sample_rate: int = 16000
    mode: str = "toggle"  # "toggle" or "push_to_talk"


@dataclass
class STTConfig:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"


@dataclass
class RefinerConfig:
    enabled: bool = True
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "echoflow-refiner"
    # 0 = greedy decoding: the same speech always cleans the same way
    temperature: float = 0.0


@dataclass
class InjectorConfig:
    type_delay_ms: int = 0
    clipboard_threshold: int = 500


@dataclass
class OverlayConfig:
    enabled: bool = True


@dataclass
class DictionaryConfig:
    path: str = "~/.config/echoflow/dictionary.txt"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    injector: InjectorConfig = field(default_factory=InjectorConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)


def _merge_section(dc_instance: object, overrides: dict) -> None:
    """Merge a dict of overrides into a dataclass instance."""
    valid = {f.name for f in fields(dc_instance)}
    for key, value in overrides.items():
        if key in valid:
            setattr(dc_instance, key, value)


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML, falling back to defaults for missing keys/file."""
    cfg = Config()
    config_path = path or CONFIG_PATH

    if config_path.exists():
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        section_map = {
            "audio": cfg.audio,
            "stt": cfg.stt,
            "refiner": cfg.refiner,
            "injector": cfg.injector,
            "overlay": cfg.overlay,
            "dictionary": cfg.dictionary,
        }
        for section_name, dc_instance in section_map.items():
            if section_name in data:
                _merge_section(dc_instance, data[section_name])

    return cfg
