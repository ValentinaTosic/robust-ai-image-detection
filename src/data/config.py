"""Utilities for loading experiment configuration and resolving data paths."""

from pathlib import Path
import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "configs" / "experiments.yaml"


def load_config(config_path: Path | str | None = None) -> dict:
    """Load the experiment configuration from a YAML file.

    Args:
        config_path: Path to the configuration file. Uses the default
            experiments.yaml when not provided.

    Returns:
        Configuration values as a dictionary.
    """
    path: Path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def active_sample_size(config: dict) -> int:
    """Return the number of images sampled per generator.

    Args:
        config: Loaded experiment configuration.

    Returns:
        Number of images to sample per generator for the active tier.
    """
    size: str = config["sample_size"]["active_size"]
    return config["sample_size"][size]


def raw_generator_dir(config: dict, generator: str) -> Path:
    """Return the path to a generator's raw data directory.

    Args:
        config: Loaded experiment configuration.
        generator: Generator name defined in the configuration.

    Returns:
        Path to the generator's raw data directory.

    Raises:
        KeyError: If the generator is not defined in the configuration.
    """
    root: Path = Path(config["paths"]["raw_data_root"])
    return root / config["paths"]["generator_dirs"][generator]
