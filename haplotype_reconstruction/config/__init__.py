"""The run configuration: schema (RunConfig), loader (preset + TOML + options + --set), presets, stage mapping."""
from .load import ConfigError, config_text, default_output, load_config, write_config
from .schema import RunConfig

__all__ = ["ConfigError", "RunConfig", "config_text", "default_output", "load_config", "write_config"]
