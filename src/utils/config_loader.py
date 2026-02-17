"""
Configuration file loader for YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union


class ConfigLoader:
    """
    Load and parse YAML configuration files.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize config loader.

        Parameters
        ----------
        config_path : str or Path
            Path to YAML configuration file
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns
        -------
        dict
            Configuration dictionary
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Parameters
        ----------
        key : str
            Configuration key (supports nested keys with dot notation)
        default : Any
            Default value if key not found

        Returns
        -------
        Any
            Configuration value
        """
        config = self.load()

        # Support nested keys like 'data.sources.primary'
        keys = key.split('.')
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @staticmethod
    def save(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        config_path : str or Path
            Path to save configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
