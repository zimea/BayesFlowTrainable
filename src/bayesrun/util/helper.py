import os, sys
import importlib.util
from types import ModuleType

def parse_config(configfile: str) -> ModuleType:
    """Parses a config file and returns the config module.

    Args:
        configfile (str): Path to the config file.

    Returns:
        ModuleType: The config module.
    """
    # load config file
    configfile = os.path.abspath(configfile)
    spec = importlib.util.spec_from_file_location("config", configfile)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    return config