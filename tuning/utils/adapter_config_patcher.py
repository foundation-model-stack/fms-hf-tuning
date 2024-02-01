import os
import json

class AdapterConfigPatcher:
    def __init__(self, checkpoint_path: str, overrides: dict):
        self.checkpoint_path = checkpoint_path
        self.overrides = overrides
        self.config_path = AdapterConfigPatcher._locate_adapter_config(self.checkpoint_path)
        # Values that we will patch later on
        self.patched_values = {}

    @staticmethod
    def _locate_adapter_config(checkpoint_path: str) -> str:
        """Given a path to a tuned checkpoint, tries to find the adapter_config
        that will be loaded through the Peft auto model API.

        Args:
            checkpoint_path: str
                Checkpoint model, which presumably holds an adapter config.

        Returns:
            str
                Path to the located adapter_config file.
        """
        config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Could not locate adapter config: {config_path}")
        return config_path

    def _apply_config_changes(self, overrides: dict) -> dict:
        """Applies a patch to a config with some override dict, returning the values
        that we patched over so that they may be restored later.
        
        Args:
            overrides: dict
                Overrides to write into the adapter_config.json. Currently, we
                require all override keys to be defined in the config being patched.

        Returns:
            dict
                Dict containing the values that we patched over.
        """
        # If we have no overrides, this context manager is a noop; no need to do anything
        if not overrides:
            return {}
        with open(self.config_path, "r") as config_file:
            adapter_config = json.load(config_file)
        overridden_values = self._get_old_config_values(adapter_config, overrides)
        adapter_config = {**adapter_config, **overrides}
        with open(self.config_path, "w") as config_file:
            json.dump(adapter_config, config_file, indent=4)
        return overridden_values

    @staticmethod
    def _get_old_config_values(adapter_config: dict, overrides: dict) -> dict:
        """Grabs the existing config subdict that we are going to clobber from the
        loaded adapter_config.

        Args:
            adapter_config: dict
                Adapter config whose values we are interested in patching.
            overrides: dict
                Dict of overrides, containing keys definined in the adapter_config with
                new values.

        Returns:
            dict
                The subdictionary of adapter_config, containing the keys in overrides,
                with the values that we are going to replace.
        """
        # For now, we only expect to patch the base model; this may change in the future,
        # but ensure that anything we are patching is defined in the original config
        if not set(overrides.keys()).issubset(set(adapter_config.keys())):
            raise KeyError("Adapter config overrides must be set in the config being patched")
        return {key: adapter_config[key] for key in overrides}

    def __enter__(self):
        """Apply the config overrides and saved the patched values."""
        self.patched_values = self._apply_config_changes(self.overrides)

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Apply the patched values over our exported overrides."""
        self._apply_config_changes(self.patched_values)
