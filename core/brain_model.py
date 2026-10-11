import os
import tempfile
from pathlib import Path


class _BrainConfigLoader:
    @staticmethod
    def load(config_path):
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "Loading the brain checkpoint requires PyYAML. Install the optional "
                "dependencies with: python -m pip install -r requirements-brain.txt"
            ) from error

        class SafeBrainLoader(yaml.SafeLoader):
            pass

        def construct_tuple(loader, node):
            return loader.construct_sequence(node)

        def construct_posix_path(loader, node):
            parts = loader.construct_sequence(node)
            return Path(*parts).as_posix()

        SafeBrainLoader.add_constructor(
            "tag:yaml.org,2002:python/tuple",
            construct_tuple,
        )
        SafeBrainLoader.add_constructor(
            "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
            construct_posix_path,
        )

        try:
            with Path(config_path).open(encoding="utf-8") as config_file:
                config = yaml.load(config_file, Loader=SafeBrainLoader)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            raise ValueError(f"Could not safely read brain model config: {error}") from error

        if not isinstance(config, dict):
            raise ValueError("Brain model config must contain a YAML mapping.")
        if not isinstance(config.get("brain_model_config"), dict):
            raise ValueError("Brain model config is missing brain_model_config.")
        if config["brain_model_config"].get("name") != "FmriEncoder":
            raise ValueError("The config does not describe a supported FmriEncoder.")
        return yaml, config


class BrainResponseModel:
    def __init__(self, checkpoint_path, config_path=None, cache_path=None, device="cpu"):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Brain checkpoint does not exist: {self.checkpoint_path}"
            )

        self.config_path = Path(
            config_path or self.checkpoint_path.parent / "config.yaml"
        ).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Brain model config does not exist: {self.config_path}"
            )

        self.device = device
        yaml, config = _BrainConfigLoader.load(self.config_path)
        infra = config.setdefault("infra", {})
        if not isinstance(infra, dict):
            raise ValueError("Brain model config infra must be a YAML mapping.")
        infra["folder"] = str(self.checkpoint_path.parent)
        data_config = config.get("data")
        if isinstance(data_config, dict):
            for modality in ("text", "audio", "video"):
                feature_config = data_config.get(f"{modality}_feature")
                if not isinstance(feature_config, dict):
                    continue
                if "device" in feature_config:
                    feature_config["device"] = self.device
                image_config = feature_config.get("image")
                if isinstance(image_config, dict) and "device" in image_config:
                    image_config["device"] = self.device
            image_feature = data_config.get("image_feature")
            if isinstance(image_feature, dict) and "device" in image_feature:
                image_feature["device"] = self.device
        self.cache_path = Path(
            cache_path or Path.home() / ".cache" / "lucy" / "tribe"
        ).expanduser(        )
        self.cache_path.mkdir(parents=True, exist_ok=True)

        try:
            from tribev2 import TribeModel
        except ImportError as error:
            raise RuntimeError(
                "Loading this brain checkpoint requires the optional TRIBE v2 runtime. "
                "Use Python 3.11 or newer and install requirements-brain.txt. "
                "TRIBE v2 is licensed CC BY-NC 4.0; check that its terms fit your use."
            ) from error

        with tempfile.TemporaryDirectory(prefix="lucy-brain-config-") as temporary_dir:
            model_dir = Path(temporary_dir)
            safe_config_path = model_dir / "config.yaml"
            with safe_config_path.open("w", encoding="utf-8") as config_file:
                yaml.safe_dump(config, config_file, sort_keys=False)
            checkpoint_link = model_dir / self.checkpoint_path.name
            checkpoint_link.symlink_to(self.checkpoint_path)

            try:
                self.model = TribeModel.from_pretrained(
                    model_dir,
                    checkpoint_name=self.checkpoint_path.name,
                    cache_folder=self.cache_path,
                    device=self.device,
                )
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                raise RuntimeError(
                    f"Could not load the TRIBE brain checkpoint: {error}"
                ) from error

    def predict_file(self, path, modality, verbose=False):
        if modality not in ("text", "audio", "video"):
            raise ValueError("Brain prediction modality must be text, audio, or video.")

        source_path = Path(path).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(f"Brain prediction input does not exist: {source_path}")

        events = self.model.get_events_dataframe(
            **{f"{modality}_path": str(source_path)}
        )
        return self.model.predict(events, verbose=verbose)
