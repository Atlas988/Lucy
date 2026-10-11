import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from core.brain_model import BrainResponseModel
from core.ai_engine import Lucy


class FakeTribeModel:
    received = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir,
        checkpoint_name,
        cache_folder,
        device,
    ):
        cls.received = {
            "checkpoint_dir": Path(checkpoint_dir),
            "checkpoint_name": checkpoint_name,
            "cache_folder": Path(cache_folder),
            "device": device,
        }
        config_path = cls.received["checkpoint_dir"] / "config.yaml"
        cls.received["config_text"] = config_path.read_text(encoding="utf-8")
        cls.received["config"] = __import__("yaml").safe_load(
            cls.received["config_text"]
        )
        if not (cls.received["checkpoint_dir"] / checkpoint_name).is_file():
            raise FileNotFoundError(checkpoint_name)
        return cls()

    def get_events_dataframe(self, **kwargs):
        self.events_request = kwargs
        return "events"

    def predict(self, events, verbose=True):
        self.prediction_request = (events, verbose)
        return types.SimpleNamespace(shape=(1, 20_484)), ["segment"]


class BrainResponseModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.checkpoint = self.root / "best.ckpt"
        self.checkpoint.write_bytes(b"test checkpoint placeholder")
        self.config = self.root / "config.yaml"
        self.config.write_text(
            "infra:\n"
            "  folder: /old/checkpoint/path\n"
            "  workdir:\n"
            "    excludes: !!python/tuple\n"
            "      - __pycache__\n"
            "data:\n"
            "  text_feature:\n"
            "    device: cuda\n"
            "  feature:\n"
            "    folder: !!python/object/apply:pathlib.PosixPath\n"
            "      - /\n"
            "      - old\n"
            "brain_model_config:\n"
            "  name: FmriEncoder\n",
            encoding="utf-8",
        )
        FakeTribeModel.received = None
        self.tribe_module = types.ModuleType("tribev2")
        self.tribe_module.TribeModel = FakeTribeModel

    def test_loads_checkpoint_through_tribe_safe_weight_loader_with_sanitized_config(self):
        cache = self.root / "cache"
        with patch.dict(sys.modules, {"tribev2": self.tribe_module}):
            model = BrainResponseModel(
                self.checkpoint,
                cache_path=cache,
                device="cpu",
            )

        self.assertIsInstance(model.model, FakeTribeModel)
        self.assertEqual("best.ckpt", FakeTribeModel.received["checkpoint_name"])
        self.assertEqual("cpu", FakeTribeModel.received["device"])
        self.assertTrue(FakeTribeModel.received["cache_folder"].is_dir())
        self.assertEqual("/old", FakeTribeModel.received["config"]["data"]["feature"]["folder"])
        self.assertEqual(
            "cpu",
            FakeTribeModel.received["config"]["data"]["text_feature"]["device"],
        )
        self.assertNotIn("!!python/", FakeTribeModel.received["config_text"])

    def test_predict_file_builds_events_and_returns_prediction_and_segments(self):
        source = self.root / "clip.mp4"
        source.write_bytes(b"video")
        with patch.dict(sys.modules, {"tribev2": self.tribe_module}):
            model = BrainResponseModel(self.checkpoint, cache_path=self.root / "cache")
            predictions, segments = model.predict_file(source, "video")

        self.assertEqual({"video_path": str(source)}, model.model.events_request)
        self.assertEqual(("events", False), model.model.prediction_request)
        self.assertEqual((1, 20_484), predictions.shape)
        self.assertEqual(["segment"], segments)

    def test_predict_file_rejects_unknown_modality(self):
        with patch.dict(sys.modules, {"tribev2": self.tribe_module}):
            model = BrainResponseModel(self.checkpoint, cache_path=self.root / "cache")
        with self.assertRaisesRegex(ValueError, "must be text, audio, or video"):
            model.predict_file(self.checkpoint, "image")

    def test_config_rejects_unrecognized_python_yaml_tags(self):
        self.config.write_text(
            "brain_model_config:\n"
            "  name: FmriEncoder\n"
            "unexpected: !!python/name:os.system\n",
            encoding="utf-8",
        )
        with patch.dict(sys.modules, {"tribev2": self.tribe_module}):
            with self.assertRaisesRegex(ValueError, "Could not safely read"):
                BrainResponseModel(self.checkpoint, cache_path=self.root / "cache")
        self.assertIsNone(FakeTribeModel.received)

    def test_lucy_reports_brain_model_disabled_without_checkpoint(self):
        lucy = Lucy(
            memory_path=self.root / "memory.json",
            knowledge_path=self.root / "knowledge.json",
        )

        self.assertIn("Brain-response model: disabled", lucy.respond("/status"))
        self.assertIn("No brain-response model is loaded", lucy.respond("/brain"))

    def test_lucy_brain_prediction_command_uses_explicit_input(self):
        source = self.root / "clip.mp4"
        source.write_bytes(b"video")
        lucy = Lucy(
            memory_path=self.root / "memory.json",
            knowledge_path=self.root / "knowledge.json",
        )
        lucy.brain_model = BrainResponseModel.__new__(BrainResponseModel)
        lucy.brain_model.checkpoint_path = self.checkpoint
        lucy.brain_model.device = "cpu"
        lucy.brain_model.model = FakeTribeModel()

        response = lucy.respond(f"/brain-predict video {source}")

        self.assertIn("prediction array shape: (1, 20484)", response)
        self.assertIn("not decoded thoughts", response)
        self.assertEqual({"video_path": str(source)}, lucy.brain_model.model.events_request)


if __name__ == "__main__":
    unittest.main()
