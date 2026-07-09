import base64
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "handler.py"
SPEC = importlib.util.spec_from_file_location("unified_worker_handler", MODULE_PATH)
handler_module = importlib.util.module_from_spec(SPEC)
sys.modules["unified_worker_handler"] = handler_module
sys.modules.setdefault(
    "requests",
    types.SimpleNamespace(
        RequestException=Exception,
        get=lambda *args, **kwargs: None,
        post=lambda *args, **kwargs: None,
    ),
)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(handler_module)


class UnifiedWorkerHandlerTest(unittest.TestCase):
    def test_missing_input_returns_error(self):
        result = handler_module.handler({"id": "job-1"})
        self.assertIn("error", result)
        self.assertIn("Please provide input", result["error"])

    def test_legacy_image_requires_workflow(self):
        result = handler_module.handler({"id": "job-1", "input": {}})
        self.assertIn("error", result)
        self.assertIn("Missing 'workflow'", result["error"])

    def test_legacy_image_routes_to_comfy_workflow(self):
        with patch.object(handler_module, "run_comfy_workflow") as run_comfy:
            run_comfy.return_value = {"images": []}
            result = handler_module.handler(
                {"id": "job-1", "input": {"workflow": {"1": {"class_type": "SaveImage"}}}}
            )
        self.assertEqual(result, {"images": []})
        run_comfy.assert_called_once()
        self.assertEqual(run_comfy.call_args.kwargs["output_kind"], "images")

    def test_video_routes_to_wan_cli(self):
        png = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            + (832).to_bytes(4, "big")
            + (1216).to_bytes(4, "big")
            + b"\x08\x02\x00\x00\x00"
        )
        encoded = base64.b64encode(png).decode("ascii")
        with patch.object(handler_module, "maybe_free_comfyui_memory"), patch.object(
            handler_module, "run_wan22_cli"
        ) as run_wan:
            run_wan.return_value = {"videos": [{"format": "mp4"}]}
            result = handler_module.handler(
                {
                    "id": "job-2",
                    "input": {
                        "task": "video",
                        "image": {"type": "base64", "data": encoded},
                        "prompt": "subtle motion",
                        "seed": 123,
                        "duration_s": 3,
                        "fps": 16,
                    },
                }
            )
        self.assertEqual(result, {"videos": [{"format": "mp4"}]})
        run_wan.assert_called_once()
        self.assertEqual(run_wan.call_args.kwargs["duration_s"], 3)
        self.assertEqual(run_wan.call_args.kwargs["requested_fps"], 16)

    def test_frames_for_duration_uses_4n_plus_1(self):
        self.assertEqual(handler_module.frames_for_duration(3, 24), 73)
        self.assertEqual((handler_module.frames_for_duration(3, 24) - 1) % 4, 0)

    def test_png_size_reader(self):
        png = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            + (704).to_bytes(4, "big")
            + (1280).to_bytes(4, "big")
            + b"\x08\x02\x00\x00\x00"
        )
        self.assertEqual(handler_module.read_png_size(png), (704, 1280))


if __name__ == "__main__":
    unittest.main()
