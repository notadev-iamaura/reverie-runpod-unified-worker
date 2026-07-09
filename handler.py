"""
RunPod unified generation worker.

Compatibility goals:
- Existing image generation keeps the worker-comfyui contract:
  {"input": {"workflow": <ComfyUI API workflow>, "images": [...]?}}
- AI image-to-video uses:
  {"input": {"task": "video", "image": {"type": "base64", "data": ...}, ...}}
- Model preload uses:
  {"input": {"task": "prepare_video_model"}}

Video generation defaults to Wan2.2 TI2V-5B via the official Wan CLI. The image
path still uses ComfyUI so the current production image workflow can continue
unchanged on the same endpoint and GPU pool.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_TIMEOUT_S = int(os.environ.get("COMFY_TIMEOUT_S", "900"))
COMFY_POLL_INTERVAL_S = float(os.environ.get("COMFY_POLL_INTERVAL_S", "2"))

VIDEO_ENGINE = os.environ.get("VIDEO_ENGINE", "wan22_cli")
WAN22_REPO_DIR = Path(os.environ.get("WAN22_REPO_DIR", "/opt/Wan2.2"))
WAN22_MODEL_DIR = Path(
    os.environ.get("WAN22_MODEL_DIR", "/runpod-volume/models/Wan2.2-TI2V-5B")
)
WAN22_NATIVE_FPS = int(os.environ.get("WAN22_NATIVE_FPS", "24"))
WAN22_SAMPLE_STEPS = int(os.environ.get("WAN22_SAMPLE_STEPS", "24"))
WAN22_AUTO_DOWNLOAD = os.environ.get("WAN22_AUTO_DOWNLOAD", "false").lower() == "true"
WAN22_LANDSCAPE_SIZE = os.environ.get("WAN22_LANDSCAPE_SIZE", "1280*704")
WAN22_PORTRAIT_SIZE = os.environ.get("WAN22_PORTRAIT_SIZE", "704*1280")
WAN22_REQUIRED_FILES = (
    "Wan2.2_VAE.pth",
    "config.json",
    "configuration.json",
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod serverless entry point."""
    try:
        job_input = _normalize_input(job.get("input"))
        task = (job_input.get("task") or "image").lower()
        if task == "image":
            return _handle_image(job.get("id", "job"), job_input)
        if task == "video":
            return _handle_video(job.get("id", "job"), job_input)
        if task in {"prepare_video_model", "prepare-video-model"}:
            return _handle_prepare_video_model()
        if task == "diagnostics":
            return _handle_diagnostics()
        return {"error": f"Unsupported task: {task}"}
    except Exception as exc:  # noqa: BLE001 - RunPod needs structured errors
        return {"error": str(exc)}


def _normalize_input(job_input: Any) -> dict[str, Any]:
    if job_input is None:
        raise ValueError("Please provide input")
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON format in input") from exc
    if not isinstance(job_input, dict):
        raise ValueError("Input must be an object")
    return job_input


def _handle_image(job_id: str, job_input: dict[str, Any]) -> dict[str, Any]:
    workflow = job_input.get("workflow")
    if workflow is None:
        raise ValueError("Missing 'workflow' parameter")
    images = job_input.get("images")
    if images is not None:
        _validate_images(images)
    return run_comfy_workflow(
        job_id=job_id,
        workflow=workflow,
        images=images,
        comfy_org_api_key=job_input.get("comfy_org_api_key"),
        output_kind="images",
    )


def _handle_video(job_id: str, job_input: dict[str, Any]) -> dict[str, Any]:
    if VIDEO_ENGINE != "wan22_cli":
        raise ValueError(f"Unsupported VIDEO_ENGINE={VIDEO_ENGINE}")
    image_payload = job_input.get("image")
    if image_payload is None:
        raise ValueError("Missing 'image' parameter")

    prompt = str(job_input.get("prompt") or "")
    if not prompt.strip():
        prompt = "natural character motion, subtle breathing, smooth cinematic camera drift"
    seed = int(job_input.get("seed") or 0)
    # Wan2.2 TI2V-5B의 검증 범위는 121프레임(24fps 기준 5초)까지 — 그 이상은
    # 장시간 GPU 점유 후 OOM/실패로 끝나므로 워커에서도 상한을 강제한다.
    duration_s = max(1, min(int(job_input.get("duration_s") or 3), 5))
    requested_fps = max(1, min(int(job_input.get("fps") or 16), WAN22_NATIVE_FPS))

    image_bytes = _decode_image_payload(image_payload)
    maybe_free_comfyui_memory()
    return run_wan22_cli(
        image_bytes=image_bytes,
        prompt=prompt,
        seed=seed,
        duration_s=duration_s,
        requested_fps=requested_fps,
    )


def _handle_prepare_video_model() -> dict[str, Any]:
    if wan22_model_present():
        return {
            "ok": True,
            "status": "already_present",
            "model_dir": str(WAN22_MODEL_DIR),
        }
    download_wan22_model()
    return {"ok": True, "status": "downloaded", "model_dir": str(WAN22_MODEL_DIR)}


def _handle_diagnostics() -> dict[str, Any]:
    paths = [Path("/runpod-volume"), Path("/workspace"), WAN22_MODEL_DIR, WAN22_REPO_DIR]
    return {
        "ok": True,
        "video_engine": VIDEO_ENGINE,
        "wan22_model_present": wan22_model_present(),
        "wan22_missing_files": missing_wan22_files(),
        "python": sys.executable,
        "huggingface_cli": shutil.which("huggingface-cli"),
        "paths": {str(path): path_report(path) for path in paths},
    }


def _validate_images(images: Any) -> None:
    if not isinstance(images, list):
        raise ValueError("'images' must be a list")
    for image in images:
        if not isinstance(image, dict) or "name" not in image or "image" not in image:
            raise ValueError("'images' must be a list of objects with 'name' and 'image' keys")


def _decode_image_payload(image_payload: Any) -> bytes:
    if isinstance(image_payload, str):
        data = image_payload
    elif isinstance(image_payload, dict):
        data = str(image_payload.get("data") or image_payload.get("image") or "")
    else:
        raise ValueError("'image' must be a base64 string or object")
    if not data:
        raise ValueError("'image.data' is required")
    if "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def check_comfy_server() -> None:
    deadline = time.monotonic() + COMFY_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"http://{COMFY_HOST}/", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(COMFY_POLL_INTERVAL_S)
    raise TimeoutError(f"ComfyUI server ({COMFY_HOST}) not reachable")


def upload_images(images: list[dict[str, str]] | None) -> None:
    if not images:
        return
    for image in images:
        data = image["image"]
        if "," in data:
            data = data.split(",", 1)[1]
        blob = base64.b64decode(data)
        files = {
            "image": (image["name"], BytesIO(blob), "image/png"),
            "overwrite": (None, "true"),
        }
        response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files, timeout=60)
        response.raise_for_status()


def run_comfy_workflow(
    *,
    job_id: str,
    workflow: dict[str, Any],
    images: list[dict[str, str]] | None,
    comfy_org_api_key: str | None,
    output_kind: str,
) -> dict[str, Any]:
    check_comfy_server()
    upload_images(images)

    client_id = f"runpod-{job_id}"
    payload: dict[str, Any] = {"prompt": workflow, "client_id": client_id}
    effective_key = comfy_org_api_key or os.environ.get("COMFY_ORG_API_KEY")
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}

    response = requests.post(f"http://{COMFY_HOST}/prompt", json=payload, timeout=60)
    if response.status_code == 400:
        raise ValueError(f"ComfyUI workflow validation failed: {response.text[:1000]}")
    response.raise_for_status()
    prompt_id = response.json().get("prompt_id")
    if not prompt_id:
        raise ValueError(f"Missing prompt_id in ComfyUI response: {response.text[:1000]}")

    history = poll_history(prompt_id)
    outputs = history.get("outputs", {})
    result: dict[str, Any] = {}
    errors: list[str] = []
    images_out: list[dict[str, str]] = []
    videos_out: list[dict[str, str]] = []

    for node_output in outputs.values():
        for image_info in node_output.get("images", []):
            if image_info.get("type") == "temp":
                continue
            images_out.append(collect_comfy_file(image_info, default_ext="png"))
        for key in ("gifs", "videos", "animated"):
            for video_info in node_output.get(key, []):
                videos_out.append(collect_comfy_file(video_info, default_ext="mp4"))
        other_keys = [key for key in node_output if key not in {"images", "gifs", "videos", "animated"}]
        if other_keys:
            errors.append(f"Unhandled ComfyUI output keys: {other_keys}")

    if output_kind == "images":
        result["images"] = images_out
    else:
        result["videos"] = videos_out
    if errors:
        result["errors"] = errors
    return result


def poll_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + COMFY_TIMEOUT_S
    while time.monotonic() < deadline:
        response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=60)
        response.raise_for_status()
        payload = response.json()
        if prompt_id in payload:
            entry = payload[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution error: {status}")
            return entry
        time.sleep(COMFY_POLL_INTERVAL_S)
    raise TimeoutError(f"ComfyUI workflow timeout ({COMFY_TIMEOUT_S}s)")


def collect_comfy_file(file_info: dict[str, Any], default_ext: str) -> dict[str, str]:
    filename = file_info.get("filename")
    if not filename:
        raise ValueError(f"ComfyUI output missing filename: {file_info}")
    params = urllib.parse.urlencode(
        {
            "filename": filename,
            "subfolder": file_info.get("subfolder", ""),
            "type": file_info.get("type", "output"),
        }
    )
    response = requests.get(f"http://{COMFY_HOST}/view?{params}", timeout=120)
    response.raise_for_status()
    ext = Path(filename).suffix.lower().lstrip(".") or default_ext
    output_type = "base64"
    data = base64.b64encode(response.content).decode("ascii")
    item = {"filename": filename, "type": output_type, "data": data}
    if default_ext != "png":
        item["format"] = ext
    return item


def maybe_free_comfyui_memory() -> None:
    try:
        requests.post(
            f"http://{COMFY_HOST}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10,
        )
    except requests.RequestException:
        return


def run_wan22_cli(
    *,
    image_bytes: bytes,
    prompt: str,
    seed: int,
    duration_s: int,
    requested_fps: int,
) -> dict[str, Any]:
    generate_py = WAN22_REPO_DIR / "generate.py"
    if not generate_py.exists():
        raise RuntimeError(f"Wan2.2 repo not found at {WAN22_REPO_DIR}")
    if not wan22_model_present():
        if WAN22_AUTO_DOWNLOAD:
            download_wan22_model()
        else:
            raise RuntimeError(
                f"Wan2.2 model not found at {WAN22_MODEL_DIR}. "
                "Preload Wan-AI/Wan2.2-TI2V-5B on the RunPod network volume "
                "or set WAN22_AUTO_DOWNLOAD=true for first-run download."
            )

    frame_num = frames_for_duration(duration_s, WAN22_NATIVE_FPS)
    with tempfile.TemporaryDirectory(prefix="charchat-video-") as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "source.png"
        raw_video_path = tmp_path / "wan22.mp4"
        final_video_path = tmp_path / "video.mp4"
        image_path.write_bytes(image_bytes)

        image_width, image_height = read_png_size(image_bytes)
        size = WAN22_PORTRAIT_SIZE if image_height >= image_width else WAN22_LANDSCAPE_SIZE

        command = [
            sys.executable,
            str(generate_py),
            "--task",
            "ti2v-5B",
            "--size",
            size,
            "--ckpt_dir",
            str(WAN22_MODEL_DIR),
            "--offload_model",
            "True",
            "--convert_model_dtype",
            "--t5_cpu",
            "--image",
            str(image_path),
            "--prompt",
            prompt,
            "--base_seed",
            str(seed),
            "--frame_num",
            str(frame_num),
            "--sample_steps",
            str(WAN22_SAMPLE_STEPS),
            "--save_file",
            str(raw_video_path),
        ]
        run_command(command, cwd=str(WAN22_REPO_DIR), timeout=COMFY_TIMEOUT_S)
        output_path = raw_video_path
        if requested_fps != WAN22_NATIVE_FPS:
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(raw_video_path),
                    "-vf",
                    f"fps={requested_fps}",
                    "-an",
                    "-movflags",
                    "+faststart",
                    str(final_video_path),
                ],
                timeout=180,
            )
            output_path = final_video_path
        video_bytes = output_path.read_bytes()

    return {
        "videos": [
            {
                "filename": "charchat-video.mp4",
                "type": "base64",
                "data": base64.b64encode(video_bytes).decode("ascii"),
                "format": "mp4",
            }
        ]
    }


def download_wan22_model() -> None:
    WAN22_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.environ.get("HF_HUB_CACHE", "/runpod-volume/.cache/huggingface/hub"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Wan-AI/Wan2.2-TI2V-5B",
            local_dir=str(WAN22_MODEL_DIR),
            cache_dir=str(cache_dir),
        )
    except Exception as exc:  # noqa: BLE001 - surface detailed RunPod job errors
        volume_report = path_report(Path("/runpod-volume"))
        raise RuntimeError(
            "Failed to download Wan-AI/Wan2.2-TI2V-5B "
            f"to {WAN22_MODEL_DIR}: {exc}; /runpod-volume={volume_report}"
        ) from exc


def wan22_model_present() -> bool:
    return WAN22_MODEL_DIR.exists() and not missing_wan22_files()


def missing_wan22_files() -> list[str]:
    return [
        relative_path
        for relative_path in WAN22_REQUIRED_FILES
        if not (WAN22_MODEL_DIR / relative_path).exists()
    ]


def path_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"exists": path.exists()}
    if path.exists():
        usage = shutil.disk_usage(path)
        report.update(
            {
                "is_dir": path.is_dir(),
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
            }
        )
    return report


def run_command(
    command: list[str], *, cwd: str | None = None, timeout: int | float | None = None
) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode == 0:
        return
    command_display = " ".join(command)
    raise RuntimeError(
        f"Command failed ({completed.returncode}): {command_display}; "
        f"stdout_tail={tail(completed.stdout)}; stderr_tail={tail(completed.stderr)}"
    )


def tail(value: str | None, limit: int = 2000) -> str:
    if not value:
        return ""
    value = value.strip()
    return value[-limit:]


def frames_for_duration(duration_s: int, native_fps: int) -> int:
    """Wan frame count must be 4n+1. Choose the nearest value at native fps."""
    target = max(5, duration_s * native_fps)
    n = max(1, round((target - 1) / 4))
    return 4 * n + 1


def read_png_size(image_bytes: bytes) -> tuple[int, int]:
    # PNG IHDR: signature 8, length 4, type 4, then width/height 4 bytes each.
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(image_bytes) >= 24:
        return (
            int.from_bytes(image_bytes[16:20], "big"),
            int.from_bytes(image_bytes[20:24], "big"),
        )
    return (1, 1)


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
