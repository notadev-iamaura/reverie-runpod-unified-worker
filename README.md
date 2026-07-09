# RunPod Unified Worker

Single RunPod Serverless worker for Character Chat generation.

## Contract

Image generation remains compatible with `runpod/worker-comfyui`:

```json
{
  "input": {
    "workflow": {}
  }
}
```

AI image-to-video uses the new task contract:

```json
{
  "input": {
    "task": "video",
    "image": {"type": "base64", "data": "<png base64>"},
    "prompt": "natural breathing, subtle camera drift",
    "seed": 123,
    "duration_s": 3,
    "fps": 16
  }
}
```

Output:

```json
{
  "videos": [
    {
      "filename": "charchat-video.mp4",
      "type": "base64",
      "data": "<mp4 base64>",
      "format": "mp4"
    }
  ]
}
```

## Model

Default video engine: Wan2.2 TI2V-5B.

Reason:

- Handles image-to-video and text-image-to-video in one model.
- Official Wan docs describe TI2V-5B as runnable on RTX 4090-class 24GB GPUs with offload flags.
- Better fit than Wan2.2 14B I2V for the current cost target.

Expected model path:

```bash
/runpod-volume/models/Wan2.2-TI2V-5B
```

Prepare the model on the endpoint network volume:

```bash
WAN22_MODEL_DIR=/runpod-volume/models/Wan2.2-TI2V-5B \
  download_wan22_ti2v_5b.sh
```

Do not bake the model into the image by default. The RunPod GitHub builder has image-size and build-time limits, and model downloads belong on the network volume.

## Endpoint Settings

Recommended first production settings:

- GPU: `NVIDIA GeForce RTX 4090`
- Workers min: `0`
- Workers max: `1`
- Idle timeout: `600-1200`
- Execution timeout: `1800000` ms
- Network volume: keep the current image model volume attached
- Env:
  - `VIDEO_ENGINE=wan22_cli`
  - `WAN22_MODEL_DIR=/runpod-volume/models/Wan2.2-TI2V-5B`
  - `WAN22_SAMPLE_STEPS=24`
  - `COMFY_TIMEOUT_S=1800`

If 4090 runs out of memory or latency is unacceptable, move the endpoint GPU tier to L40S/48GB.

## Deploy Path

1. Build and publish this directory as a Docker image, or use RunPod GitHub integration.
2. Create/update a RunPod serverless template with this image.
3. Update the existing endpoint to the new template while keeping the same endpoint ID.
4. Test image generation with the old `workflow` payload.
5. Test `task=video`.
6. Only after both pass, set Railway:
   - `RUNPOD_WORKER_MODE=unified`
   - `VIDEO_BACKEND=runpod`

The current Railway app should stay at `VIDEO_BACKEND=disabled` until step 5 passes.
