#!/usr/bin/env python3
"""
Create/update a RunPod Serverless template and point an existing endpoint to it.

Required env:
- RUNPOD_ACCOUNT_API_KEY: RunPod account/API key with GraphQL endpoint permissions.
- RUNPOD_ENDPOINT_ID: Existing endpoint ID to update.
- RUNPOD_IMAGE_NAME: Published Docker image, e.g. ghcr.io/org/reverie-runpod-worker:sha.

Optional env:
- RUNPOD_TEMPLATE_NAME: template name, default "reverie-unified-generation-worker".
- RUNPOD_CONTAINER_DISK_GB: default 30.
- RUNPOD_IDLE_TIMEOUT: default 900.
- RUNPOD_WORKERS_MIN: default 0.
- RUNPOD_WORKERS_MAX: default 1.
- RUNPOD_TEMPLATE_ID: reuse an existing template instead of creating one.
- RUNPOD_UPDATE_BOUND_TEMPLATE: default true; update endpoint.template in place
  when RunPod reports a bound template.

The script is dry-run by default. Pass --apply to mutate RunPod.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


GRAPHQL_URL = "https://api.runpod.io/graphql"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually mutate RunPod")
    args = parser.parse_args()

    api_key = require_env("RUNPOD_ACCOUNT_API_KEY")
    endpoint_id = require_env("RUNPOD_ENDPOINT_ID")
    image_name = require_env("RUNPOD_IMAGE_NAME")
    template_name = os.environ.get(
        "RUNPOD_TEMPLATE_NAME", "reverie-unified-generation-worker"
    )
    container_disk_gb = int(os.environ.get("RUNPOD_CONTAINER_DISK_GB", "30"))
    idle_timeout = int(os.environ.get("RUNPOD_IDLE_TIMEOUT", "900"))
    workers_min = int(os.environ.get("RUNPOD_WORKERS_MIN", "0"))
    workers_max = int(os.environ.get("RUNPOD_WORKERS_MAX", "1"))
    template_id_override = os.environ.get("RUNPOD_TEMPLATE_ID")
    update_bound_template = (
        os.environ.get("RUNPOD_UPDATE_BOUND_TEMPLATE", "true").lower() == "true"
    )

    endpoint = get_endpoint(api_key, endpoint_id)
    if not endpoint:
        print(json.dumps({"ok": False, "error": "endpoint_not_found"}))
        return 1

    plan = {
        "endpoint_id": endpoint_id,
        "endpoint_name": endpoint.get("name"),
        "current_template_id": endpoint.get("templateId"),
        "new_template_name": template_name,
        "template_id_override": template_id_override,
        "update_bound_template": update_bound_template,
        "new_image": image_name,
        "gpuIds": endpoint.get("gpuIds"),
        "networkVolumeId": endpoint.get("networkVolumeId"),
        "workersMin": workers_min,
        "workersMax": workers_max,
        "idleTimeout": idle_timeout,
        "apply": args.apply,
    }
    print(json.dumps({"plan": plan}, ensure_ascii=False))

    if not args.apply:
        return 0

    endpoint_template = endpoint.get("template") or {}
    if update_bound_template and endpoint_template.get("id"):
        template = save_template(
            api_key=api_key,
            name=endpoint_template.get("name") or template_name,
            image_name=image_name,
            # 기존 template의 disk/env/dockerArgs를 보존한다 (image만 교체).
            container_disk_gb=endpoint_template.get("containerDiskInGb") or container_disk_gb,
            template_id=endpoint_template["id"],
            volume_mount_path=endpoint_template.get("volumeMountPath") or "/workspace",
            env=endpoint_template.get("env") or [],
            docker_args=endpoint_template.get("dockerArgs") or "",
        )
    elif template_id_override:
        template = {
            "id": template_id_override,
            "name": template_name,
            "imageName": image_name,
        }
    else:
        template = save_template(
            api_key=api_key,
            name=template_name,
            image_name=image_name,
            container_disk_gb=container_disk_gb,
        )
    template_id = template["id"]
    settings_update = save_endpoint(
        api_key=api_key,
        endpoint=endpoint,
        template_id=template_id,
        workers_min=workers_min,
        workers_max=workers_max,
        idle_timeout=idle_timeout,
    )
    if update_bound_template and endpoint_template.get("id"):
        template_update = {"id": endpoint_id, "templateId": template_id}
    else:
        template_update = update_endpoint_template(
            api_key=api_key,
            endpoint_id=endpoint_id,
            template_id=template_id,
        )
    print(
        json.dumps(
            {
                "ok": True,
                "template": {
                    "id": template_id,
                    "name": template.get("name"),
                    "imageName": template.get("imageName"),
                },
                "endpoint": {
                    "id": template_update.get("id"),
                    "name": settings_update.get("name"),
                    "templateId": template_update.get("templateId"),
                    "workersMin": settings_update.get("workersMin"),
                    "workersMax": settings_update.get("workersMax"),
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


def require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        print(json.dumps({"ok": False, "error": f"missing_env:{key}"}))
        sys.exit(2)
    return value


def graphql(api_key: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=body,
        headers={
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "character-chat-codex/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"RunPod GraphQL HTTP {exc.code}: {detail}") from exc
    if payload.get("errors"):
        raise RuntimeError(f"RunPod GraphQL errors: {payload['errors']}")
    return payload["data"]


def get_endpoint(api_key: str, endpoint_id: str) -> dict[str, Any] | None:
    data = graphql(
        api_key,
        """
        query {
          myself {
            endpoints {
              id
              name
              gpuIds
              idleTimeout
              locations
              networkVolumeId
              scalerType
              scalerValue
              templateId
              workersMax
              workersMin
              template {
                id
                name
                imageName
                volumeMountPath
                containerDiskInGb
                dockerArgs
                env {
                  key
                  value
                }
              }
            }
          }
        }
        """,
    )
    endpoints = data.get("myself", {}).get("endpoints", [])
    return next((endpoint for endpoint in endpoints if endpoint.get("id") == endpoint_id), None)


def save_template(
    *,
    api_key: str,
    name: str,
    image_name: str,
    container_disk_gb: int,
    template_id: str | None = None,
    volume_mount_path: str = "/workspace",
    env: list[dict[str, str]] | None = None,
    docker_args: str = "",
) -> dict[str, Any]:
    """Serverless template을 생성/갱신한다.

    주의: saveTemplate은 전체 필드를 덮어쓰므로, 기존 template을 in-place로
    갱신할 때는 호출자가 현재 env/dockerArgs를 반드시 넘겨야 한다.
    (예: PYTORCH_CUDA_ALLOC_CONF 같은 운영 env가 배포마다 초기화되는 사고 방지)
    """
    id_field = f"id: {gql_string(template_id)}" if template_id else ""
    env_items = ", ".join(
        f"{{key: {gql_string(item['key'])}, value: {gql_string(item['value'])}}}"
        for item in (env or [])
    )
    data = graphql(
        api_key,
        f"""
        mutation {{
          saveTemplate(input: {{
            {id_field}
            name: {gql_string(name)}
            imageName: {gql_string(image_name)}
            isServerless: true
            dockerArgs: {gql_string(docker_args)}
            env: [{env_items}]
            containerDiskInGb: {container_disk_gb}
            volumeInGb: 0
            volumeMountPath: {gql_string(volume_mount_path)}
          }}) {{
            id
            name
            imageName
            isServerless
            containerDiskInGb
            volumeMountPath
            env {{
              key
              value
            }}
          }}
        }}
        """,
    )
    return data["saveTemplate"]


def save_endpoint(
    *,
    api_key: str,
    endpoint: dict[str, Any],
    template_id: str,
    workers_min: int,
    workers_max: int,
    idle_timeout: int,
) -> dict[str, Any]:
    network_volume = ""
    if endpoint.get("networkVolumeId"):
        network_volume = f'networkVolumeId: {gql_string(endpoint["networkVolumeId"])}'
    data = graphql(
        api_key,
        f"""
        mutation {{
          saveEndpoint(input: {{
            id: {gql_string(endpoint["id"])}
            name: {gql_string(endpoint["name"])}
            templateId: {gql_string(template_id)}
            gpuIds: {gql_string(endpoint["gpuIds"])}
            workersMin: {workers_min}
            workersMax: {workers_max}
            idleTimeout: {idle_timeout}
            {network_volume}
          }}) {{
            id
            name
            templateId
            workersMin
            workersMax
          }}
        }}
        """,
    )
    return data["saveEndpoint"]


def update_endpoint_template(
    *, api_key: str, endpoint_id: str, template_id: str
) -> dict[str, Any]:
    data = graphql(
        api_key,
        f"""
        mutation {{
          updateEndpointTemplate(input: {{
            endpointId: {gql_string(endpoint_id)}
            templateId: {gql_string(template_id)}
          }}) {{
            id
            templateId
          }}
        }}
        """,
    )
    return data["updateEndpointTemplate"]


def gql_string(value: str) -> str:
    return json.dumps(value)


if __name__ == "__main__":
    raise SystemExit(main())
