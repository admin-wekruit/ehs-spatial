"""One 16-frame fal structure probe using the existing report container's identity.

Run with the serving venv Python. No credentials leave that container, no service
is redeployed, and the unique output directory prevents accidental resubmission.
"""

import argparse
import asyncio
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

APP_NAME = "panoptes-report-workspace"
ENDPOINT = "fal-ai/sam-3-1/video-rle"

# A single raw queue POST avoids the SDK's automatic submission retries.
REMOTE = r'''
import base64, gzip, hashlib, json, os, re, sys, time, urllib.request
key = os.environ.get("FAL_KEY")
if not key:
    raise RuntimeError("FAL_KEY is absent in the existing provider container")
headers = {"Authorization": "Key " + key, "Content-Type": "application/json"}
def request(url, body=None):
    req = urllib.request.Request(url, headers=headers,
        data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(65536).decode("utf-8", errors="replace").replace(key, "[credential]")
        detail = re.sub(r"(?:https?://|data:)[^\s\"']+", "[url]", detail)
        emit("http_error", {"status": error.code, "method": req.get_method(), "detail": detail})
        raise
def emit(phase, data):
    print(json.dumps({"phase": phase, "data": data}), flush=True)
payload = json.load(sys.stdin)
endpoint = payload["endpoint"]
started = time.perf_counter()
if payload["mode"] == "recover":
    submitted = payload["submission"]
    emit("resuming", {"request_id": submitted["request_id"]})
else:
    pricing = request("https://api.fal.ai/v1/models/pricing?endpoint_id=" + endpoint)
    price = next(p for p in pricing["prices"] if p["endpoint_id"] == endpoint)
    emit("pricing", pricing)
    units, cap = payload.get("billing_units", 1), payload.get("max_fal_usd", 0.02)
    if (not isinstance(units, int) or not 1 <= units <= 12 or not 0 < cap <= 2
            or price["currency"] != "USD" or price["unit"] != "units"
            or float(price["unit_price"]) * units > cap):
        raise RuntimeError("Pricing exceeds the authorized bounded request contract")
    emit("cost_estimate", {"billing_units": units, "published_price_estimate_usd": float(price["unit_price"]) * units,
                           "max_fal_usd": cap, "actual_billed_usd": None})
    submitted = request("https://queue.fal.run/" + endpoint, payload["input"])
    emit("submitted", submitted)
deadline = time.monotonic() + 240
while time.monotonic() < deadline:
    status = request(submitted["status_url"])
    if status["status"] == "COMPLETED":
        result = request(submitted["response_url"])
        raw = json.dumps(result).encode()
        encoded = base64.b64encode(gzip.compress(raw)).decode()
        chunks = [encoded[i:i + 4096] for i in range(0, len(encoded), 4096)]
        emit("provider_output_meta", {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                       "chunks": len(chunks), "keys": list(result)})
        for index, chunk in enumerate(chunks):
            emit("provider_output_chunk", {"index": index, "chunk": chunk})
        emit("provider_output_complete", {})
        emit("timing", {"provider_call_wall_seconds": time.perf_counter() - started})
        break
    time.sleep(2)
else:
    emit("pending", {"request_id": submitted["request_id"], "resubmit": False})
'''


async def drain_process(process, input_bytes: bytes, on_line):
    async def send_input():
        for offset in range(0, len(input_bytes), 65536):
            process.stdin.write(input_bytes[offset:offset + 65536])
            await process.stdin.drain()
        process.stdin.write_eof()
        await process.stdin.drain()

    async def read_output():
        async for line in process.stdout:
            on_line(line)

    code, _, stderr, _ = await asyncio.gather(
        process.wait(), read_output(), process.stderr.read(), send_input())
    return code, len(stderr)


async def container_command(container_id: str, payload: dict, on_line, program: str = REMOTE):
    # ponytail: Modal 1.5.4 has no public exec handle for an existing app container.
    # Use its CLI's process construction, but await PIPE EOF as well as exit;
    # replace this small SDK-version-bound boundary when a public exec handle exists.
    from importlib.metadata import version
    from modal.client import _Client
    from modal.container_process import _ContainerProcess
    from modal._utils.task_command_router_client import TaskCommandRouterClient
    from modal_proto import task_command_router_pb2 as sr_pb2

    if version("modal") != "1.5.4":
        raise RuntimeError("Recheck the existing-container exec contract for this Modal SDK version")
    client = await _Client.from_env()
    router = await TaskCommandRouterClient.init(client, container_id)
    process_id = str(uuid.uuid4())
    await router.exec_start(sr_pb2.TaskExecStartRequest(
        task_id=container_id, exec_id=process_id,
        command_args=["python", "-c", program],
        stdout_config=sr_pb2.TaskExecStdoutConfig.TASK_EXEC_STDOUT_CONFIG_PIPE,
        stderr_config=sr_pb2.TaskExecStderrConfig.TASK_EXEC_STDERR_CONFIG_PIPE))
    process = _ContainerProcess(process_id, container_id, client,
                                command_router_client=router, by_line=True)
    async with asyncio.timeout(300):
        return await drain_process(process, json.dumps(payload).encode(), on_line)


def execute(payload: dict, output: Path, log_name: str, app_name: str = APP_NAME, *, program: str = REMOTE) -> None:
    import modal
    web_url = modal.Function.from_name(app_name, "web").get_web_url()
    if not web_url:
        raise RuntimeError("The requested app has no deployed web endpoint")
    try:
        with urllib.request.urlopen(web_url.rstrip("/") + "/health", timeout=50):
            pass
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    cli = str(Path(sys.executable).with_name("modal"))
    apps = json.loads(subprocess.check_output([cli, "app", "list", "--json"], timeout=20))
    matches = [app for app in apps if app["description"] == app_name and app["state"] == "deployed"]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one deployed app with the requested name")
    containers = json.loads(subprocess.check_output(
        [cli, "container", "list", "--app-id", matches[0]["app_id"], "--json"], timeout=20))
    if len(containers) != 1:
        raise RuntimeError("Expected one existing report service container")
    container_id = containers[0]["container_id"]
    started = time.perf_counter()
    phases = set()
    chunks = []
    metadata = None
    with (output / log_name).open("x") as log:
        def on_line(line):
            nonlocal metadata
            event = json.loads(line)
            log.write(json.dumps(event) + "\n")
            log.flush()
            phases.add(event["phase"])
            if event["phase"] == "provider_output_meta":
                metadata = event["data"]
            elif event["phase"] == "provider_output_chunk":
                if event["data"]["index"] != len(chunks):
                    raise ValueError("Out-of-order output chunk")
                chunks.append(event["data"]["chunk"])
            elif event["phase"] == "provider_output_complete":
                raw = gzip.decompress(base64.b64decode("".join(chunks), validate=True))
                if (len(chunks) != metadata["chunks"] or len(raw) != metadata["bytes"]
                        or hashlib.sha256(raw).hexdigest() != metadata["sha256"]):
                    raise ValueError("Provider output hash/length mismatch")
                with (output / "provider-output.json").open("xb") as complete:
                    complete.write(raw)
                print(json.dumps({"phase": "provider_output", "keys": metadata["keys"],
                                  "bytes": len(raw), "sha256_verified": True}), flush=True)
            elif event["phase"] == "submitted":
                print(json.dumps({"phase": "submitted", "request_id": event["data"]["request_id"]}), flush=True)
            else:
                print(json.dumps({"phase": event["phase"]}), flush=True)
        from modal._utils.async_utils import synchronizer
        code, stderr_bytes = synchronizer.create_blocking(container_command)(container_id, payload, on_line, program)
    receipt = {"exit_code": code, "phases": sorted(phases), "stderr_bytes": stderr_bytes,
               "app_name": app_name, "app_id": matches[0]["app_id"],
               "client_wall_seconds": time.perf_counter() - started,
               "output_directory": str(output), "resubmit": False}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    if code or "provider_output_complete" not in phases:
        raise RuntimeError("Probe did not return output; inspect saved events without resubmitting")


def run(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    clip = output / "first-16-frames.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:v:0",
                    "-frames:v", "16", "-fps_mode", "passthrough", "-c:v", "libx264",
                    "-crf", "23", "-preset", "fast", "-an", str(clip)], check=True, timeout=30)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time:stream=width,height",
        "-of", "json", str(clip)], timeout=20))
    if len(probe["frames"]) != 16 or clip.stat().st_size > 160_000:
        raise ValueError("Probe requires exactly 16 frames and at most 160000 encoded bytes")
    source_probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)], timeout=30))
    manifest = {"endpoint": ENDPOINT, "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
                "source_frame_indices": list(range(16)),
                "source_timestamps_seconds": [float(f["best_effort_timestamp_time"])
                                              for f in source_probe["frames"][:16]],
                "clip_metadata": probe, "video_bytes": clip.stat().st_size,
                "frames": 16, "fal_max_usd": 0.02, "cpu_preparation_max_usd": 0.1,
                "retries": 0, "input_encoding": "H264 CRF23, original frame order, no resampling"}
    (output / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    encoded = base64.b64encode(clip.read_bytes()).decode()
    execute({"mode": "submit", "endpoint": ENDPOINT, "input": {
        "video_url": "data:video/mp4;base64," + encoded,
        "prompt": "person", "apply_mask": False, "boundingbox_zip": True,
        "detection_threshold": 0.5, "max_num_objects": 16,
    }}, output, "provider-events.jsonl")


def recover(output: Path) -> None:
    events = [json.loads(line) for path in output.glob("*.jsonl")
              for line in path.read_text().splitlines()]
    submissions = [event["data"] for event in events if event["phase"] == "submitted"]
    if len(submissions) != 1 or (output / "provider-output.json").exists():
        raise ValueError("Recovery requires exactly one existing submission without a saved output")
    execute({"mode": "recover", "endpoint": ENDPOINT, "submission": submissions[0]},
            output, f"recovery-{time.time_ns()}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path)
    mode.add_argument("--recover", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.output) if args.recover else run(args.source, args.output)
