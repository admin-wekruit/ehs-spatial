"""Apply the reviewed mesh-only patch to the exact pinned SAM3D source before image build.

Use --fetch for a small, isolated source layer for CPU regression. For a runtime
image, use --source with the installed VCS distribution root (site-packages)
after installing the pinned official package. No weights or GPU calls are made.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import urllib.request


CODE_REVISION = "f91db411c50efee93d8db7aeb323885650f6f722"
PATCH = Path(__file__).resolve().parents[1] / "modal_apps/sam3d_mesh_only.patch"
RECEIPT = "sam3d_objects/panoptes_mesh_build.json"
SOURCE_HASHES = {
    "sam3d_objects/pipeline/inference_pipeline.py": "e83e9560a727b06565134ea855503034d3510cbfbeeff2b6ca6dc13013536d2f",
    "sam3d_objects/pipeline/layout_post_optimization_utils.py": "0603a4455acbe390f46b0c482aaaab9e2ad6da5668be575e646149e84f06ee11",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, *, fetch=False):
    source = Path(source).resolve()
    if (source / RECEIPT).exists():
        raise ValueError("Source already prepared; use the pristine pinned source")
    contents = {}
    for relative, expected in SOURCE_HASHES.items():
        path = source / relative
        if fetch and not path.exists():
            url = f"https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/{CODE_REVISION}/{relative}"
            contents[relative] = urllib.request.urlopen(url, timeout=30).read()
        else:
            contents[relative] = path.read_bytes()
        if sha256(contents[relative]) != expected:
            raise ValueError(f"Source hash differs from pinned upstream: {relative}")
    for relative, content in contents.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    # Apply once during the image build, never modify imported modules at runtime.
    for arguments in (("--check",), ()):
        subprocess.run(["git", "apply", *arguments, str(PATCH)], cwd=source, check=True, capture_output=True)
    report = {"codeRevision": CODE_REVISION, "patchSha256": sha256(PATCH.read_bytes()),
              "files": {relative: {"upstreamSha256": expected, "patchedSha256": sha256((source / relative).read_bytes())}
                        for relative, expected in SOURCE_HASHES.items()}}
    if any(row["upstreamSha256"] == row["patchedSha256"] for row in report["files"].values()):
        raise RuntimeError("Patch was not applied to every expected source file")
    raw = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    (source / RECEIPT).write_bytes(raw)
    return {"status": "source_prepared", "meshSourceBuildSha256": sha256(raw), "newModelCalls": 0, **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Default: installed sam3d_objects distribution root")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch and args.source is None:
        parser.error("--fetch requires an isolated --source directory")
    source = args.source or importlib.metadata.distribution('sam3d_objects').locate_file('')
    print(json.dumps(prepare(source, fetch=args.fetch), indent=2))


if __name__ == "__main__":
    main()
