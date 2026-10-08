"""Compare measurement geometry while allowing the explicitly approved display-schema migration."""

import argparse
import json
from pathlib import Path


def geometry(layer):
    box_keys = ("centerNative", "axes", "faceNormals", "sizeM", "bottomM", "topM", "floorContact", "dims", "highlight", "confidence")
    face_keys = ("photos", "status", "confidence")
    return {
        "coordinateFrameId": layer["coordinateFrameId"],
        "scale": {key: layer["scale"][key] for key in ("nativeToMeters", "status")},
        "ground": {key: layer["ground"][key] for key in ("normal", "offset", "plane")},
        "boxes": {
            entity: {
                **{key: box[key] for key in box_keys},
                "faces": {face: {key: value[key] for key in face_keys} for face, value in box["faces"].items()},
            }
            for entity, box in layer["boxes"].items()
        },
        "confidence": {entity: value["level"] for entity, value in layer["confidence"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.baseline.read_text())
    actual = geometry(json.loads(args.candidate.read_text()))
    if actual != expected:
        raise SystemExit("measurement regression: geometry, uncertainty, confidence or object identity changed")
    print(f"measurement regression passed: {len(actual['boxes'])} boxes")


if __name__ == "__main__":
    main()
