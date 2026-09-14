"""Make source-pixel review crops for every new geometry identity decision."""
from __future__ import annotations

import argparse
from itertools import combinations, product
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.research.prepare_object_identity_inputs import checked
from scripts.research.scan_object_identity import encoded, file_ref, sha


def materialize(results_path, output):
    results = json.loads(Path(results_path).read_bytes())
    manifest = json.loads(checked(results["inputManifest"]))
    assets = {a["id"]: a for a in manifest["assets"]}
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    pairs = {}
    for target in results["results"]:
        document = json.loads(checked(target["resultDocument"]))
        observations = {o["id"]: o for o in document["observations"]}
        for decision in document.get("identityDecisions", []):
            if decision["source"] != "geometry" or decision["decision"] != "same":
                continue
            for left, right in combinations(decision["observationGroups"], 2):
                for a, b in product(left, right):
                    ids = sorted((a, b))
                    key = sha(encoded([{"observationId": o, "imageSha256": assets[observations[o]["imageId"]]["sha256"],
                        "originalPixelBox": observations[o]["originalPixelBox"],
                        "maskSha256": assets[observations[o]["maskAssetId"]]["sha256"] if observations[o].get("maskAssetId") else None} for o in ids]))
                    record = pairs.setdefault(key, {"key": key, "observationIds": ids,
                        "observations": [observations[o] for o in ids], "coveredRevisions": [], "decisions": []})
                    if target["revisionId"] not in record["coveredRevisions"]:
                        record["coveredRevisions"].append(target["revisionId"])
                    record["decisions"].append({"revisionId": target["revisionId"], "decisionId": decision["id"],
                        "entityIds": decision["entityIds"], "observationGroups": decision["observationGroups"]})
    font = ImageFont.load_default(size=20)
    records, overview = [], Image.new("RGB", (1400, max(1, len(pairs)) * 430), "#f0f0f0")
    for index, record in enumerate(sorted(pairs.values(), key=lambda r: r["key"]), 1):
        pair_id = f"pair-{index:02d}"
        canvas = Image.new("RGB", (1600, 1150), "#f0f0f0")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), pair_id + " | unreviewed geometry merge | original photo and exact crop", fill="black", font=font)
        source_records = []
        for column, observation in enumerate(record.pop("observations")):
            asset = assets[observation["imageId"]]
            image = Image.open(io.BytesIO(checked(asset["file"]))).convert("RGB")
            box = observation["originalPixelBox"]
            pixel_box = (max(0, math.floor(box[0])), max(0, math.floor(box[1])), min(image.width, math.ceil(box[2])), min(image.height, math.ceil(box[3])))
            if pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3]:
                raise ValueError("Empty original source box")
            crop = image.crop(pixel_box)
            crop_path = output / f"{pair_id}-{column + 1}-original-crop.png"
            crop.save(crop_path)
            context = image.copy()
            ImageDraw.Draw(context).rectangle(box, outline="#ff7300", width=max(3, image.width // 400))
            context = ImageOps.contain(context, (760, 500))
            crop_preview = ImageOps.contain(crop, (760, 500))
            x = 20 + column * 800
            canvas.paste(context, (x + (760 - context.width) // 2, 75))
            canvas.paste(crop_preview, (x + (760 - crop_preview.width) // 2, 630))
            labels = ", ".join(str(e.get("label", "")) for e in observation.get("labelEvidence", []))
            draw.text((x, 47), observation["id"][:8] + " | " + labels[:42], fill="black", font=font)
            draw.text((x, 596), "Original box " + str([round(v, 2) for v in box]), fill="black", font=font)
            thumb = ImageOps.contain(crop, (670, 360))
            overview.paste(thumb, (15 + column * 700 + (670 - thumb.width) // 2, (index - 1) * 430 + 50))
            source_records.append({"observationId": observation["id"], "originalPixelBox": box, "cropPixelBox": pixel_box,
                "imageAssetId": asset["id"], "image": asset["file"], "maskAssetId": observation.get("maskAssetId"),
                "labels": labels, "crop": file_ref(crop_path), "rawCropResampled": False})
        ImageDraw.Draw(overview).text((18, (index - 1) * 430 + 12), pair_id + " | " + " / ".join(s["observationId"][:8] for s in source_records), fill="black", font=font)
        path = output / f"{pair_id}.png"; canvas.save(path)
        records.append({**record, "pairId": pair_id, "sources": source_records, "reviewImage": file_ref(path), "truth": "unknown", "reviewer": None})
    overview_path = output / "contact-sheet.png"; overview.save(overview_path)
    report = {"schemaVersion": 1, "sourceResults": file_ref(Path(results_path).resolve()), "pairs": records,
        "pairCount": len(records), "coveredRevisionCount": len({r for p in records for r in p["coveredRevisions"]}),
        "contactSheet": file_ref(overview_path), "note": "Geometry proposals only; no human labels inferred. Display previews are scaled; saved raw crops retain original source pixels."}
    (output / "manifest.json").write_bytes(encoded(report) + b"\n")
    print(json.dumps({"pairs": len(records), "coveredRevisions": report["coveredRevisionCount"], "manifest": str(output / "manifest.json")}))
    return report


def materialize_groups(document_path, manifest_path, output, comparisons=()):
    document_ref, manifest_ref = file_ref(Path(document_path).resolve()), file_ref(Path(manifest_path).resolve())
    document, manifest = json.loads(checked(document_ref)), json.loads(checked(manifest_ref))
    assets = {a["id"]: a for a in manifest["assets"]}
    observations = {o["id"]: o for o in document["observations"]}
    groups = [{"id": d["id"], "kind": "existing_source_binding", "observationIds": [o for g in d["observationGroups"] for o in g],
        "claim": "source says same; requires visual review", "decision": d} for d in document.get("identityDecisions", []) if d["source"] == "source_binding"]
    groups.extend({**g, "kind": "requested_comparison", "claim": "candidate different objects; requires visual review"} for g in comparisons)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default(size=20)
    for index, group in enumerate(groups, 1):
        canvas = Image.new("RGB", (600 * len(group["observationIds"]), 1100), "#f0f0f0")
        draw, sources = ImageDraw.Draw(canvas), []
        draw.text((15, 10), group.get("description", group["kind"]) + " | UNREVIEWED", fill="black", font=font)
        for column, oid in enumerate(group["observationIds"]):
            observation = observations[oid]; asset = assets[observation["imageId"]]
            image = Image.open(io.BytesIO(checked(asset["file"]))).convert("RGB")
            box = observation["originalPixelBox"]
            pixel_box = (max(0, math.floor(box[0])), max(0, math.floor(box[1])), min(image.width, math.ceil(box[2])), min(image.height, math.ceil(box[3])))
            crop = image.crop(pixel_box)
            path = output / f"group-{index:02d}-{column + 1}-original-crop.png"; crop.save(path)
            ImageDraw.Draw(image).rectangle(box, outline="#ff7300", width=max(3, image.width // 400))
            x = column * 600 + 20
            labels = ", ".join(e.get("label", "") for e in observation.get("labelEvidence", []))
            draw.text((x, 45), oid[:8] + " | " + labels[:35], fill="black", font=font)
            for rendered, y, size in ((image, 85, (560, 560)), (crop, 690, (560, 385))):
                thumb = ImageOps.contain(rendered, size)
                canvas.paste(thumb, (x + (560 - thumb.width) // 2, y))
            draw.text((x, 655), "Exact source crop; original mask retained", fill="black", font=font)
            sources.append({"observationId": oid, "image": asset["file"], "originalPixelBox": box, "cropPixelBox": pixel_box,
                "crop": file_ref(path), "maskAssetId": observation.get("maskAssetId"), "sourceRefs": observation["sourceRefs"]})
        path = output / f"group-{index:02d}.png"; canvas.save(path)
        group.update(sources=sources, reviewImage=file_ref(path), truth="unknown", reviewer=None)
    report = {"document": document_ref, "assetManifest": manifest_ref, "groups": groups,
        "sourceBindingCount": sum(g["kind"] == "existing_source_binding" for g in groups), "requestedComparisonCount": len(comparisons),
        "note": "No identity decisions or human labels are written by this review gallery."}
    (output / "manifest.json").write_bytes(encoded(report) + b"\n")
    print(json.dumps({"sourceGroups": report["sourceBindingCount"], "comparisonGroups": len(comparisons), "manifest": str(output / "manifest.json")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--results", type=Path)
    mode.add_argument("--source-document", type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--comparisons", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.results:
        materialize(args.results, args.output)
    else:
        materialize_groups(args.source_document, args.asset_manifest, args.output, json.loads(args.comparisons.read_bytes()) if args.comparisons else [])
