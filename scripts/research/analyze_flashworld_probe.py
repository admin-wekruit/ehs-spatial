"""Local generation diagnostics; no inference, resampling, or physical-accuracy claims.

  .venv/bin/python scripts/research/analyze_flashworld_probe.py --result-dir <result>
  .venv/bin/python scripts/research/analyze_flashworld_probe.py --self-check
"""

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


DEFAULT = Path(__file__).resolve().parents[2] / "outputs/candidate-evaluation/flashworld-probe-01/result"
GAUSSIAN_FIELDS = {"x", "y", "z", "opacity", *(f"f_dc_{i}" for i in range(3)),
                   *(f"scale_{i}" for i in range(3)), *(f"rot_{i}" for i in range(4))}
REVIEWED_RENDER_HASHES = {
    "source-render.png": "e64e59954dc9811c3f13fc2fd2d2fb8b329ad14225e5c01497f9ff9a38aeffd6",
    "novel-render.png": "4441f00c22848ed105a3a5aba06ce4436ad399f43fb353301a1cd8a33842f1ba",
}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def image_metrics(reference, rendered):
    if reference.shape != rendered.shape:
        return {"status": "not_measured", "reason": "image_shape_mismatch",
                "reference_shape": list(reference.shape), "render_shape": list(rendered.shape),
                "mae_0_255": None, "psnr_db": None}
    delta = reference.astype(np.float64) - rendered.astype(np.float64)
    mse = float(np.mean(delta ** 2))
    return {"status": "measured", "shape_hwc": list(reference.shape),
            "mae_0_255": float(np.mean(np.abs(delta))), "mse_0_255_squared": mse,
            "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse > 0 else None,
            "psnr_note": "positive_infinity_exact_match" if mse == 0 else "peak=255",
            "exact_pixel_match": mse == 0,
            "scope": "All pixels in gamma-encoded RGB; condition.png versus source-render.png; no registration or resize"}


def array_metrics(path, expected_hw, alpha=False):
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False, mmap_mode="r")
    if data.ndim != 2 or (expected_hw is not None and data.shape != expected_hw):
        return {"status": "invalid_shape", "shape": list(data.shape), "expected_hw": expected_hw}
    finite = np.isfinite(data)
    values = data[finite]
    result = {"status": "measured", "shape_hw": list(data.shape), "pixels": int(data.size),
              "finite_fraction": float(finite.mean()), "nonfinite_pixels": int((~finite).sum()),
              "min_finite": float(values.min()) if values.size else None,
              "max_finite": float(values.max()) if values.size else None,
              "mean_finite": float(values.mean(dtype=np.float64)) if values.size else None}
    if alpha:
        result.update({"coverage_gt_0_5_fraction": float((finite & (data > 0.5)).mean()),
                       "coverage_gt_0_95_fraction": float((finite & (data > 0.95)).mean()),
                       "outside_0_1_finite_pixels": int(((values < 0) | (values > 1)).sum()),
                       "coverage_denominator": "all source-render pixels; nonfinite values count as uncovered"})
    else:
        result.update({"positive_finite_fraction": float((finite & (data > 0)).mean()),
                       "units": "normalized model coordinates; native accumulated RGB+D, not meters or surface-depth GT"})
    return result


def ply_header(path):
    """Read at most 64 KiB; Gaussian payload/renderer compatibility remain unverified."""
    if not path.exists():
        return None
    lines, size = [], 0
    with path.open("rb") as stream:
        while size < 65536:
            line = stream.readline(65536 - size)
            if not line:
                break
            size += len(line)
            try:
                lines.append(line.decode("ascii", errors="strict").strip())
            except UnicodeDecodeError:
                return {"status": "invalid_header", "reason": "non_ascii_before_end_header", "bytes": path.stat().st_size}
            if lines[-1] == "end_header":
                break
    if not lines or lines[0] != "ply" or lines[-1] != "end_header":
        return {"status": "invalid_header", "bytes": path.stat().st_size}
    vertex_count, vertex, fields, format_name = None, False, [], None
    for line in lines:
        parts = line.split()
        if parts[:1] == ["format"]:
            format_name = " ".join(parts[1:])
        elif parts[:1] == ["element"]:
            vertex = parts[1] == "vertex"
            if vertex:
                vertex_count = int(parts[2])
        elif parts[:1] == ["property"] and vertex:
            fields.append({"name": parts[-1], "type": " ".join(parts[1:-1])})
    missing = sorted(GAUSSIAN_FIELDS - {f["name"] for f in fields})
    return {"status": "header_read", "bytes": path.stat().st_size, "header_bytes": size,
            "format": format_name, "vertex_count": vertex_count, "vertex_properties": fields,
            "missing_gaussian_fields": missing, "has_required_gaussian_fields": not missing,
            "payload_numeric_validity": None, "external_3d_viewer_compatibility": None,
            "scope": "Header fields only; not full payload validation, geometry accuracy, or independent rendering verification"}


def video_info(path):
    if not path.exists():
        return None
    result = {"bytes": path.stat().st_size, "status": "not_measured"}
    try:
        import cv2
    except ImportError:
        return {**result, "reason": "optional_existing_opencv_unavailable; no dependency installed"}
    capture = cv2.VideoCapture(str(path))
    try:
        opened, frame = capture.read()
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return {**result, "status": "measured" if opened else "decode_failed", "reader": "installed OpenCV",
                "first_frame_decoded": bool(opened), "first_frame_shape_hwc": list(frame.shape) if opened else None,
                "reported_fps": fps if math.isfinite(fps) and fps > 0 else None,
                "reported_frames": int(count) if math.isfinite(count) and count > 0 else None,
                "reported_duration_seconds": count / fps if math.isfinite(fps) and math.isfinite(count) and fps > 0 and count > 0 else None,
                "scope": "Container-reported counts and first-frame decode; not all-frame verification"}
    finally:
        capture.release()


def protocol_table(run_id, baseline):
    if baseline is not None and (run_id is None or baseline.get("run_id") != run_id):
        raise ValueError("Refusing cross-run or unknown-run baseline comparison")
    fields = [
        ("reprojection.worst_frac", "原协议回投 worst / 图高"),
        ("reprojection.median_frac", "原协议回投 median / 图高"),
        ("geometry.collinearity_max_residual_m", "原协议共线残差 / 模型单位标称 m"),
        ("geometry.parallel_max_spread_deg", "原协议平行角差 / °"),
        ("geometry.manhattan_max_off_axis_deg", "原协议 Manhattan 偏轴 / °"),
        ("inventory.object_count", "已有 inventory 对象数（不等于召回率）"),
        ("ground_truth.entity_recall", "真实实体召回率"),
        ("ground_truth.physical_scale_error", "实测物理尺度误差"),
        ("ground_truth.physical_accuracy", "物理空间准确率"),
        ("protocol.complete_paired_ab", "原协议完整同 run A/B"),
    ]
    rows = []
    for metric_id, label in fields:
        value = baseline
        for part in metric_id.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        rows.append({"run_id": run_id, "metric_id": metric_id, "metric": label, "baseline": value,
                     "flashworld": None, "delta": None, "status": "not_measured_for_generated_scene"})
    return rows


def analyze(directory):
    if not directory.is_dir() or not (directory / "metadata.json").exists():
        raise FileNotFoundError("Actual result/metadata.json is required; no results or placeholder scores written")
    metadata = read_json(directory / "metadata.json")
    weights = read_json(directory / "weights-manifest.json") or {}
    export = read_json(directory / "export-validation.json")
    source = read_json(directory / "source-manifest.json") or {}
    run_id = source.get("run_id")
    baseline = read_json(directory.parent / "baseline-metrics.json")
    images = {}
    for name in ("source.png", "condition.png", "source-render.png", "novel-render.png"):
        if (directory / name).exists():
            with Image.open(directory / name) as image:
                images[name] = np.asarray(image.convert("RGB"))
    fidelity = image_metrics(images["condition.png"], images["source-render.png"]) if all(
        name in images for name in ("condition.png", "source-render.png")) else {
            "status": "not_measured", "reason": "condition_or_source_render_missing", "mae_0_255": None, "psnr_db": None}
    expected = images["source-render.png"].shape[:2] if "source-render.png" in images else None
    recorded_hash = source.get("source_sha256", metadata.get("source_sha256"))
    source_hash = hashlib.sha256((directory / "source.png").read_bytes()).hexdigest() if (directory / "source.png").exists() else None
    assets = ["viewer.html", "viewer-validation.json", "source.png", "condition.png", "source-render.png", "novel-render.png", "trajectory.mp4",
              "gaussians.ply", "gaussians.spz", "source-alpha.npy", "source-depth.npy", "input.json",
              "metadata.json", "source-manifest.json", "weights-manifest.json", "runtime.log", "pip-freeze.txt",
              "export-validation.json", "corrected-gaussians.ply", "corrected-gaussians.spz"]
    reviewed = all((directory / name).exists() and hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
                   for name, digest in REVIEWED_RENDER_HASHES.items())
    observations = ["源机位仍能辨认红色机器人和主体布局。",
                    "细杆、透明围栏和纹理出现明显变形，不能据此确认间距或细小结构。",
                    "末端视角右侧出现大量重复黄黑控制件；没有真实新视角 GT，不能确认现场存在这些部件。"] if reviewed else []
    peak = metadata.get("peak_cuda_allocated_bytes")
    return {"run_id": run_id, "native_job_status": metadata.get("status"),
            "scope": "Single-image generated-scene diagnostics; not spatial accuracy or full geometry-slot A/B",
            "heldout_real_novel_view_gt": False, "physical_scale_known": False,
            "source_camera_registration": metadata.get("source_camera_registration", "unknown"),
            "runtime": {"hardware": metadata.get("hardware"),
                        "cpu_weight_prepare_seconds": weights.get("prepare_seconds"),
                        "model_load_seconds": metadata.get("model_load_seconds"),
                        "native_generate_seconds": metadata.get("native_generate_seconds"),
                        "gpu_function_seconds": metadata.get("gpu_function_seconds"),
                        "peak_cuda_allocated_bytes": peak,
                        "peak_cuda_allocated_gib": peak / 1024 ** 3 if peak is not None else None,
                        "gaussian_count": metadata.get("gaussian_count"),
                        "spherical_harmonics_degree": metadata.get("spherical_harmonics_degree"),
                        "native_render_crosscheck_max_abs": metadata.get("source_render_crosscheck_max_abs"),
                        "crosscheck_scope": "Two invocations of the same native renderer; not input-image error or independent browser rendering"},
            "export_validation": export,
            "viewer_validation": read_json(directory / "viewer-validation.json"),
            "visual_review": {"render_hashes_matched": reviewed, "observations": observations,
                              "provenance": "Root agent visual inspection of the exact SHA-bound render files"},
            "source_sha256": source_hash,
            "source_hash_matches_manifest": source_hash == recorded_hash if source_hash and recorded_hash else None,
            "source_fidelity": fidelity, "alpha": array_metrics(directory / "source-alpha.npy", expected, alpha=True),
            "depth": array_metrics(directory / "source-depth.npy", expected),
            "ply": ply_header(directory / "gaussians.ply"), "video": video_info(directory / "trajectory.mp4"),
            "same_run_protocol_table": protocol_table(run_id, baseline),
            "preprocess": metadata.get("preprocess"),
            "assets": [{"file": name, "bytes": (directory / name).stat().st_size if (directory / name).exists() else None} for name in assets]}


def make_html(data):
    def show(value):
        return "未测 / null" if value is None else escape(f"{value:.6g}" if isinstance(value, float) else str(value))

    available = {asset["file"] for asset in data["assets"] if asset["bytes"] is not None}
    pictures = "".join(f'<figure><img src="{name}" alt="{label}"><figcaption>{label}</figcaption></figure>'
                       if name in available else f'<figure><p>产物缺失</p><figcaption>{label}</figcaption></figure>'
                       for name, label in [("source.png", "原始 canonical 输入"), ("condition.png", "官方裁剪后的条件图"),
                                           ("source-render.png", "生成场景：源相机渲染"), ("novel-render.png", "生成场景：轨迹末端，无真实新视角 GT")])
    links = " · ".join(f'<a href="{escape(a["file"])}">{escape(a["file"])}</a>' for a in data["assets"] if a["file"] in available)
    rows = "".join(f'<tr><td>{escape(row["metric"])}</td><td>{show(row["baseline"])}</td><td>未测 / null</td><td>未测 / null</td></tr>' for row in data["same_run_protocol_table"])
    fidelity, alpha = data["source_fidelity"], data["alpha"] or {}
    runtime = data["runtime"]
    runtime_rows = "".join(f'<tr><td>{label}</td><td>{show(runtime.get(key))}</td></tr>' for key, label in [
        ("cpu_weight_prepare_seconds", "公共权重 CPU 预备 / s（单独记录）"),
        ("model_load_seconds", "模型加载 / s"), ("native_generate_seconds", "原生生成 / s"),
        ("gpu_function_seconds", "GPU 函数总耗时 / s（含加载、渲染和导出）"),
        ("peak_cuda_allocated_gib", "PyTorch CUDA 峰值已分配内存 / GiB"),
        ("gaussian_count", "Gaussian 数量"), ("spherical_harmonics_degree", "球谐阶数"),
        ("native_render_crosscheck_max_abs", "原生渲染交叉核对：RGB 最大绝对差 / 0–1")])
    observations = data["visual_review"]["observations"]
    review = '<ul>' + ''.join(f'<li>{escape(item)}</li>' for item in observations) + '</ul>' if observations else '<p>尚未登记这组渲染的具体图像观察。</p>'
    export = data["export_validation"] or {}
    viewer = data.get("viewer_validation") or {}
    preview = '<p><a href="viewer.html">打开可拖动的 3D 空间预览 →</a>（SPZ 有损压缩）</p>' if "viewer.html" in available else ""
    viewer_note = "独立本地查看器已验证加载、拖动和回原机位；尚未测浏览器与原生渲染的像素一致性，也未接入产品查看器。" if viewer.get("status") == "interactive_preview_verified" else "浏览器独立渲染及产品查看器兼容性尚未验证。"
    exact = export.get("ply_exact_roundtrip", {})
    roundtrip_ok = export.get("status") == "complete" and all(exact.get(key) is True for key in ["positions", "scales", "rotations", "alphas", "colors", "sh"])
    export_note = ("CPU 数据往返已验：校正 PLY 的位置、尺度、旋转、透明度、颜色及球谐六组数组，可无损往返到原生 PLY 恢复的数组；这不等于回到导出前的 GPU 张量。SPZ 为有损压缩，量化误差已记录。" if roundtrip_ok else
                   "CPU 导出数据往返尚未全部验证；已存在的校正资产不自动视为验证通过。")
    psnr = "∞（完全相同；JSON 用 null + note 表示）" if fidelity.get("exact_pixel_match") else show(fidelity.get("psnr_db"))
    video = '<video controls preload="metadata" src="trajectory.mp4"></video>' if "trajectory.mp4" in available else '<p>轨迹视频尚未生成。</p>'
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlashWorld 单图空间生成诊断</title><style>
body{{font:16px/1.65 system-ui,sans-serif;max-width:1250px;margin:auto;padding:28px;color:#18252e;background:#f5f7f8}}
h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}a{{color:#075b9b}}.note{{padding:16px;background:#fff3d6;border-left:4px solid #b07710}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}figure{{margin:0;background:white;padding:12px}}img,video{{width:100%;max-height:480px;object-fit:contain;background:#e5e9eb}}figcaption{{margin-top:8px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce3e7}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:14px}}.assets{{overflow-wrap:anywhere}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}body{{padding:16px}}table{{font-size:13px}}}}
</style><h1>FlashWorld：同一工位的生成空间能保留多少原图信息？</h1>
<p>工位 {show(data['run_id'])} · 原生作业状态 {show(data['native_job_status'])} · 输入 SHA 一致 {show(data['source_hash_matches_manifest'])}</p>
<p class="note">这是单图生成保真和产物完整性诊断，不是空间准确率。没有真实新视角 GT，也没有物理尺寸真值。生成的遮挡后方和新物体仍是待验证假设，不能进入真实证据或安全判定。</p>
{preview}<div class="grid">{pictures}</div><h2>源视角保真：条件图与源相机渲染</h2>
<p>只比较同尺寸、未经额外配准或缩放的两张图。所有像素参与，采用 gamma 编码 RGB；不把输入与自身的零误差当作基线。配准状态：{show(data['source_camera_registration'])}。</p>
<table><tr><th>诊断</th><th>实际值</th></tr><tr><td>MAE / 0–255</td><td>{show(fidelity.get('mae_0_255'))}</td></tr><tr><td>PSNR / dB</td><td>{psnr}</td></tr><tr><td>alpha &gt; 0.5 覆盖率</td><td>{show(alpha.get('coverage_gt_0_5_fraction'))}</td></tr><tr><td>alpha &gt; 0.95 覆盖率</td><td>{show(alpha.get('coverage_gt_0_95_fraction'))}</td></tr><tr><td>alpha finite 比例</td><td>{show(alpha.get('finite_fraction'))}</td></tr></table>
<h2>生成轨迹</h2>{video}<p>相机平移是模型坐标单位，不能称为现场测距。视频能播放不代表补全几何正确。</p>
<h2>实际观察与可用范围</h2>{review}<p>可用于工位空间预览，以及提出补拍方向或遮挡后方的待验证假设；这些假设需要真实照片确认。不能自动补安全证据、测量现场距离或产生安全判定。</p>
<h2>本次实际运行</h2><p>硬件：{show((runtime.get('hardware') or {}).get('gpu'))}。耗时和内存来自本次记录，CPU 权重预备与 GPU 生成分开。CUDA 内存值是 PyTorch 已分配峰值，并非整卡所有进程的显存峰值。</p>
<table><tr><th>运行项</th><th>实际值</th></tr>{runtime_rows}</table><p>原生 render crosscheck 只核对相同原生渲染器的两次调用；不是源照片误差，也不是浏览器独立渲染验证。</p>
<h2>同 run、同指标的原协议对照</h2><p>基线字段来自同一 run 的已有结果，仅作原协议背景；其内部几何一致性也不等于真实测量准确率。FlashWorld 尚未配准并接入这些指标，因此没有可报告的差值。</p>
<table><tr><th>指标</th><th>既有基线</th><th>FlashWorld</th><th>差值</th></tr>{rows}</table>
<h2>原生资产与导出校验</h2><p>{export_note}校正 PLY 坐标：{show(export.get('corrected_ply_coordinates'))}；校正 SPZ 坐标：{show(export.get('corrected_spz_coordinates'))}。具体变换和误差见 export-validation.json；仍没有物理单位标定或源相机配准。{viewer_note}不能改后缀当作 GLB 使用。</p><p class="assets">{links} · <a href="analysis.json">analysis.json</a></p>
<details><summary>裁剪、finite、PLY 字段、导出往返和视频信息</summary><pre>{escape(json.dumps({key: data[key] for key in ['preprocess', 'source_fidelity', 'alpha', 'depth', 'ply', 'video', 'export_validation']}, ensure_ascii=False, indent=2, allow_nan=False))}</pre></details></html>'''


def self_check():
    left = np.zeros((2, 2, 3), dtype=np.uint8)
    scored = image_metrics(left, left + 10)
    assert scored["mae_0_255"] == 10
    assert abs(scored["psnr_db"] - 28.1308036086791) < 1e-10
    assert image_metrics(left, left)["psnr_note"] == "positive_infinity_exact_match"
    assert image_metrics(left, left[:1])["status"] == "not_measured"
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        np.save(directory / "alpha.npy", [[1.0, 0.96], [0.5, np.nan]])
        alpha = array_metrics(directory / "alpha.npy", (2, 2), alpha=True)
        assert alpha["finite_fraction"] == 0.75 and alpha["coverage_gt_0_5_fraction"] == 0.5
        assert alpha["coverage_gt_0_95_fraction"] == 0.5
        header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        header += "".join(f"property float {field}\n" for field in sorted(GAUSSIAN_FIELDS)) + "end_header\n"
        (directory / "gaussians.ply").write_bytes(header.encode() + bytes(2 * 4 * len(GAUSSIAN_FIELDS)))
        assert ply_header(directory / "gaussians.ply")["has_required_gaussian_fields"]
        try:
            analyze(directory)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing real metadata must not produce a report")
        (directory / "metadata.json").write_text('{"status":"synthetic_self_check"}')
        (directory / "source-manifest.json").write_text('{"run_id":"self-check"}')
        Image.fromarray(left).save(directory / "condition.png")
        Image.fromarray(left + 10).save(directory / "source-render.png")
        data = analyze(directory)
        assert data["source_fidelity"]["mae_0_255"] == 10
        assert 'condition.png' in make_html(data) and '未测 / null' in make_html(data)
        json.dumps(data, allow_nan=False)
    assert all(row["flashworld"] is None and row["delta"] is None for row in protocol_table("test", {"run_id": "test"}))
    print("self-check passed: known pixels, PSNR, shape rejection, finite coverage, PLY header, missing-result gate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    data = analyze(args.result_dir)
    (args.result_dir / "analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.result_dir / "index.html").write_text(make_html(data))
    print(json.dumps({"analysis": str(args.result_dir / "analysis.json"), "page": str(args.result_dir / "index.html"),
                      "source_fidelity_status": data["source_fidelity"]["status"]}))


if __name__ == "__main__":
    main()
