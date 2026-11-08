"""CPU ONNX road-damage detection and explicitly reviewable inspection records."""

from pathlib import Path
from functools import lru_cache
import ast, json, hashlib, urllib.request, tempfile, io
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    "D00": "Longitudinal crack",
    "D10": "Transverse crack",
    "D20": "Alligator crack",
    "D40": "Pothole",
}
COLORS = {"D00": "#5de4b5", "D10": "#ffce63", "D20": "#ad9eff", "D40": "#ff7685"}


def source():
    return json.loads((ROOT / "data/model_source.json").read_text())


def ensure_model():
    config = source()
    folder = ROOT / ".cache"
    folder.mkdir(exist_ok=True)
    path = folder / "road_damage.onnx"
    if (
        path.exists()
        and hashlib.sha256(path.read_bytes()).hexdigest() == config["sha256"]
    ):
        return path
    url = f"https://huggingface.co/{config['repo']}/resolve/{config['revision']}/{config['file']}"
    with tempfile.NamedTemporaryFile(dir=folder, delete=False, suffix=".download") as f:
        temp = Path(f.name)
    try:
        urllib.request.urlretrieve(url, temp)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != config["sha256"]:
            raise ValueError("Downloaded model checksum does not match pinned artifact")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


@lru_cache(maxsize=1)
def session():
    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    opt.inter_op_num_threads = 1
    model = ort.InferenceSession(
        str(ensure_model()), sess_options=opt, providers=["CPUExecutionProvider"]
    )
    meta = model.get_modelmeta().custom_metadata_map
    names = ast.literal_eval(meta.get("names", "{}"))
    if (
        names != {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
        or meta.get("task") != "detect"
    ):
        raise ValueError("Unexpected model classes/task")
    return model


def load_image(data):
    if not 0 < len(data) <= 20 * 1024 * 1024:
        raise ValueError("Image must be 1 byte to 20 MB")
    image = Image.open(io.BytesIO(data))
    if image.width * image.height > 30_000_000:
        raise ValueError("Maximum 30 megapixels")
    image.load()
    return ImageOps.exif_transpose(image).convert("RGB")


def letterbox(image, size=640):
    w, h = image.size
    ratio = min(size / w, size / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    return tensor, ratio, (left, top)


def iou(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    inter = float(np.prod(np.maximum(0, rb - lt)))
    area_a = float(np.prod(np.maximum(0, a[2:] - a[:2])))
    area_b = float(np.prod(np.maximum(0, b[2:] - b[:2])))
    return inter / max(area_a + area_b - inter, 1e-12)


def nms(boxes, scores, classes, threshold=0.45, max_detections=200):
    order = np.argsort(-np.asarray(scores), kind="stable").tolist()
    kept = []
    while order and len(kept) < max_detections:
        idx = order.pop(0)
        kept.append(idx)
        order = [
            j
            for j in order
            if classes[j] != classes[idx] or iou(boxes[idx], boxes[j]) <= threshold
        ]
    return kept


def decode(output, ratio, pad, original_size, confidence=0.3, nms_iou=0.45):
    if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("Thresholds must be 0..1")
    a = np.asarray(output)
    if a.ndim != 3 or a.shape[0] != 1 or a.shape[1] != 8:
        raise ValueError("Expected YOLOv8 detection output [1,8,N]")
    a = a[0].T
    classes = a[:, 4:].argmax(axis=1)
    scores = a[np.arange(len(a)), classes + 4]
    keep = (
        (scores >= confidence)
        & np.isfinite(a).all(axis=1)
        & (a[:, 2] > 0)
        & (a[:, 3] > 0)
    )
    a = a[keep]
    classes = classes[keep]
    scores = scores[keep]
    if not len(a):
        return []
    boxes = np.column_stack(
        [
            a[:, 0] - a[:, 2] / 2,
            a[:, 1] - a[:, 3] / 2,
            a[:, 0] + a[:, 2] / 2,
            a[:, 1] + a[:, 3] / 2,
        ]
    )
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
    w, h = original_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    result = []
    for idx in nms(boxes, scores, classes, nms_iou):
        box = boxes[idx]
        code = f"D{[0, 10, 20, 40][int(classes[idx])]:02d}"
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        result.append(
            {
                "id": f"d{len(result) + 1:03}",
                "class_id": int(classes[idx]),
                "code": code,
                "label": DESCRIPTIONS[code],
                "score": float(scores[idx]),
                "box_xyxy": [round(float(x), 2) for x in box],
                "image_area_fraction": float(
                    (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
                ),
                "review_status": "candidate",
            }
        )
    return result


def detect(image, confidence=0.3, nms_iou=0.45):
    import time

    start = __import__("time").perf_counter()
    tensor, ratio, pad = letterbox(image)
    m = session()
    output = m.run(None, {m.get_inputs()[0].name: tensor})[0]
    detections = decode(output, ratio, pad, image.size, confidence, nms_iou)
    return {
        "detections": detections,
        "width": image.width,
        "height": image.height,
        "confidence_threshold": confidence,
        "nms_iou": nms_iou,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "model_sha256": source()["sha256"],
        "note": "Candidate image detections only; not physical severity, repair priority or road-safety certification.",
    }


def annotate(image, detections):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for d in detections:
        box = d["box_xyxy"]
        color = COLORS[d["code"]]
        draw.rectangle(box, outline=color, width=max(2, round(image.width / 300)))
        text = d["id"] + " " + d["code"] + " " + f"{d['score']:.2f}"
        x, y = box[0], max(0, box[1] - 16)
        draw.rectangle((x, y, x + len(text) * 7, y + 16), fill="#101724")
        draw.text((x + 2, y + 1), text, fill=color)
    return result


def image_bytes(image):
    b = io.BytesIO()
    image.save(b, format="PNG")
    return b.getvalue()


def geojson(report, latitude, longitude):
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Invalid latitude/longitude")
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                "properties": {
                    "image_sha256": report["image_sha256"],
                    "model_sha256": report["model_sha256"],
                    "candidate_count": len(report["detections"]),
                    "reviewed_accepted": sum(
                        d.get("review_status") == "accepted"
                        for d in report["detections"]
                    ),
                    "location_source": "user_entered_photo_location; not per-defect GPS",
                    "note": "Photo-level point only; no defect geometry or physical severity estimated",
                },
            }
        ],
    }
