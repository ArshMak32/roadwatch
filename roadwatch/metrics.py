"""Class-aware 101-point AP on user-supplied boxes. No bundled accuracy claim."""

import numpy as np
from roadwatch.core import iou


def average_precision(predictions, ground_truth, class_id, threshold=0.5):
    truth = {}
    total = 0
    for image, objects in ground_truth.items():
        boxes = [x["box_xyxy"] for x in objects if x["class_id"] == class_id]
        truth[image] = {"boxes": boxes, "matched": set()}
        total += len(boxes)
    if not total:
        return None
    selected = sorted(
        [
            (image, p)
            for image, objects in predictions.items()
            for p in objects
            if p["class_id"] == class_id
        ],
        key=lambda x: -x[1]["score"],
    )
    tp = []
    fp = []
    for image, p in selected:
        item = truth.get(image, {"boxes": [], "matched": set()})
        candidates = [
            (iou(p["box_xyxy"], b), j)
            for j, b in enumerate(item["boxes"])
            if j not in item["matched"]
        ]
        best = max(candidates, default=(0, -1))
        if best[0] >= threshold:
            item["matched"].add(best[1])
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not tp:
        return 0.0
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / total
    precision = tp / np.maximum(tp + fp, 1)
    return float(
        np.mean(
            [max(precision[recall >= r], default=0) for r in np.linspace(0, 1, 101)]
        )
    )


def evaluate(predictions, ground_truth):
    classes = sorted({o["class_id"] for row in ground_truth.values() for o in row})
    rows = []
    if not classes:
        raise ValueError("At least one ground-truth object is required")
    for cls in classes:
        ap = [
            average_precision(predictions, ground_truth, cls, float(t))
            for t in np.linspace(0.5, 0.95, 10)
        ]
        rows.append({"class_id": cls, "ap50": ap[0], "ap50_95": float(np.mean(ap))})
    return {
        "images": len(ground_truth),
        "map50": float(np.mean([r["ap50"] for r in rows])),
        "map50_95": float(np.mean([r["ap50_95"] for r in rows])),
        "per_class": rows,
        "note": "101-point class-aware box AP; not a substitute for official COCO/RDD tooling. Classes with no ground-truth instances are excluded from mAP.",
    }
