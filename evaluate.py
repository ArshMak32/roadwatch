import argparse, json
from pathlib import Path
from roadwatch.core import load_image, detect
from roadwatch.metrics import evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "manifest", help="JSON with images [{id,path,objects:[{class_id,box_xyxy}]}]"
    )
    a = p.parse_args()
    path = Path(a.manifest)
    data = json.loads(path.read_text())
    pred = {}
    truth = {}
    for row in data["images"]:
        image = load_image((path.parent / row["path"]).read_bytes())
        pred[row["id"]] = detect(image, confidence=0.001)["detections"]
        truth[row["id"]] = row["objects"]
    result = evaluate(pred, truth)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
