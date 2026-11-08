import unittest, io
import numpy as np
from PIL import Image
from roadwatch.core import iou, nms, letterbox, decode, geojson, load_image
from roadwatch.metrics import evaluate


class GeometryTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)

    def test_class_aware_nms(self):
        boxes = [[0, 0, 10, 10]] * 3
        self.assertEqual(nms(boxes, [0.9, 0.8, 0.7], [0, 0, 1]), [0, 2])

    def test_letterbox(self):
        tensor, r, pad = letterbox(Image.new("RGB", (1000, 500)))
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(pad, (0, 160))
        self.assertEqual(r, 0.64)

    def test_decode_roundtrip(self):
        out = np.zeros((1, 8, 1), dtype=np.float32)
        out[0, :4, 0] = [320, 320, 320, 160]
        out[0, 7, 0] = 0.9
        d = decode(out, 0.64, (0, 160), (1000, 500))
        self.assertEqual(d[0]["code"], "D40")
        self.assertEqual(d[0]["box_xyxy"], [250, 125, 750, 375])

    def test_empty_detection(self):
        self.assertEqual(decode(np.zeros((1, 8, 3)), 1, (0, 0), (640, 640)), [])

    def test_unexpected_model_shape_rejected(self):
        with self.assertRaises(ValueError):
            decode(np.zeros((1, 84, 3)), 1, (0, 0), (640, 640))

    def test_geojson_coordinate_order(self):
        g = geojson(
            {"image_sha256": "a", "model_sha256": "b", "detections": []}, 29, -98
        )
        self.assertEqual(g["features"][0]["geometry"]["coordinates"], [-98, 29])

    def test_invalid_geo_rejected(self):
        with self.assertRaises(ValueError):
            geojson({}, 95, 1)

    def test_image_read(self):
        b = io.BytesIO()
        Image.new("RGB", (20, 10)).save(b, format="PNG")
        self.assertEqual(load_image(b.getvalue()).size, (20, 10))

    def test_perfect_ap(self):
        truth = {"a": [{"class_id": 0, "box_xyxy": [0, 0, 10, 10]}]}
        pred = {"a": [{"class_id": 0, "box_xyxy": [0, 0, 10, 10], "score": 0.9}]}
        self.assertEqual(evaluate(pred, truth)["map50_95"], 1)

    def test_missed_ap(self):
        self.assertEqual(
            evaluate({"a": []}, {"a": [{"class_id": 0, "box_xyxy": [0, 0, 10, 10]}]})[
                "map50"
            ],
            0,
        )
