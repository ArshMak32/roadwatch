# RoadWatch model and testing boundary

## Task and model
Localize candidate road-damage boxes in an image. Uses the pinned ONNX artifact from zainab-nayeem/roadscan-ai-model, revision 32bcf3ec2ebc5c38541667e6eaf475a8c8726cfe. Embedded model metadata describes an Ultralytics YOLOv8 detection model trained on /kaggle/working/rdd.yaml, with D00/D10/D20/D40 labels. Full source training data, splits and metrics are not documented by the uploader. Metadata does not independently validate training provenance.

No performance numbers from another model are substituted for this model. RoadWatch did not train this network. It implements local inference, preprocessing/postprocessing, review and reporting around the public model.

## Source and license uncertainty
The repository card says MIT; the ONNX metadata says AGPL-3.0 and links Ultralytics licensing. The source file is downloaded separately, hash pinned and not redistributed inside this package. Resolve contradictory declarations before commercial redistribution or deployment. This document does not determine applicable legal rights.

## Measured checks
A blank synthetic image produced no candidates at cutoff 0.30. A single public RDD2022 photo produced a D10 candidate at cutoff 0.25. That establishes that inference and coordinate decoding execute; it does not estimate precision, recall, AP, domain robustness or safety. Source sample and output are recorded in reports/real_photo_smoke.json. Unknown overlap with model training cannot be excluded.

## Major limits
Shadows, patches, lane markings, rain, camera angle, resolution, country and image quality can cause errors. Potholes or cracks may be missed. Confidence is not probability of physical danger. Image-area fraction is not actual size or severity. No depth, geolocation inference, defect tracking, cost estimate or safety assessment is implemented. Human acceptance/rejection is a local review annotation, not an engineering certification.

The app is a local single-user prototype. No user authentication, road agency integration, field validation, mobile capture, multi-user record store, benchmark training pipeline or production deployment is provided. The evaluator only measures supplied boxes under its documented contract.
