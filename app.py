from pathlib import Path
import json, hashlib, io
import pandas as pd
import streamlit as st
from roadwatch.core import load_image, detect, annotate, image_bytes, geojson, source

st.set_page_config(
    page_title="RoadWatch | Inspect, Review, Report", page_icon="🛣️", layout="wide"
)
st.markdown(
    '<style>.block-container{max-width:1400px;padding-top:2rem}h1,h2,h3{letter-spacing:-.03em}div[data-testid="stMetric"]{border:1px solid #3c4653;border-radius:12px;padding:14px}</style>',
    unsafe_allow_html=True,
)
st.caption("ROADWATCH / COMPUTER VISION INSPECTION AID")
st.title("Spot candidates. Confirm the damage.")
st.write(
    "Analyze road photos for four damage categories, review the candidate boxes, and export an inspection packet with optional photo location."
)
st.info(
    "Model detections are inspection candidates. A blank result does not mean the road is safe. Image area and confidence do not measure physical severity."
)
with st.sidebar:
    st.subheader("Inspection settings")
    confidence = st.slider("Detection score cutoff", 0.05, 0.95, 0.30, 0.05)
    overlap = st.slider("Duplicate-box IoU cutoff", 0.1, 0.9, 0.45, 0.05)
    st.caption(
        "Scores are not calibrated probabilities. Changing the cutoff trades missed candidates against false alarms."
    )
    location = st.checkbox("Attach a photo location", False)
    lat_text = st.text_input(
        "Latitude", placeholder="Enter photo latitude", disabled=not location
    )
    lon_text = st.text_input(
        "Longitude", placeholder="Enter photo longitude", disabled=not location
    )
    valid_location = False
    lat = lon = 0.0
    if location:
        try:
            lat = float(lat_text)
            lon = float(lon_text)
            valid_location = -90 <= lat <= 90 and -180 <= lon <= 180
            if not valid_location:
                st.error("Coordinates are outside valid ranges")
        except ValueError:
            st.caption("Enter both coordinates to enable location export.")
    st.caption(
        "Coordinates are manually entered, not inferred. Confirm them before export. No live map requests are made."
    )
inspect, about = st.tabs(["Inspect a photo", "Model & evaluation"])
with inspect:
    upload = st.file_uploader(
        "Road photo — JPG/PNG/WebP, up to 20 MB", type=["jpg", "jpeg", "png", "webp"]
    )
    use_demo = st.checkbox("Use the licensed public road-photo demo", False)
    if use_demo:
        payload = (Path(__file__).resolve().parent / "data/demo_road.jpg").read_bytes()
        upload = io.BytesIO(payload)
        upload.name = "China_Drone_000008.jpg"
        st.caption(
            "Public RDD2022 sample, CC BY-SA 4.0. Attribution and pinned source: data/demo_source.json. A single photo is not an accuracy benchmark."
        )
    if upload is not None:
        try:
            image = load_image(upload.getvalue())
            st.image(image, caption="Original photo", width="stretch")
            if st.button("Detect road damage candidates", type="primary"):
                with st.spinner(
                    "Running the road-damage ONNX model on CPU. First use downloads approximately 12 MB…"
                ):
                    report = detect(image, confidence, overlap)
                    report["image_sha256"] = hashlib.sha256(
                        upload.getvalue()
                    ).hexdigest()
                    report["filename"] = upload.name
                    st.session_state["report"] = report
            report = st.session_state.get("report")
            if (
                report
                and report["image_sha256"]
                == hashlib.sha256(upload.getvalue()).hexdigest()
            ):
                st.image(
                    annotate(image, report["detections"]),
                    caption="Candidate boxes — require human review",
                    width="stretch",
                )
                a, b, c = st.columns(3)
                a.metric("Candidate detections", len(report["detections"]))
                b.metric(
                    "Damage types detected",
                    len({d["code"] for d in report["detections"]}),
                )
                c.metric("Inference pipeline", str(report["latency_ms"]) + " ms")
                if report["detections"]:
                    rows = pd.DataFrame(report["detections"])
                    rows["box_xyxy"] = rows.box_xyxy.astype(str)
                    edits = st.data_editor(
                        rows,
                        hide_index=True,
                        disabled=[x for x in rows.columns if x != "review_status"],
                        column_config={
                            "review_status": st.column_config.SelectboxColumn(
                                options=["candidate", "accepted", "rejected"]
                            )
                        },
                        key="review_"
                        + report["image_sha256"]
                        + "_"
                        + str(report["confidence_threshold"]),
                    )
                    for obj, status in zip(report["detections"], edits.review_status):
                        obj["review_status"] = status
                else:
                    st.warning(
                        "No candidates at this cutoff. This is not a clean-road assessment; review the photo yourself."
                    )
                if location and valid_location:
                    report["photo_location"] = {
                        "latitude": lat,
                        "longitude": lon,
                        "source": "user_entered",
                    }
                else:
                    report.pop("photo_location", None)
                st.download_button(
                    "Inspection report JSON",
                    json.dumps(report, indent=2),
                    "road_inspection.json",
                    "application/json",
                )
                st.download_button(
                    "Annotated photo PNG",
                    image_bytes(annotate(image, report["detections"])),
                    "road_candidates.png",
                    "image/png",
                )
                if location and valid_location:
                    st.download_button(
                        "Photo-location GeoJSON",
                        json.dumps(geojson(report, lat, lon), indent=2),
                        "inspection_location.geojson",
                        "application/geo+json",
                    )
        except Exception as exc:
            st.error(
                "Inspection failed: "
                + type(exc).__name__
                + ". Check the image, model download and dependency versions."
            )
    else:
        st.caption(
            "Use a photo taken safely from a stationary position or an existing licensed road image. This app does not capture images while driving."
        )
with about:
    st.subheader("What the model recognizes")
    st.table(
        pd.DataFrame(
            {
                "Code": ["D00", "D10", "D20", "D40"],
                "Category": [
                    "Longitudinal crack",
                    "Transverse crack",
                    "Alligator crack",
                    "Pothole",
                ],
            }
        )
    )
    st.write(
        "A road-specific pretrained YOLOv8 ONNX detector runs locally on CPU. This project implements preprocessing, class-aware nonmaximum suppression, inspection review and geospatial exports. It does not train a new detector."
    )
    st.json(source())
    st.caption(
        "The upstream model has limited training documentation and conflicting license declarations. No benchmark accuracy or commercial redistribution permission is asserted."
    )
    st.markdown("**Evaluate on your own held-out annotations:**")
    st.code("python evaluate.py path/to/manifest.json", language="bash")
    st.write(
        "The evaluator supports class-aware box AP50 and AP50:95. The manifest format is in README.md. No independent labeled road benchmark is bundled, so no road-detection accuracy score is claimed."
    )
    st.caption(
        "No depth estimation, repair-cost prediction, road-safety certification, automatic GPS extraction or external report submission. Uploaded images are processed in memory; inference cache stores model weights only."
    )
