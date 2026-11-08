# Presenting roadwatch

## Title
RoadWatch — AI-Assisted Road Damage Inspection

## Description
Implemented a CPU-based road-damage inspection application using a pretrained ONNX detector. Added aspect-preserving preprocessing, class-aware box suppression, human candidate review and image/model provenance. Supported annotated-image, JSON and photo-level GeoJSON exports, plus a held-out-box evaluation utility. Verified inference on a public road image; no independent accuracy or severity claim.

Use the description after running and understanding the app. Do not claim training the pretrained model, company deployment, measured cost/time savings, or benchmark accuracy that was not measured. These are working prototypes with explicit limitations. The implementation was created with AI assistance; state your own contribution accurately and use actual project dates.

## GitHub
Create an empty repository named `roadwatch`. In this folder:
```bash
git init
git add .
git status
```
Review staged files. Exclude `.venv`, caches, private recordings/photos, credentials and local exports. Keep source attribution, source metadata, tests and reports. Then:
```bash
git commit -m "Build evaluated local AI workflow"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/roadwatch.git
git push -u origin main
```
Replace YOUR_USERNAME and authenticate with GitHub's supported flow. Do not force-push over existing work. Choose a code license before inviting reuse, keeping upstream model/data terms separate.

## Interview preparation
Explain the user problem, actual model, preprocessing, evaluation boundary, false positives/recognition errors, human review and one improvement you personally made. Show source-linked evidence and a failure rather than claiming perfect automation. Add screenshots only after capturing the actual app.
