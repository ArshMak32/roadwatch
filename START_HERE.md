# Run RoadWatch

Move this `roadwatch` folder to Desktop. In Terminal, stop your current app with Control+C, then:
```bash
cd ~/Desktop/roadwatch
bash run.sh
```
Open http://localhost:8506 . Requires Python 3.11 or 3.12. The script creates an environment and installs dependencies.

Enable the public road-photo demo checkbox and click Detect road damage candidates. First use downloads ~12 MB. Inspect the box, review status and exports. No API key is needed. For the bundled sample, leave location disabled unless you know its real coordinates.

Manual start:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8506
```

Screenshot: title + annotated image + candidate table. Keep the inspection-candidate caveat visible. Do not label a model score as accuracy or claim severity/road safety from the result.
