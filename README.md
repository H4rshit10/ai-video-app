# AI Educational Video Generator

A multi-scene educational video generator built entirely on the Google AI stack.

| Layer | Service |
|---|---|
| Director | Gemini 2.5 Flash (structured output) |
| Visuals | Imagen 4 via Vertex AI (precision) + Veo 3 (narrative, optional) |
| Voice | Google Cloud Text-to-Speech (Neural2, SSML) |
| Assembly | MoviePy (Ken Burns motion, burned-in captions) |
| UI | Streamlit |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

# First-time auth — see docs/superpowers/specs/ for full setup
gcloud auth login
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com texttospeech.googleapis.com

# Run
streamlit run app.py
```

## Try it
Topics that exercise each content-type path:

| Topic | Audience | Hits |
|---|---|---|
| Fractions | 6-year-old | math (Imagen + Ken Burns) |
| Hampi | adult | place (Imagen documentary) |
| The Lion and the Mouse | 10-year-old | story (Veo for narrative, falls back to Imagen if locked) |

## Layout

```
app.py                          Streamlit entrypoint
src/config.py                   env loading, model IDs, constants
src/director.py                 Gemini structured-output scene planner
src/visuals.py                  Imagen + Veo dispatcher with fallback
src/voice.py                    Cloud TTS with SSML
src/assemble.py                 MoviePy stitcher (motion + captions)
src/pipeline.py                 orchestrator + cost telemetry
docs/superpowers/specs/         design spec
NOTES.md                        research log
outputs/<run_id>/               per-run generated assets
```

Full design and research links: [design spec](docs/superpowers/specs/2026-05-13-ai-video-app-design.md) · [research log](NOTES.md)
