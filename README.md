# AI Educational Video Generator

Type a topic and an audience, get back a short voiced video with motion, captions, and an interactive layer — built entirely on the Google AI stack.

The point of this project was to see how far you can push educational video generation using only Google's APIs: Gemini for the script, Imagen for the visuals, Veo for cinematic clips when you want them, Cloud TTS (Chirp 3 HD) for the voice. No third-party AI anywhere in the runtime.

## What you get

A single Gemini call (the "Director") plans the whole video as a structured object — title, scene list, visual prompts, dialogue, a mid-video checkpoint question, an end-of-video quiz. The pipeline then runs each scene through Imagen (or Veo if you've opted in), Cloud TTS, and MoviePy to produce both a concatenated `final.mp4` and per-scene `.mp4` clips. Streamlit serves it two ways:

- **Linear mode** plays the full assembled video (title card → scenes → outro card) and shows the quiz at the end.
- **Interactive mode** plays scene by scene. When the Director set a checkpoint, the player pauses, asks the question, and won't continue until the viewer answers. After the last scene, the same quiz appears.

Title and outro cards bracket every video. Spoken dialogue is burned in as a subtitle band on the bottom third. A short on-screen headline sits on the upper third for context. There's a 3-second pre-roll ad placeholder up top — the slot where a real Google IMA SDK integration would plug in next.

## Examples

Three real videos generated end-to-end by this pipeline — one per content domain.

| Topic | Audience | Strategy | Cost | Video |
|---|---|---|---|---|
| **Fractions** | 6-year-old | Imagen 4 + Ken Burns motion (math: precision matters) | ~$0.21 | [examples/fractions/](examples/fractions/) |
| **Hampi** | adult | Imagen 4 documentary style + Ken Burns motion (place) | ~$0.21 | [examples/hampi/](examples/hampi/) |
| **The Lion and the Mouse** | 10-year-old | Veo 3 for narrative scenes + Imagen fallback (story) | ~$9.05 | [examples/the-lion-and-the-mouse/](examples/the-lion-and-the-mouse/) |

Each folder contains the rendered `final.mp4`, the `plan.json` driving generation, and `quiz.json` / `checkpoints.json` when applicable.

## Stack

| Layer | Service | Model / detail |
|---|---|---|
| Director | Gemini 2.5 Flash (structured output) | `gemini-2.5-flash` |
| Visuals (precision) | Imagen 4 via Vertex AI | `imagen-4.0-generate-001` |
| Visuals (narrative) | Veo 3 via Vertex AI — opt-in toggle | `veo-3.0-generate-001` |
| Voice | **Google Cloud Text-to-Speech, Chirp 3 HD** (Neural2 silent fallback) | `en-US-Chirp3-HD-Aoede` (math), `Kore` (place), `Leda` (story) |
| Assembly | MoviePy 2.1 — Ken Burns motion + subtitle burn-in + title/outro cards | — |
| UI | Streamlit 1.57 — Linear + Interactive playback modes | — |

**Auth model (hybrid):** AI Studio API key for Gemini on its free tier; Application Default Credentials via `gcloud` for Imagen, Veo, and Cloud TTS through Vertex AI on a single GCP project.

## How it works

```
User input (topic, audience, mode)
        │
        ▼
┌────────────────────────────────────────────┐
│ Director — Gemini 2.5 Flash                │   one call → structured Pydantic plan
│ returns VideoPlan {                         │   with checkpoints + end_quiz
│   scenes[], checkpoints, end_quiz,          │
│   recurring_character, style_guide,         │
│   voice_recommendation                      │
│ }                                           │
└──────────────┬─────────────────────────────┘
               │
   ┌───────────┼────────────┐
   ▼           ▼            ▼
┌────────┐ ┌────────┐ ┌─────────────────┐
│ Imagen │ │  Veo   │ │ Cloud TTS       │   per-scene assets
│   4    │ │   3    │ │ Chirp 3 HD      │
└────┬───┘ └────┬───┘ └────┬────────────┘
     └──────────┴──────────┘
                │
                ▼
        MoviePy assembler
        (title card · Ken Burns · subtitles · outro card)
                │
                ├─── outputs/<run_id>/final.mp4       (full concatenated)
                └─── outputs/<run_id>/scene_*_clip.mp4 (per-scene for Interactive mode)
```

The Director picks the generator **per scene** based on content type — Imagen for math/place (Veo's tendency to morph labels breaks precision); Veo for narrative scenes (real motion shines). When Veo fails on a scene, the pipeline silently falls back to Imagen + motion so the run always completes. The same silent-fallback pattern applies to **Chirp 3 HD → Neural2-C** voice generation.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

# Auth — one-time setup
gcloud auth login
gcloud auth application-default login
gcloud config set project <YOUR_PROJECT_ID>
gcloud auth application-default set-quota-project <YOUR_PROJECT_ID>
gcloud services enable aiplatform.googleapis.com texttospeech.googleapis.com

# Copy the env template and fill in real values
copy .env.example .env

# Launch the UI
streamlit run app.py
```

Open `http://localhost:8501`, type a topic in the sidebar, pick **Interactive** mode, click **Generate video**.

## Layout

```
app.py                          Streamlit UI (Linear + Interactive modes)
src/config.py                   env vars, model IDs, cost constants
src/director.py                 Gemini structured-output planner — senior-director system prompt
src/visuals.py                  Imagen + Veo dispatcher (with silent fallback)
src/voice.py                    Cloud TTS — Chirp 3 HD primary, Neural2 silent fallback
src/assemble.py                 MoviePy stitcher (motion + subtitle burn-in + title/outro cards + per-scene exports)
src/pipeline.py                 orchestrator + cost telemetry
examples/                       three committed demo videos with plans, quizzes, checkpoints
docs/superpowers/specs/         technical design spec
NOTES.md                        research log with Google doc links
```

Per-run output layout:
```
outputs/<run_id>/
├── plan.json                  # full VideoPlan from Director
├── quiz.json                  # end-of-video quiz (if applicable)
├── checkpoints.json           # mid-video checkpoints (if any)
├── title_card.mp4             # 2-sec branded title
├── scene_0.png / scene_0.mp4  # raw Imagen frame or Veo clip
├── scene_0.mp3                # Cloud TTS audio
├── scene_0_clip.mp4           # assembled per-scene clip (Interactive mode)
├── outro_card.mp4             # 2-sec branded outro
└── final.mp4                  # concatenated full video (Linear mode)
```

## Cost

| Mode | Per video |
|---|---|
| Default (Veo OFF — all Imagen) | ~**$0.21** |
| Veo enabled (story content) | ~**$9.05** |

New GCP accounts get $300 free credit for 90 days, comfortably covering hundreds of default runs or ~30 Veo-enabled runs.

## Further reading

- [Design spec](docs/superpowers/specs/2026-05-13-ai-video-app-design.md) — architecture, module breakdown, decisions
- [NOTES.md](NOTES.md) — links to every Google doc consulted during the build
- [examples/README.md](examples/README.md) — guide to the three example outputs
