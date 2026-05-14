# AI Video App — Design Spec

**Author:** Harshit
**Date:** 2026-05-13
**Status:** V1 shipped, v1.1 + v1.2 live

---

## 1. What I'm building

A small app that takes a topic (say *fractions* or *Hampi* or *the Lion and the Mouse*) and turns it into a short voiced video a learner can watch and quiz themselves on. I wanted to see how far I could push this using only the Google AI stack — no OpenAI, no ElevenLabs, no scraping. The whole thing runs locally with one `streamlit run` and produces an mp4 with motion, narration, captions, and an optional interactive layer (mid-video checkpoint, end-of-video quiz).

The brief I gave myself was deliberately broad — three categorically different content types (math, place, story) on the same architecture, with audio that doesn't sound robotic, and enough cost awareness that a stranger could fork the repo without bankrupting themselves on their first run.

A few things I wanted to prove out:

- **Stack discipline.** Every layer is a Google service. The Director is Gemini 2.5 Flash with structured output. The frames are Imagen 4 (via Vertex AI). Optional cinematic video clips come from Veo 3. The voice is Cloud Text-to-Speech with the new Chirp 3 HD family. No third-party APIs anywhere in the runtime.
- **Generalisation across content.** The same pipeline has to handle a math diagram (where precision is non-negotiable), a documentary-style historical place, and a children's fable. Each calls for a different visual style, voice, and motion grammar. I let the Director (Gemini) decide all of that per scene.
- **An interactive layer.** A linear video is fine, but the moment you bake in a mid-video comprehension checkpoint and a summative quiz at the end, the architecture has to support state that survives across scene boundaries. That's a meaningfully different problem from "render an mp4."
- **Cost honesty.** Every run prints a cost breakdown. Default mode (Imagen everywhere) lands at ~$0.20 per video; turning on Veo for narrative scenes pushes it to ~$9. I made Veo opt-in so the default never surprises anyone.

---

## 2. Goals and non-goals

### V1 goals
- One-command (`streamlit run app.py`) launches a working UI.
- Generate a multi-scene educational video for **fractions** (math), **Hampi** (place), and **"The Lion and the Mouse"** (Panchatantra story) — three categorically different content types proven on the same architecture.
- Use only the Google AI ecosystem: Gemini, Imagen, Veo (if available), Cloud Text-to-Speech.
- Visible cost telemetry per run.
- Captions burned into the final video for accessibility.
- Research log (`NOTES.md`) with links to every Google doc consulted.

### Non-goals (V1)
- Multi-language UI (only English narration for V1; voice language is selectable but UI is English).
- User accounts / persistence.
- Background music (Lyria) — possible V2 addition.
- Mobile-optimized UI.
- Production deployment / hosting.
- True streaming generation (V1 generates the whole video then plays it back).

### Stretch goals (only if V1 is solid)
- Veo 3 for narrative scenes (gated; falls back gracefully to Imagen + motion).
- SSML pause/emphasis tuning per scene.
- Lyria background music.

---

## 3. Architecture overview

### High-level data flow

```
┌────────────────────────────────────────────────────┐
│  app.py — Streamlit UI                             │
│  inputs: topic, audience, content_type, voice      │
└────────────────────┬───────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────┐
│  src/pipeline.py — orchestrator                    │
│  • mkdir outputs/<run_id>/                         │
│  • call director → assets → voice → assemble       │
│  • collect cost telemetry                          │
└────────────────────┬───────────────────────────────┘
                     │
        ┌────────────┼─────────────┬─────────────┐
        ▼            ▼             ▼             ▼
┌──────────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐
│ director.py  │ │visuals.py│ │ voice.py │ │assemble.py│
│ Gemini       │ │Imagen +  │ │Cloud TTS │ │MoviePy   │
│ structured   │ │Veo       │ │Neural2 / │ │Ken Burns │
│ output →     │ │dispatch  │ │Chirp HD  │ │+ captions│
│ List[Scene]  │ │per scene │ │+ SSML    │ │+ sync    │
└──────────────┘ └─────────┘ └──────────┘ └──────────┘
                     │            │             │
                     └────────────┴─────────────┘
                              │
                              ▼
                     outputs/<run_id>/final.mp4
```

### Authentication model (Hybrid — Option C)

| Service | Auth path | Notes |
|---|---|---|
| Gemini (Director) | AI Studio API key in `.env` | `google-genai` SDK, free tier |
| Imagen (frames) | ADC via gcloud + Vertex AI mode | `google-genai` SDK with `vertexai=True`. AI Studio's Imagen requires a separate paid-plan upgrade; Vertex uses our existing GCP billing. |
| Veo (narrative video) | ADC via gcloud + Vertex AI mode | Gated; runtime fallback to Imagen if quota denied |
| Cloud TTS | ADC via gcloud | `google-cloud-texttospeech` |

Rationale: only Gemini stays on AI Studio (free tier covers our usage). Imagen, Veo, and Cloud TTS all go through Vertex AI / ADC against our GCP project. This is a clean split — one API key for the Director, one set of ADC credentials for everything that generates media. Production migration would move Gemini to Vertex AI too for unified IAM and audit logging.

---

## 4. Module specifications

### `src/config.py`
Pure config — no I/O beyond loading `.env`. Single source of truth for model IDs and constants.

Exports:
- `GEMINI_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (loaded from env, defaults applied)
- `DIRECTOR_MODEL` = `"gemini-2.5-flash"` (verified working 2026-05-13)
- `IMAGEN_MODEL` = `"imagen-4.0-generate-001"` (Vertex AI; updated 2026-05-13 — AI Studio's Imagen 3 endpoint required paid upgrade, Vertex's Imagen 4 works on our existing billing)
- `VEO_MODEL` = `"veo-3.0-generate-001"`
- `TTS_VOICE_DEFAULT` = `"en-US-Neural2-C"`
- `TTS_LANGUAGE_DEFAULT` = `"en-US"`
- `OUTPUT_DIR` = `Path("outputs")`
- `IMAGE_WIDTH` = 1920, `IMAGE_HEIGHT` = 1080
- Cost constants: `COST_PER_IMAGEN`, `COST_PER_TTS_CHAR`, `COST_PER_VEO_SECOND`
- `validate_config()` → raises if required env vars missing

### `src/director.py`
Gemini structured-output planner. One call per video.

```python
class Scene(BaseModel):
    scene_index: int
    visual_prompt: str         # what the image/video should look like
    generator: Literal["imagen", "veo"]
    audio_dialogue: str        # narrator's words (plain text)
    on_screen_text: str | None # short caption, optional
    duration_seconds: float    # target duration

class VideoPlan(BaseModel):
    title: str
    content_type: Literal["math", "place", "story", "general"]
    scenes: List[Scene]        # 3–7 scenes
```

Function: `plan_video(topic: str, audience: str, content_type: str | None = None) -> VideoPlan`
- Auto-detects `content_type` via Gemini if not provided.
- Per-scene `generator` chosen by content type:
  - `math` → all `imagen` (precision; Veo morphs labels)
  - `place` → all `imagen` (high-res stills are documentary-grade)
  - `story` → narrative scenes `veo`, title/outro `imagen`
  - `general` → all `imagen` (safe default)
- System prompt enforces: educational tone, audience-appropriate language, ~5–8 sec per scene, clear visuals.

### `src/visuals.py`
Dispatches per-scene generation. Sync (V1) — parallelize in V2.

```python
def generate_visual(scene: Scene, run_dir: Path) -> Path:
    """Returns path to .png (imagen) or .mp4 (veo)."""
```

Imagen path: `google-genai` `Client(vertexai=True, project=..., location=...).models.generate_images(model=IMAGEN_MODEL, prompt=..., config={"number_of_images": 1, "aspect_ratio": "16:9", "person_generation": "allow_adult"})`. Auth via ADC. Save as `scene_<index>.png`.

Veo path: Vertex client mode (`Client(vertexai=True, project=..., location=...)`). Long-running operation; poll until done, download .mp4. Save as `scene_<index>.mp4`. **On any error (quota, access denied, timeout), fall back to Imagen + a `fallback_used=True` flag in the returned metadata.** This keeps demos resilient if Veo isn't enabled on the account.

### `src/voice.py`
Cloud TTS, one call per scene (so we can match audio precisely to scene durations).

```python
def synthesize_scene_audio(text: str, voice: str, lang: str, out_path: Path) -> Path:
    """Returns path to .mp3."""
```

Uses `google-cloud-texttospeech` with `Neural2-C` (proven working 2026-05-13). Optional SSML wrapping for pauses between sentences. Returns duration via `mutagen` or by re-reading the file with `moviepy.AudioFileClip` — pick one approach in implementation.

### `src/assemble.py`
MoviePy stitcher. Per-scene assembly then concat.

For each scene:
- If asset is `.png`: build `ImageClip` sized to 1920×1080, apply Ken Burns (slow zoom from 1.0 → 1.05 over the scene duration).
- If asset is `.mp4` (Veo): use `VideoFileClip` directly.
- Add `TextClip` for `on_screen_text` (bottom third, semi-transparent box, large readable font).
- Set scene duration = associated audio duration (clip or pad as needed).
- Attach the scene's audio.

Concatenate scenes with 200ms crossfade. Write final `.mp4` at 24 fps, 1920×1080, H.264.

### `src/pipeline.py`
Orchestrator with cost tracking.

```python
@dataclass
class RunResult:
    run_id: str
    final_video: Path
    plan: VideoPlan
    cost_estimate_usd: float
    elapsed_seconds: float
    veo_fallback_used: bool

def generate_video(topic, audience, content_type=None) -> RunResult:
    run_id = timestamp + slug(topic)
    run_dir = OUTPUT_DIR / run_id
    # 1) Director
    plan = director.plan_video(...)
    # 2) For each scene: visual + voice (sync for V1; parallel V2)
    # 3) Assemble
    # 4) Total cost = imagen_count * COST_PER_IMAGEN + tts_chars * COST_PER_TTS_CHAR + veo_seconds * COST_PER_VEO_SECOND
    return RunResult(...)
```

### `app.py`
Streamlit UI. Minimal, focused.

Layout:
- Sidebar: topic input, audience selector (preset list: "6-year-old / 12-year-old / adult"), content type selector ("auto-detect / math / place / story"), voice selector, "Generate" button.
- Main pane:
  - Section 1: "Plan" — show the JSON returned by the Director as it streams.
  - Section 2: "Generating assets" — per-scene status (✓ or ⏳).
  - Section 3: "Final video" — embedded video player + download button + cost summary line.
- Footer: link to `NOTES.md`.

---

## 5. Per-content-type strategies

Director enforces these rules in its system prompt and via the `generator` field on each scene.

| Content type | Visual style | Generator | Audio style |
|---|---|---|---|
| **Math** (e.g., fractions) | Bold, clean diagrams; labels readable; minimal background | Imagen, illustration style | Slow, encouraging, child-appropriate |
| **Place** (e.g., Hampi) | Documentary photography; warm color grade; wide shots | Imagen, photorealistic | Measured, informative, slightly cinematic |
| **Story** (e.g., Panchatantra) | Storybook illustration; consistent character design across scenes | Veo for action scenes, Imagen for title/outro | Engaging, expressive, kid-friendly |

---

## 6. Error handling

- **Config errors** (missing env vars): fail fast at startup with clear message in UI.
- **Gemini errors** (safety blocks, schema validation fails): retry once with a relaxed prompt; if still failing, show user-friendly error.
- **Imagen errors** (safety filter triggered, quota): retry once with a sanitized prompt; if still failing, show a placeholder image and continue (don't abort the whole video).
- **Veo errors** (quota denied, timeout, access not granted): silently fall back to Imagen + Ken Burns motion. Surface `veo_fallback_used=True` in `RunResult` so UI can mention it.
- **Cloud TTS errors** (rare; usually billing/quota): fail loudly — there is no fallback for audio.
- **MoviePy errors** (codec, missing ffmpeg): log full traceback; show user-friendly error.

---

## 7. Cost model

Rough per-video estimate (3–5 scenes):

| Item | Quantity | Unit cost | Total |
|---|---|---|---|
| Gemini 2.5 Flash (Director) | ~2K tokens | ~$0.0001 / 1K tok | ~$0.0002 |
| Imagen (1024×1024 → upscaled) | 3–5 frames | ~$0.04 each | $0.12 – $0.20 |
| Cloud TTS (Neural2) | ~600 chars | $16 / 1M chars | ~$0.01 |
| **Total (no Veo)** | | | **~$0.13 – $0.21** |
| Veo 3 (if used) | ~10 sec | ~$0.50 / sec | +$5.00 |

UI displays this rolling total per run.

---

## 8. Caching strategy (V1 lightweight)

Cache key: SHA256 of `(topic, audience, content_type, model_versions)`.

- **Director cache:** `outputs/_cache/plans/<hash>.json` — skip Gemini call on hit.
- **Imagen cache:** `outputs/_cache/images/<hash_per_prompt>.png` — skip image gen on hit.
- **TTS cache:** `outputs/_cache/audio/<hash_per_text>.mp3` — skip TTS on hit.

Reruns of the same topic become essentially free and near-instant — useful when iterating on prompts or showing the app live without burning through API calls every time.

---

## 9. Project layout

```
ai-video-app/
├── .env                              # secrets (gitignored)
├── .env.example                      # template
├── .gitignore
├── requirements.txt
├── README.md                         # quick-start + stack overview
├── NOTES.md                          # research log w/ Google doc links
├── docs/superpowers/specs/           # this file lives here
├── app.py                            # Streamlit entrypoint
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── director.py
│   ├── visuals.py
│   ├── voice.py
│   ├── assemble.py
│   └── pipeline.py
└── outputs/                          # gitignored
    ├── _cache/
    └── <run_id>/
        ├── plan.json
        ├── scene_*.png|mp4
        ├── scene_*.mp3
        └── final.mp4
```

---

## 10. Out of scope / V2 backlog

- Lyria background music
- Tests (V1 verifies via manual end-to-end runs on 3 demo topics)
- CI / linting / type checking
- Vertex AI migration of the Gemini call (the rest of the pipeline already runs on Vertex)
- Streaming UI updates during generation
- Multi-language UI

---

## 11. Acceptance criteria

V1 ships when **all** of these hold:

1. `streamlit run app.py` opens a working UI in browser.
2. Entering `"fractions"` + `"6-year-old"` produces a `final.mp4` ≥ 20 seconds with multi-scene visuals, narration, and burned-in captions.
3. Entering `"Hampi"` + `"adult"` produces a documentary-style video.
4. Entering `"The Lion and the Mouse"` + `"10-year-old"` produces a story-style video.
5. UI displays cost estimate per run (< $0.50 without Veo).
6. `NOTES.md` exists with ≥ 5 links to Google docs (Gemini, Imagen, Veo, Cloud TTS, Vertex AI overview).
7. No third-party APIs anywhere in the code (only `google-genai`, `google-cloud-texttospeech`, `moviepy`, `streamlit`, `pydantic`, `python-dotenv`).

---

## 12. v1.1 — Interactive Video Platform (2026-05-14)

After shipping V1 I went back and asked the obvious follow-up: a linear video is fine for a one-time watch, but what if the architecture could carry interaction state — a comprehension check mid-video, a quiz at the end, a slot for a pre-roll ad? Those are real product asks for anyone trying to use this for actual teaching. The point of v1.1 was less about "build a quiz UI" and more about proving the architecture extends to interactive video without rewriting the pipeline.

### 12.1 Scope of v1.1

The capabilities baked in:

| Feature | Mechanism |
|---|---|
| Mid-video comprehension checkpoints | Director places `Scene.checkpoint: Question \| None` at the conceptually meaningful scene. The UI pauses between scenes and blocks progress until answered. |
| End-of-video summative quiz | `VideoPlan.end_quiz: Quiz \| None` with 3-5 MCQ questions. UI renders scoring + reveal-explanation. |
| Pre-roll ad placeholder | 3-second branded card in Streamlit before the video plays. Insertion point for full Google IMA SDK integration in v2. |
| Subtitle burn-in | Spoken dialogue burned at bottom third; on-screen headline stays at upper third. Controlled by `Scene.enable_subtitles`. |
| Title + outro cards | 2-second branded cards bracketing the scene sequence in `final.mp4`. |
| Per-scene MP4 export | `assemble.py` writes `scene_<i>_clip.mp4` per scene to support the Interactive player. |
| Chirp 3 HD voice primary | Per-content-type mapping (`math -> Aoede`, `place -> Kore`, `story -> Leda`). Silent fallback to Neural2-C on failure. |

### 12.2 Director — Senior Director persona

`SYSTEM_INSTRUCTIONS` rewritten as a ~150-line senior video director persona covering:

- The 7 Rules of Enhanced Output (Specific > Generic, consistent recurring character, 6-element visual language template, show-don't-tell, hook→build→reveal→recap pacing, spoken English only, captions are headlines)
- Per-content-type style guides (math / place / story / general)
- Dialogue quality calibration (bad vs good examples baked into the prompt)
- Visual prompt quality calibration (bad vs good examples baked into the prompt)
- Rules for placing one mid-video checkpoint and a 3-5 question end quiz (math / place / general only — stories skip quizzes)

### 12.3 Schema evolution (backward compatible)

```python
class VideoPlan(BaseModel):
    title: str
    tagline: str                                       # NEW
    content_type: Literal["math", "place", "story", "general"]
    style_guide: str                                   # NEW
    recurring_character_description: str | None        # NEW
    voice_recommendation: str                          # NEW
    scenes: list[Scene]
    end_quiz: Quiz | None                              # NEW

class Scene(BaseModel):
    # ... existing fields preserved ...
    checkpoint: Question | None = None                 # NEW
    enable_subtitles: bool = True                      # NEW
```

### 12.4 Streamlit UI — Linear vs Interactive modes

- Sidebar toggle picks the playback mode.
- **Linear:** plays the full concatenated `final.mp4` (title card → scenes → outro card), then renders the end quiz if present.
- **Interactive:** state-machine over `st.session_state` plays `scene_<i>_clip.mp4` files sequentially. Between scenes, if a checkpoint is set, renders the MCQ and blocks progress with a "Continue" button gated on submission. After the final scene, renders the end quiz with scoring + reveal explanations.

### 12.5 Acceptance criteria for v1.1 (all met 2026-05-14)

1. Director schema and SYSTEM_INSTRUCTIONS support quiz + checkpoint generation.
2. Cloud TTS Chirp 3 HD voices auto-selected by content type; Neural2 silent fallback works.
3. Pipeline writes `quiz.json`, `checkpoints.json`, and per-scene `scene_<i>_clip.mp4` files alongside `final.mp4`.
4. Streamlit UI exposes Linear and Interactive modes with a pre-roll ad placeholder.
5. End-to-end test on fractions produces a valid run (~$0.17, 4 scenes, 1 checkpoint, 3-question end quiz, Chirp 3 HD voice used without fallback).
6. README, NOTES.md, and this design spec updated to reflect v1.1 architecture.
