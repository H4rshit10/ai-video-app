# Research Log

Google AI documentation consulted while building this app.

## Gemini API (Director)
- Models overview — https://ai.google.dev/gemini-api/docs/models
- Structured JSON output — https://ai.google.dev/gemini-api/docs/structured-output
- google-genai Python SDK — https://github.com/googleapis/python-genai
- System instructions — https://ai.google.dev/gemini-api/docs/text-generation#system-instructions

## Imagen (Frames) — `imagen-4.0-generate-001` via Vertex AI
- Imagen 4 model card — https://cloud.google.com/vertex-ai/generative-ai/docs/image/imagen-models
- Image generation guide (Vertex AI) — https://cloud.google.com/vertex-ai/generative-ai/docs/image/generate-images
- Aspect-ratio + person-generation flags — https://ai.google.dev/gemini-api/docs/imagen#parameters

> **Build finding (2026-05-13):** AI Studio's `imagen-3.0-generate-002` returned `404 Not Found` on the v1beta endpoint; only Imagen 4 IDs are served. AI Studio's Imagen 4 then required a separate paid-plan upgrade (`Imagen 3 is only available on paid plans`). Pivoted to Vertex AI mode (`Client(vertexai=True, ...)`), which uses our existing GCP billing and works out of the box.

## Veo (Narrative video)
- Veo 3 on Vertex AI — https://cloud.google.com/vertex-ai/generative-ai/docs/video/generate-videos
- Veo prompting guide — https://ai.google.dev/gemini-api/docs/video

## Google Cloud Text-to-Speech — Chirp 3 HD primary, Neural2 fallback
- Voices catalog (Neural2 / Chirp 3 HD) — https://cloud.google.com/text-to-speech/docs/voices
- Chirp 3 HD overview — https://cloud.google.com/text-to-speech/docs/chirp3-hd
- Quickstart with ADC — https://cloud.google.com/text-to-speech/docs/quickstarts
- SSML reference (Neural2 only — Chirp 3 HD has limited SSML support, so plain text is used) — https://cloud.google.com/text-to-speech/docs/ssml

> **Build finding (2026-05-14):** Chirp 3 HD voices work natively on the project with ADC — no extra enablement needed. Per-content-type mapping baked into `src/voice.py`: `math -> Aoede` (warm), `place -> Kore` (narrator), `story -> Leda` (expressive). Silent fallback to `Neural2-C` if a Chirp voice fails.

## Interactive video — quiz + ad insertion
- IAB VAST (Video Ad Serving Template) — https://iabtechlab.com/standards/vast/
- Google IMA SDK (HTML5) — https://developers.google.com/interactive-media-ads/docs/sdks/html5
- Google Ad Manager (server-side ad decisioning) — https://admanager.google.com/

> **v1.1 decision:** Interactive quiz + mid-video checkpoints are baked into the Director's structured output (`Scene.checkpoint` and `VideoPlan.end_quiz`). The Streamlit UI implements a state-machine player that pauses between scenes for checkpoints and renders the summative quiz after the final scene. Pre-roll ad is a placeholder card; full IMA SDK integration is on the v2 roadmap (would require a custom HTML5 player to replace `st.video`).

## Vertex AI vs AI Studio
- Differences + when to use which — https://cloud.google.com/vertex-ai/generative-ai/docs/migrate-from-gemini
- Application Default Credentials (ADC) — https://cloud.google.com/docs/authentication/application-default-credentials

## MoviePy 2.x
- Migration from v1 — https://zulko.github.io/moviepy/getting_started/updating_to_v2.html

## Architecture decisions made during build

- **Auth split (AI Studio for Gemini, ADC + Vertex AI for everything else):** Gemini's free tier covers Director usage; Imagen/Veo/TTS all share one ADC credential against our GCP project. Production migration would move Gemini to Vertex AI too for unified IAM and audit logging.
- **Per-content-type generator selection:** Veo morphs text labels and counts, which breaks math/diagram precision. Director picks Imagen for math/place, Veo for story.
- **Ken Burns over true generative motion for math:** deterministic, label-stable, no model artifacts.
- **SSML prosody wrap:** lets us slow down narration for first-time learners without re-recording.
- **Burned-in captions:** accessibility + retention on muted playback.
- **Veo silent fallback to Imagen:** keeps demo robust when Veo quota is denied.

## v1.1 — Interactive video architecture (2026-05-14)

- **Director schema evolution:** added `tagline`, `style_guide`, `recurring_character_description`, `voice_recommendation`, `end_quiz` to `VideoPlan`; added `checkpoint` and `enable_subtitles` to `Scene`. Backward-compatible — all new fields optional with sensible defaults.
- **Senior-director system prompt:** ~150-line `SYSTEM_INSTRUCTIONS` covering 7 quality rules (Specific > Generic, consistent recurring character, 6-element visual language template, show-don't-tell, hook→build→reveal→recap pacing, spoken English only, captions are headlines), per-content-type style guides, dialogue/visual calibration examples, and rules for placing checkpoints and end-quiz questions.
- **Per-scene MP4 export:** `assemble.py` writes each `scene_<i>_clip.mp4` alongside the concatenated `final.mp4`. The Interactive mode in the UI plays these clips sequentially with checkpoint pauses.
- **Title + outro cards:** 2-second branded cards bracketing the scene sequence in `final.mp4`. Title uses `plan.title + plan.tagline`; outro uses a project-wide line.
- **Subtitle burn-in:** when `Scene.enable_subtitles` is True (default), MoviePy renders the spoken dialogue at the bottom third, with the existing on-screen headline staying at the upper third.
- **Chirp 3 HD with silent Neural2 fallback:** voice resolution order is UI override > Director `voice_recommendation` > content-type default. Chirp 3 HD uses plain-text input (limited SSML support); Neural2 fallback uses full SSML for prosody control.
