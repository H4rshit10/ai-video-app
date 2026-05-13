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

## Google Cloud Text-to-Speech
- Voices catalog (Neural2 / Chirp 3 HD) — https://cloud.google.com/text-to-speech/docs/voices
- Quickstart with ADC — https://cloud.google.com/text-to-speech/docs/quickstarts
- SSML reference — https://cloud.google.com/text-to-speech/docs/ssml

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
