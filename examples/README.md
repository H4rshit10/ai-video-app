# Example outputs

Real videos generated end-to-end by the pipeline, committed here so you can preview the app's output without running it.

Each folder contains:
- `final.mp4` — the rendered 1080p video with title card, motion, subtitle burn-in, narration, and outro card
- `plan.json` — the Gemini-produced VideoPlan that drove generation
- `quiz.json` — the end-of-video summative quiz (math / place / general videos)
- `checkpoints.json` — mid-video comprehension checkpoint questions (when set by the Director)

| Example | Content type | Generator strategy | Voice (Chirp 3 HD) | Cost |
|---|---|---|---|---|
| [`fractions/`](fractions/) | math (a 6-year-old learner) | Imagen 4 + Ken Burns motion + subtitles | `Aoede` (warm) | ~$0.17 |
| [`hampi/`](hampi/) | place (adult audience) | Imagen 4 documentary style + Ken Burns motion | `Kore` (narrator) | ~$0.17 |
| [`the-lion-and-the-mouse/`](the-lion-and-the-mouse/) | story (a 10-year-old learner) | Imagen 4 storybook style + Ken Burns motion | `Leda` (expressive) | ~$0.17 |

## What's new in v1.1

These examples reflect the v1.1 Interactive Video Platform features:

- **Title and outro cards** bracketing every video
- **Subtitle burn-in** — the spoken dialogue is rendered at the bottom third of every scene for accessibility
- **Chirp 3 HD voices** auto-selected per content type
- **End-of-video quiz** baked into the Director's structured output (see each `quiz.json`)
- **Mid-video checkpoints** at conceptually meaningful scenes (see each `checkpoints.json`)
- **Recurring character description** in the plan enforces visual consistency across scenes

Open each `plan.json` to see how the Director (Gemini 2.5 Flash) decomposes a topic into a paced multi-scene script with embedded interaction points. Open `final.mp4` to see the result. Open `quiz.json` and `checkpoints.json` to see the interactive layer that drives the Streamlit Interactive playback mode.
