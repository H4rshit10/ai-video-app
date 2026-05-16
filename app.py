"""Streamlit UI for the Interactive Video Platform.

v1.1 features:
  - Linear mode (single final.mp4) and Interactive mode (scene-by-scene with checkpoints)
  - Pre-roll ad placeholder (3s branded card)
  - End-of-video quiz with scoring and reveal explanations
  - Mid-video checkpoints that pause playback and block progress until answered
"""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.pipeline import RunResult, generate_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(page_title="AI Video Generator", page_icon="🎬", layout="wide")


# ====================================================================
# Session state initialisation
# ====================================================================

def _init_state() -> None:
    defaults = {
        "result": None,
        "interactive_index": 0,              # current scene cursor (state machine)
        "checkpoint_answers": {},            # scene_index -> selected choice index
        "checkpoint_submitted": {},          # scene_index -> bool
        "quiz_answers": {},                  # question_index -> selected choice index
        "quiz_submitted": False,
        "preroll_done": False,
        "midroll_done": False,               # mid-roll ad dismissed for this run
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_playback_state() -> None:
    st.session_state.interactive_index = 0
    st.session_state.checkpoint_answers = {}
    st.session_state.checkpoint_submitted = {}
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.preroll_done = False
    st.session_state.midroll_done = False


# ====================================================================
# Playback render functions
# ====================================================================

def _render_linear(result: RunResult) -> None:
    """Linear mode: play full final.mp4, then show the end quiz if any."""
    st.video(str(result.final_video))

    with open(result.final_video, "rb") as f:
        st.download_button(
            "⬇️ Download MP4",
            data=f.read(),
            file_name=f"{result.run_id}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    if result.plan.end_quiz is not None:
        st.divider()
        _render_end_quiz(result)


def _render_interactive(result: RunResult) -> None:
    """Interactive mode: per-scene playback with mid-roll ad + checkpoints between scenes."""
    scenes = result.plan.scenes
    clips = result.per_scene_clips
    idx = st.session_state.interactive_index

    if idx >= len(scenes):
        st.success(f"✅ All {len(scenes)} scenes complete.")
        if result.plan.end_quiz is not None:
            st.divider()
            _render_end_quiz(result)
        else:
            st.info("This content type does not include an end quiz.")
        if st.button("🔁 Replay from start", use_container_width=True):
            _reset_playback_state()
            st.rerun()
        return

    # ---- Mid-roll ad gate (VAST/IMA Compliance Anchor Point) ---------------
    # The ad slot fires once per run, right before the midpoint scene.
    # Production wiring would replace this placeholder with a real Google IMA
    # SDK call serving a VAST-tagged creative. Skip-after-N is the standard
    # IAB linear-ad UX; we expose a manual skip here as the demo equivalent.
    midpoint = len(scenes) // 2
    if idx == midpoint and not st.session_state.midroll_done and idx > 0:
        _render_midroll_ad(scene_index=idx, total_scenes=len(scenes))
        return
    # -----------------------------------------------------------------------

    scene = scenes[idx]
    clip_path = clips[idx] if idx < len(clips) else None

    st.caption(f"Scene {idx + 1} of {len(scenes)} · {scene.on_screen_text or scene.audio_dialogue[:50]}")
    if clip_path and Path(clip_path).exists():
        st.video(str(clip_path))
    else:
        st.warning(f"Per-scene clip not found at {clip_path}.")

    if scene.checkpoint is not None:
        _render_checkpoint(scene_index=idx, scene=scene)
    else:
        if st.button("Continue to next scene ▶", type="primary", use_container_width=True, key=f"continue_{idx}"):
            st.session_state.interactive_index += 1
            st.rerun()


def _render_midroll_ad(scene_index: int, total_scenes: int) -> None:
    """Mid-roll ad slot — fires once per run at the midpoint scene.

    VAST/IMA Compliance Anchor Point: this is the structural placeholder where
    a production deployment plugs in Google IMA SDK's mid-roll cue point. The
    SDK would (a) request a VAST-tagged creative from the ad server, (b) gate
    the skip button on the ad's skip_offset attribute, and (c) emit IAB-defined
    tracking pixels for start / first-quartile / midpoint / third-quartile /
    complete. For this demo, the visual treatment + skip control match the
    standard IAB linear-ad pattern; the network call is mocked out.
    """
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #f6a93b 0%, #ff5e62 100%);
            padding: 50px 30px;
            border-radius: 14px;
            text-align: center;
            color: white;
            margin: 10px 0;
        ">
            <div style="font-size: 14px; opacity: 0.75; letter-spacing: 2px;">MID-ROLL AD</div>
            <h2 style="margin: 14px 0 8px;">Your brand here</h2>
            <p style="opacity: 0.9; margin: 0;">5-second placeholder · VAST / IMA SDK cue point in v2</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Inserted between scene {scene_index} and scene {scene_index + 1} of {total_scenes}.")
    if st.button("Skip mid-roll ▶", type="primary", use_container_width=True, key="midroll_skip"):
        st.session_state.midroll_done = True
        st.rerun()


def _render_checkpoint(scene_index: int, scene) -> None:
    q = scene.checkpoint
    submitted = st.session_state.checkpoint_submitted.get(scene_index, False)

    st.markdown("### 🛑 Quick check before continuing")
    st.markdown(f"**{q.question}**")

    selected = st.radio(
        "Pick one:",
        options=list(range(len(q.choices))),
        format_func=lambda i: q.choices[i],
        key=f"checkpoint_{scene_index}_choice",
        index=None,
    )

    if not submitted:
        if st.button("Submit answer", key=f"checkpoint_{scene_index}_submit", type="primary"):
            if selected is None:
                st.warning("Pick an option first.")
            else:
                st.session_state.checkpoint_answers[scene_index] = selected
                st.session_state.checkpoint_submitted[scene_index] = True
                st.rerun()
    else:
        chosen = st.session_state.checkpoint_answers[scene_index]
        is_correct = chosen == q.correct_index
        if is_correct:
            st.success(f"✅ Correct! {q.explanation}")
        else:
            st.error(
                f"❌ Not quite. The right answer was **{q.choices[q.correct_index]}**. {q.explanation}"
            )
        if st.button(
            "Continue to next scene ▶",
            key=f"checkpoint_{scene_index}_next",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.interactive_index += 1
            st.rerun()


def _render_end_quiz(result: RunResult) -> None:
    quiz = result.plan.end_quiz
    total = len(quiz.questions)
    answered = len(st.session_state.quiz_answers)

    st.markdown("## 📝 End-of-video quiz")
    st.caption(f"{total} questions on what you just watched.")

    if not st.session_state.quiz_submitted:
        # Progress indicator — dots + count.
        dots = "● " * answered + "○ " * (total - answered)
        st.markdown(f"**Progress:** {dots}  ({answered} of {total} answered)")
        st.write("")

        for i, q in enumerate(quiz.questions):
            st.markdown(f"**Question {i + 1} of {total}**")
            st.markdown(q.question)
            choice = st.radio(
                f"Q{i + 1}",
                options=list(range(len(q.choices))),
                format_func=lambda idx, q=q: q.choices[idx],
                key=f"quiz_q_{i}",
                label_visibility="collapsed",
                index=None,
            )
            if choice is not None:
                st.session_state.quiz_answers[i] = choice
            st.write("")

        if st.button("Submit quiz", type="primary", use_container_width=True):
            if len(st.session_state.quiz_answers) < total:
                st.warning(f"Please answer every question before submitting ({answered} of {total} done).")
            else:
                st.session_state.quiz_submitted = True
                st.rerun()
    else:
        score = sum(
            1 for i, q in enumerate(quiz.questions)
            if st.session_state.quiz_answers.get(i) == q.correct_index
        )
        st.markdown(f"### Score: **{score} / {len(quiz.questions)}**")

        for i, q in enumerate(quiz.questions):
            chosen = st.session_state.quiz_answers.get(i)
            ok = chosen == q.correct_index
            icon = "✅" if ok else "❌"
            with st.expander(f"{icon} Q{i + 1}: {q.question}", expanded=not ok):
                for j, choice_text in enumerate(q.choices):
                    if j == q.correct_index:
                        st.markdown(f"- **{choice_text}** ← correct")
                    elif j == chosen:
                        st.markdown(f"- ~~{choice_text}~~ ← your answer")
                    else:
                        st.markdown(f"- {choice_text}")
                st.info(q.explanation)

        if st.button("Try again", use_container_width=True):
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()


def _render_run_details(result: RunResult, mode: str) -> None:
    st.subheader("Run details")
    st.metric("Mode", mode)
    st.metric("Content type", result.plan.content_type)
    st.metric("Scenes", len(result.plan.scenes))
    st.metric("Estimated cost", f"${result.cost.total_usd:.4f}")
    st.metric("Voice", result.voice_used)

    if result.voice_fallback_used:
        st.info("Chirp 3 HD unavailable — fell back to Neural2-C silently.")
    if result.veo_attempted and result.veo_fallback_used:
        st.info(
            "Veo succeeded for some scenes and failed for others — failed scenes fell back to Imagen + motion."
        )
    elif not result.veo_attempted:
        st.caption("Veo was disabled — all scenes used Imagen + motion.")

    with st.expander("Cost breakdown"):
        st.json({
            "director (Gemini)": f"${result.cost.director_usd:.4f}",
            "imagen": f"${result.cost.imagen_usd:.4f}",
            "veo": f"${result.cost.veo_usd:.4f}",
            "tts": f"${result.cost.tts_usd:.4f}",
            "total": f"${result.cost.total_usd:.4f}",
        })

    with st.expander("Scene plan"):
        st.json(result.plan.model_dump())


# ====================================================================
# Main UI
# ====================================================================

st.title("🎬 AI Educational Video Generator")
st.caption("Google AI stack · Gemini 2.5 · Imagen 4 · Veo 3 · Cloud TTS Chirp 3 HD")

_init_state()


with st.sidebar:
    st.header("Generate a video")
    topic = st.text_input("Topic", value="Fractions", help="What concept should we explain?")
    audience = st.selectbox(
        "Audience",
        options=["6-year-old", "10-year-old", "12-year-old", "adult"],
        index=0,
    )
    content_type = st.selectbox(
        "Content type",
        options=["auto", "math", "place", "story", "general"],
        index=0,
    )
    voice_override = st.selectbox(
        "Voice (optional override)",
        options=[
            "auto (Director picks)",
            "en-US-Chirp3-HD-Aoede",
            "en-US-Chirp3-HD-Kore",
            "en-US-Chirp3-HD-Leda",
            "en-US-Chirp3-HD-Charon",
            "en-US-Neural2-C (legacy fallback)",
        ],
        index=0,
    )

    st.divider()
    playback_mode = st.radio(
        "Playback mode",
        options=["Linear", "Interactive"],
        index=1,
        help="Linear: full assembled .mp4. Interactive: per-scene with mid-video checkpoints.",
    )

    st.divider()
    allow_veo = st.checkbox(
        "🎥 Allow Veo (expensive)",
        value=False,
        help="Off (default): Imagen for every scene, ~$0.20/video. On: Veo for narrative scenes.",
    )
    if allow_veo:
        st.warning("Veo enabled. A story-type video may cost $5–15.")
    else:
        st.caption("Imagen + motion only. Predictable ~$0.20/video.")

    generate_btn = st.button("Generate video", type="primary", use_container_width=True)


if generate_btn:
    progress_box = st.empty()
    log_expander = st.expander("Generation log", expanded=True)

    def _progress(msg: str) -> None:
        progress_box.markdown(f"**{msg}**")
        with log_expander:
            st.write(msg)

    voice_name: str | None
    if voice_override.startswith("auto"):
        voice_name = None
    elif voice_override.startswith("en-US-Neural2-C"):
        voice_name = "en-US-Neural2-C"
    else:
        voice_name = voice_override

    try:
        with st.spinner("Generating..."):
            result = generate_video(
                topic=topic,
                audience=audience,
                content_type=content_type if content_type != "auto" else None,
                voice_name=voice_name,
                allow_veo=allow_veo,
                progress=_progress,
            )
        st.session_state.result = result
        _reset_playback_state()
        progress_box.markdown(
            f"**✅ Done in {result.elapsed_seconds:.1f}s · run `{result.run_id}`**"
        )
    except Exception as e:
        st.error(f"Generation failed: {type(e).__name__}: {e}")
        st.exception(e)


result: RunResult | None = st.session_state.result

if result is None:
    st.info("Enter a topic in the sidebar and click **Generate video** to start.")
    st.markdown(
        """
        **Example topics:**

        | Topic | Audience | Content type |
        |---|---|---|
        | `Fractions` | `6-year-old` | math |
        | `Hampi` | `adult` | place |
        | `The Lion and the Mouse` | `10-year-old` | story |
        """
    )
else:
    col_player, col_meta = st.columns([2, 1])

    with col_player:
        st.subheader(result.plan.title)
        st.caption(result.plan.tagline)

        if not st.session_state.preroll_done:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #6e8efb 0%, #a777e3 100%);
                    padding: 50px 30px;
                    border-radius: 14px;
                    text-align: center;
                    color: white;
                    margin: 10px 0;
                ">
                    <div style="font-size: 14px; opacity: 0.7; letter-spacing: 2px;">PRE-ROLL AD</div>
                    <h2 style="margin: 14px 0 8px;">Your brand here</h2>
                    <p style="opacity: 0.85; margin: 0;">3-second placeholder · Full Google IMA SDK integration in v2</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Skip ad ▶", type="primary", use_container_width=True):
                st.session_state.preroll_done = True
                st.rerun()
        else:
            if playback_mode == "Linear":
                _render_linear(result)
            else:
                _render_interactive(result)

    with col_meta:
        _render_run_details(result, playback_mode)
