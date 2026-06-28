"""Streamlit UI — Domain-Specific Creative Workspace.

The vertical pivot: the app is no longer "a video generator with a dropdown."
It's a domain-aware creative workspace where video rendering is one of several
deliverable formats. Picking a domain in the sidebar swaps the main panel into
a layout tuned to that profession's actual deliverable:

  - General           — original interactive/linear video flow (default)
  - Interior Design   — 3-column layout variant grid + material palette board
  - Logo / Branding   — Brand Asset Kit (mark, palette, typography, archetype)
  - Marketing / PPT   — Sequential Slide Deck Blueprint (cards)
  - Teaching          — original interactive state machine with checkpoints + quiz

Video rendering (TTS + MoviePy assembly) is OPTIONAL. The 'Render video clip
sequence' toggle decouples rapid asset iteration (instant grid output) from
finalised video production. Asset-only mode skips TTS + assembly entirely.

System telemetry — cost breakdown, fallback flags, run JSON — is collapsed at
the bottom of the page so the creative workspace stays uncluttered.
"""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src import dispatcher as dispatcher_mod
from src.pipeline import (
    DispatchedAssetResult,
    RunResult,
    generate_dispatched_asset,
    generate_video,
)

CAMPAIGN_ROOT = Path("config/marketing")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(page_title="Multi-Domain Visual Factory", page_icon="🎨", layout="wide")


# ====================================================================
# Session state
# ====================================================================

def _init_state() -> None:
    defaults = {
        "result": None,
        "interactive_index": 0,
        "checkpoint_answers": {},
        "checkpoint_submitted": {},
        "quiz_answers": {},
        "quiz_submitted": False,
        "preroll_done": False,
        "midroll_done": False,
        # Campaign-mode state — populated when the user runs a profile dispatch.
        "campaign_result": None,
        "dispatcher_preview": None,
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
# Domain configuration (drives sidebar labels + workspace routing)
# ====================================================================

DOMAIN_CHOICES = [
    "General (horizontal)",
    "Interior Design",
    "Logo / Branding",
    "Marketing / Presentation",
    "Teaching / Educational",
]
DOMAIN_KEYS = {
    "General (horizontal)": None,
    "Interior Design": "interior_design",
    "Logo / Branding": "branding",
    "Marketing / Presentation": "marketing",
    "Teaching / Educational": "teaching",
}

# Per-domain prompt config for the sidebar — labels, defaults, and helper text.
DOMAIN_PROMPT_CONFIG: dict[str | None, dict] = {
    None: {
        "topic_label": "Topic",
        "topic_help": "What concept should we explain?",
        "topic_default": "Fractions",
        "secondary": None,
    },
    "interior_design": {
        "topic_label": "Describe the space layout or style",
        "topic_help": "e.g., 'Mid-century Scandinavian living room with reading nook by the window'",
        "topic_default": "Mid-century Scandinavian living room",
        "secondary": None,
    },
    "branding": {
        "topic_label": "Company name",
        "topic_help": "The name we're building identity around",
        "topic_default": "Nexora",
        "secondary": {
            "label": "Brand core vibe",
            "help": "The strategic positioning in one phrase — e.g., 'precise, calm, slightly defiant'",
            "default": "Confident and minimal — a heritage tech brand",
        },
    },
    "marketing": {
        "topic_label": "Presentation topic / pitch vibe",
        "topic_help": "The deck's single thesis — what you actually want to land",
        "topic_default": "Launching our new AI feature — Code Mode",
        "secondary": None,
    },
    "teaching": {
        "topic_label": "Lesson topic",
        "topic_help": "The concept the student should leave understanding",
        "topic_default": "Fractions",
        "secondary": None,
    },
}


# ====================================================================
# Workspace renderers (one per domain — main panel content)
# ====================================================================

def _hex_swatch_html(palette: list[str]) -> str:
    """Render a horizontal hex-color palette as styled HTML chips."""
    chips = []
    for hex_code in palette:
        hex_code = (hex_code or "").strip()
        if not hex_code.startswith("#"):
            continue
        chips.append(
            f'<div style="display:inline-flex;flex-direction:column;align-items:center;margin:6px 12px 6px 0;">'
            f'<div style="width:64px;height:64px;border-radius:10px;background:{hex_code};'
            f'border:1px solid rgba(0,0,0,0.08);box-shadow:0 1px 4px rgba(0,0,0,0.06);"></div>'
            f'<div style="font-family:monospace;font-size:12px;margin-top:6px;color:#444;">{hex_code}</div>'
            f'</div>'
        )
    return f'<div style="display:flex;flex-wrap:wrap;margin:8px 0;">{"".join(chips)}</div>'


def _render_workspace_interior(result: RunResult) -> None:
    """Interior Design workspace: 3-column variant grid + material board + (optional) cinematic pan."""
    plan = result.plan
    st.subheader(plan.title)
    st.caption(plan.tagline)

    # --- Material palette board ---
    if plan.material_palette:
        st.markdown("### Material Specification")
        cols = st.columns(min(len(plan.material_palette), 4))
        for i, mat in enumerate(plan.material_palette):
            with cols[i % len(cols)]:
                st.markdown(
                    f'<div style="padding:14px 18px;border:1px solid #eee;border-radius:10px;'
                    f'background:#fafafa;margin-bottom:10px;font-family:Georgia,serif;font-size:14px;">'
                    f'<div style="font-size:11px;letter-spacing:1.5px;color:#888;margin-bottom:4px;">'
                    f'MATERIAL {i+1:02d}</div>'
                    f'<div style="color:#222;">{mat}</div></div>',
                    unsafe_allow_html=True,
                )

    if plan.lighting_specification:
        st.markdown("### Lighting Specification")
        st.markdown(
            f'<div style="padding:14px 18px;border-left:3px solid #c9a063;background:#fdfaf3;'
            f'border-radius:0 8px 8px 0;font-family:Georgia,serif;font-size:14px;color:#3a2e1a;">'
            f'{plan.lighting_specification}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Style Direction")
    st.caption(plan.style_guide)

    # --- Layout variant grid (3-column) ---
    st.markdown("### Layout Variants")
    variants = list(zip(plan.scenes, result.scene_images))[:3]
    if not variants:
        st.warning("No scene images available — re-run the workspace to populate the grid.")
        return

    cols = st.columns(len(variants))
    for i, (scene, img_path) in enumerate(variants):
        with cols[i]:
            st.markdown(
                f'<div style="font-size:12px;letter-spacing:2px;color:#888;margin-bottom:6px;">'
                f'OPTION {i+1:02d}</div>',
                unsafe_allow_html=True,
            )
            if img_path and Path(img_path).exists():
                st.image(str(img_path), use_container_width=True)
            label = scene.on_screen_text or "Untitled view"
            st.markdown(f"**{label}**")
            st.caption(scene.audio_dialogue)

    # --- Layer Modifications stub (the future inpainting workspace) ---
    st.markdown("### Layer Modifications")
    st.markdown(
        '<div style="padding:18px 22px;border:1.5px dashed #c9a063;border-radius:10px;'
        'background:#fdfaf3;color:#5a4a2a;">'
        '<div style="font-weight:600;margin-bottom:6px;">Region-aware editing — coming in v2</div>'
        'Click any wall, floor, or object above to define a region. Type the modification '
        '(e.g., <em>"replace this wall with limewash plaster"</em>) and the inpainter will '
        're-render only that area — preserving the rest of the composition. '
        'Backed by SAM 2 segmentation + Imagen 3 inpainting via Vertex AI.'
        '</div>',
        unsafe_allow_html=True,
    )

    if result.rendered and result.final_video:
        st.divider()
        st.markdown("### Cinematic Pan")
        st.video(str(result.final_video))


def _render_workspace_branding(result: RunResult) -> None:
    """Branding workspace: brand mark, palette, typography, archetype."""
    plan = result.plan
    st.subheader(plan.title)
    st.caption(plan.tagline)

    col_mark, col_meta = st.columns([1.4, 1])

    with col_mark:
        st.markdown("### Brand Mark")
        if result.scene_images:
            st.image(str(result.scene_images[0]), use_container_width=True)
        else:
            st.warning("No brand mark generated — re-run the workspace.")

        if len(result.scene_images) > 1:
            st.markdown("**Alternate marks**")
            alt_cols = st.columns(min(len(result.scene_images) - 1, 3))
            for i, img in enumerate(result.scene_images[1:4]):
                with alt_cols[i % len(alt_cols)]:
                    if Path(img).exists():
                        st.image(str(img), use_container_width=True)

    with col_meta:
        st.markdown("### Brand Asset Kit")

        if plan.brand_archetype:
            st.markdown(
                f'<div style="padding:10px 14px;background:#f4f0ff;border-radius:8px;'
                f'margin-bottom:14px;">'
                f'<div style="font-size:11px;letter-spacing:1.5px;color:#7c5ed6;'
                f'margin-bottom:4px;">ARCHETYPE</div>'
                f'<div style="font-size:18px;font-weight:600;color:#3a2670;">'
                f'{plan.brand_archetype}</div></div>',
                unsafe_allow_html=True,
            )

        if plan.brand_palette:
            st.markdown("**Colour Palette**")
            st.markdown(_hex_swatch_html(plan.brand_palette), unsafe_allow_html=True)

        if plan.typography_pairing:
            st.markdown("**Typography Pairing**")
            st.markdown(
                f'<div style="padding:12px 16px;background:#fafafa;border-radius:8px;'
                f'border-left:3px solid #222;font-size:15px;color:#222;">'
                f'{plan.typography_pairing}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("**Style Direction**")
        st.caption(plan.style_guide)

    if result.rendered and result.final_video:
        st.divider()
        st.markdown("### Motion Reveal")
        st.video(str(result.final_video))


def _render_workspace_marketing(result: RunResult) -> None:
    """Marketing/PPT workspace: sequential slide deck blueprint with cards."""
    plan = result.plan
    st.subheader(plan.title)
    st.caption(plan.tagline)

    if plan.narrative_thesis:
        st.markdown("### Narrative Thesis")
        st.markdown(
            f'<div style="padding:18px 24px;background:linear-gradient(135deg,#f3f6ff,#fff);'
            f'border-left:4px solid #4a6cf7;border-radius:0 10px 10px 0;font-size:17px;'
            f'color:#1a2540;font-style:italic;">{plan.narrative_thesis}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Slide Deck Blueprint")
    st.caption(plan.style_guide)

    scenes_imgs = list(zip(plan.scenes, result.scene_images))
    if not scenes_imgs:
        st.warning("No slide assets available.")
        return

    for i, (scene, img_path) in enumerate(scenes_imgs):
        with st.container():
            slide_col, content_col = st.columns([1.2, 1.8])
            with slide_col:
                if img_path and Path(img_path).exists():
                    st.image(str(img_path), use_container_width=True)
            with content_col:
                st.markdown(
                    f'<div style="font-size:11px;letter-spacing:2px;color:#888;'
                    f'margin-bottom:4px;">SLIDE {i+1:02d}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"### {scene.on_screen_text or 'Untitled'}")
                st.markdown(scene.audio_dialogue)
            st.divider()

    if result.rendered and result.final_video:
        st.markdown("### Animated Sizzle")
        st.video(str(result.final_video))


def _render_workspace_teaching(result: RunResult, playback_mode: str) -> None:
    """Teaching workspace: keeps the interactive state machine + linear modes intact."""
    if not result.rendered or not result.final_video:
        # Asset-only teaching mode — show image grid + a hint to enable rendering.
        st.subheader(result.plan.title)
        st.caption(result.plan.tagline)
        st.info("Lesson assets generated. Enable **Render Video Clip Sequence** in the sidebar to produce the narrated lesson video with checkpoints + quiz.")
        if result.scene_images:
            cols = st.columns(min(len(result.scene_images), 3))
            for i, img in enumerate(result.scene_images):
                with cols[i % len(cols)]:
                    if Path(img).exists():
                        st.image(str(img), use_container_width=True)
                    if i < len(result.plan.scenes):
                        st.caption(result.plan.scenes[i].on_screen_text or "")
        return

    st.subheader(result.plan.title)
    st.caption(result.plan.tagline)

    if not st.session_state.preroll_done:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#6e8efb 0%,#a777e3 100%);
                padding:50px 30px;border-radius:14px;text-align:center;color:white;
                margin:10px 0;">
                <div style="font-size:14px;opacity:0.7;letter-spacing:2px;">PRE-ROLL AD</div>
                <h2 style="margin:14px 0 8px;">Your brand here</h2>
                <p style="opacity:0.85;margin:0;">3-second placeholder · Full Google IMA SDK integration in v2</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Skip ad ▶", type="primary", use_container_width=True):
            st.session_state.preroll_done = True
            st.rerun()
        return

    if playback_mode == "Linear":
        _render_linear(result)
    else:
        _render_interactive(result)


def _render_workspace_general(result: RunResult, playback_mode: str) -> None:
    """General workspace: original two-column layout for the horizontal mode."""
    col_player, col_meta = st.columns([2, 1])

    with col_player:
        st.subheader(result.plan.title)
        st.caption(result.plan.tagline)
        if not result.rendered or not result.final_video:
            st.info("Asset-only run. Enable **Render Video Clip Sequence** to produce a video.")
            if result.scene_images:
                for img in result.scene_images:
                    if Path(img).exists():
                        st.image(str(img), use_container_width=True)
            return

        if not st.session_state.preroll_done:
            st.markdown(
                """
                <div style="background:linear-gradient(135deg,#6e8efb 0%,#a777e3 100%);
                    padding:50px 30px;border-radius:14px;text-align:center;color:white;
                    margin:10px 0;">
                    <div style="font-size:14px;opacity:0.7;letter-spacing:2px;">PRE-ROLL AD</div>
                    <h2 style="margin:14px 0 8px;">Your brand here</h2>
                    <p style="opacity:0.85;margin:0;">3-second placeholder · v2 IMA SDK</p>
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
        st.markdown("### Quick stats")
        st.metric("Scenes", len(result.plan.scenes))
        st.metric("Cost", f"${result.cost.total_usd:.4f}")
        if not result.rendered:
            st.caption("Asset-only mode — no video rendered.")


# ====================================================================
# Linear / Interactive playback helpers (Teaching + General)
# ====================================================================

def _render_linear(result: RunResult) -> None:
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
    scenes = result.plan.scenes
    clips = result.per_scene_clips
    idx = st.session_state.interactive_index

    if idx >= len(scenes):
        st.success(f"✅ All {len(scenes)} scenes complete.")
        if result.plan.end_quiz is not None:
            st.divider()
            _render_end_quiz(result)
        else:
            st.info("No end quiz for this run.")
        if st.button("🔁 Replay from start", use_container_width=True):
            _reset_playback_state()
            st.rerun()
        return

    midpoint = len(scenes) // 2
    if idx == midpoint and not st.session_state.midroll_done and idx > 0:
        _render_midroll_ad(scene_index=idx, total_scenes=len(scenes))
        return

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
    """Mid-roll ad slot (VAST/IMA Compliance Anchor Point)."""
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#f6a93b 0%,#ff5e62 100%);
            padding:50px 30px;border-radius:14px;text-align:center;color:white;
            margin:10px 0;">
            <div style="font-size:14px;opacity:0.75;letter-spacing:2px;">MID-ROLL AD</div>
            <h2 style="margin:14px 0 8px;">Your brand here</h2>
            <p style="opacity:0.9;margin:0;">5-second placeholder · VAST / IMA SDK cue point in v2</p>
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
            st.error(f"❌ Not quite. The right answer was **{q.choices[q.correct_index]}**. {q.explanation}")
        if st.button("Continue to next scene ▶", key=f"checkpoint_{scene_index}_next", type="primary", use_container_width=True):
            st.session_state.interactive_index += 1
            st.rerun()


def _render_end_quiz(result: RunResult) -> None:
    quiz = result.plan.end_quiz
    total = len(quiz.questions)
    answered = len(st.session_state.quiz_answers)

    st.markdown("## 📝 End-of-video quiz")
    st.caption(f"{total} questions on what you just watched.")

    if not st.session_state.quiz_submitted:
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
                st.warning(f"Please answer every question first ({answered}/{total} done).")
            else:
                st.session_state.quiz_submitted = True
                st.rerun()
    else:
        score = sum(1 for i, q in enumerate(quiz.questions) if st.session_state.quiz_answers.get(i) == q.correct_index)
        st.markdown(f"### Score: **{score} / {total}**")
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


# ====================================================================
# System telemetry (collapsed at the bottom of every page)
# ====================================================================

def _render_telemetry(result: RunResult, mode: str, domain: str | None) -> None:
    """Collapsed system telemetry — cost, fallback flags, raw plan JSON."""
    with st.expander("⚙️ System Telemetry & Logs", expanded=False):
        meta_cols = st.columns(5)
        meta_cols[0].metric("Mode", mode if (result.rendered or domain in (None, "teaching")) else "Asset only")
        meta_cols[1].metric("Domain", domain or "general")
        meta_cols[2].metric("Scenes", len(result.plan.scenes))
        meta_cols[3].metric("Cost", f"${result.cost.total_usd:.4f}")
        meta_cols[4].metric("Voice", result.voice_used if result.rendered else "—")

        if result.voice_fallback_used:
            st.info("Chirp 3 HD unavailable — fell back to Neural2-C silently.")
        if result.veo_attempted and result.veo_fallback_used:
            st.info("Veo failed for some scenes — failed scenes fell back to Imagen + motion.")
        elif not result.veo_attempted and result.rendered:
            st.caption("Veo was disabled — all scenes used Imagen + motion.")

        sub_a, sub_b = st.columns(2)
        with sub_a:
            st.markdown("**Cost breakdown**")
            st.json({
                "director (Gemini)": f"${result.cost.director_usd:.4f}",
                "imagen": f"${result.cost.imagen_usd:.4f}",
                "veo": f"${result.cost.veo_usd:.4f}",
                "tts": f"${result.cost.tts_usd:.4f}",
                "total": f"${result.cost.total_usd:.4f}",
            })
        with sub_b:
            st.markdown("**Run metadata**")
            st.json({
                "run_id": result.run_id,
                "rendered": result.rendered,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "scene_images": [str(p) for p in result.scene_images],
                "final_video": str(result.final_video) if result.final_video else None,
            })

        st.markdown("**Director plan (full JSON)**")
        st.json(result.plan.model_dump())


# ====================================================================
# Campaign mode — dispatcher audit + dual-tab result
# ====================================================================

def _render_dispatcher_audit(job: "dispatcher_mod.DispatchedJob") -> None:
    """Live telemetry: which tasks the dispatcher caught vs bypassed."""
    st.markdown("#### 📡 Dispatcher audit trail")
    st.caption(
        "The Master Supervisor's tasks.md, parsed task-by-task. Lines marked "
        "**Catching** route into this node; **Bypassing** lines are forwarded "
        "to external specialist agents and never touch the Director."
    )

    log_lines = []
    for line in job.audit_log:
        if "Catching" in line or "Forwarding to Director" in line:
            log_lines.append(f"<span style='color:#10b981;font-weight:600'>✓ {line}</span>")
        elif "Bypassing" in line:
            log_lines.append(f"<span style='color:#94a3b8'>↳ {line}</span>")
        else:
            log_lines.append(f"<span style='color:#cbd5e1'>· {line}</span>")

    st.markdown(
        "<div style='background:#0f172a;border:1px solid #1e293b;border-radius:8px;"
        "padding:14px 18px;font-family:ui-monospace,monospace;font-size:13px;"
        "line-height:1.7'>"
        + "<br>".join(log_lines)
        + "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1, 2])
    cols[0].metric("Tasks discovered", len(job.visual_tasks) + len(job.bypassed_tasks))
    cols[1].metric("Caught for this node", len(job.visual_tasks))
    cols[2].metric(
        "Bypassed to other agents",
        ", ".join(t.assigned_to for t in job.bypassed_tasks) or "—",
    )


def _render_campaign_result(result: "DispatchedAssetResult") -> None:
    """Dual-tab result: rendered visual asset + the manifest spec document."""

    st.markdown(f"### 🏭 Campaign: `{result.campaign}` · Task {result.task.index}")
    st.markdown(f"**{result.task.title}**")
    st.caption(
        f"Run ID `{result.run_id}` · "
        f"rendered via **{result.image_generator_used}**"
        + (" (with Imagen 4 fallback)" if result.fallback_used else "")
        + f" · {result.elapsed_seconds:.1f}s · ${result.cost.total_usd:.4f}"
    )

    tab_assets, tab_manifest = st.tabs(["🖼️ Visual Assets", "📄 Campaign Manifest (.md)"])

    with tab_assets:
        if result.asset_path.exists():
            st.image(
                str(result.asset_path),
                caption=result.manifest.brief_one_liner,
                use_container_width=True,
            )
        else:
            st.warning(f"Asset file not found at {result.asset_path}")

        st.markdown("##### Brand palette resolved in this render")
        if result.manifest.palette:
            swatches = "".join(
                f"<div style='flex:1;min-width:120px;background:{c.hex_value};"
                f"color:#fff;padding:18px 14px;border-radius:6px;text-shadow:0 1px 2px rgba(0,0,0,0.4)'>"
                f"<div style='font-family:ui-monospace,monospace;font-size:12px;opacity:0.9'>{c.token}</div>"
                f"<div style='font-family:ui-monospace,monospace;font-weight:700;font-size:15px;letter-spacing:0.4px'>{c.hex_value}</div>"
                f"</div>"
                for c in result.manifest.palette
            )
            st.markdown(
                f"<div style='display:flex;gap:8px;flex-wrap:wrap'>{swatches}</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Asset file paths"):
            st.code(f"asset:    {result.asset_path}\nmanifest: {result.manifest_path}")

    with tab_manifest:
        manifest_md = result.manifest_path.read_text(encoding="utf-8")
        st.download_button(
            label="⬇️ Download manifest.md",
            data=manifest_md,
            file_name="manifest.md",
            mime="text/markdown",
            use_container_width=False,
        )
        st.markdown(manifest_md)


def _render_campaign_telemetry(result: "DispatchedAssetResult") -> None:
    """Collapsed telemetry for a campaign run — mirrors the video-mode block."""
    with st.expander("⚙️ System Telemetry & Logs", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Mode", "Campaign dispatch")
        cols[1].metric("Cost", f"${result.cost.total_usd:.4f}")
        cols[2].metric("Elapsed", f"{result.elapsed_seconds:.1f}s")
        cols[3].metric("Generator", result.image_generator_used)

        sub_a, sub_b = st.columns(2)
        with sub_a:
            st.markdown("**Cost breakdown**")
            st.json({
                "director (Gemini)": f"${result.cost.director_usd:.4f}",
                "imagen": f"${result.cost.imagen_usd:.4f}",
                "total": f"${result.cost.total_usd:.4f}",
            })
        with sub_b:
            st.markdown("**Run metadata**")
            st.json({
                "run_id": result.run_id,
                "campaign": result.campaign,
                "task_index": result.task.index,
                "asset_path": str(result.asset_path),
                "manifest_path": str(result.manifest_path),
                "fallback_used": result.fallback_used,
            })

        st.markdown("**Audit log (full)**")
        st.code("\n".join(result.audit_log))

        st.markdown("**Manifest (structured JSON)**")
        st.json(result.manifest.model_dump())


# ====================================================================
# Main UI
# ====================================================================

st.title("🎨 Multi-Domain Visual Factory")
st.caption("Pick a vertical. Describe your idea. Get the right deliverable.")

_init_state()


with st.sidebar:
    st.header("Workspace")

    # ---------- Campaign Profile (multi-agent dispatch mode) ----------
    profile_paths = dispatcher_mod.list_profiles(CAMPAIGN_ROOT)
    profile_labels = ["— Manual mode (no campaign) —"] + [p.name for p in profile_paths]
    profile_choice = st.selectbox(
        "🏭 Campaign Profile",
        options=profile_labels,
        index=0,
        help=(
            "Pick a folder from config/marketing/ to enter campaign-dispatch "
            "mode. The dispatcher reads tasks.md, filters tasks targeting "
            "mcp_visual_factory, and routes only those to the Director."
        ),
    )
    campaign_mode = profile_choice != profile_labels[0]
    active_profile_path: Path | None = None
    if campaign_mode:
        active_profile_path = next(
            p for p in profile_paths if p.name == profile_choice
        )
        st.caption(
            f"Dispatching from `{active_profile_path}/tasks.md`. "
            "Manual topic/audience inputs below are ignored."
        )
        # Preview the dispatcher audit live as soon as a profile is picked,
        # so the user sees the routing decision before clicking Run.
        try:
            preview_job = dispatcher_mod.dispatch_campaign(active_profile_path)
            st.session_state.dispatcher_preview = preview_job
        except Exception as e:
            st.error(f"Failed to load profile: {type(e).__name__}: {e}")
            st.session_state.dispatcher_preview = None
            campaign_mode = False
    else:
        st.session_state.dispatcher_preview = None

    st.divider()

    domain_label = st.selectbox(
        "🎯 Domain",
        options=DOMAIN_CHOICES,
        index=0,
        help=(
            "Each domain reshapes the Director's brief and the workspace layout. "
            "Interior gets a variant grid. Branding gets an Asset Kit. "
            "Marketing gets a Slide Deck Blueprint. Teaching keeps the "
            "interactive lesson flow."
        ),
    )
    domain = DOMAIN_KEYS[domain_label]
    if domain:
        st.caption(f"Domain Director: **{domain_label}**")

    prompt_cfg = DOMAIN_PROMPT_CONFIG.get(domain, DOMAIN_PROMPT_CONFIG[None])

    topic = st.text_input(
        prompt_cfg["topic_label"],
        value=prompt_cfg["topic_default"],
        help=prompt_cfg["topic_help"],
    )

    secondary_topic = ""
    if prompt_cfg["secondary"]:
        secondary_topic = st.text_input(
            prompt_cfg["secondary"]["label"],
            value=prompt_cfg["secondary"]["default"],
            help=prompt_cfg["secondary"]["help"],
        )

    if domain == "teaching":
        audience = st.selectbox(
            "Audience",
            options=["6-year-old", "10-year-old", "12-year-old", "adult"],
            index=0,
        )
    elif domain == "branding":
        audience = "brand client / founder team"
    elif domain == "interior_design":
        audience = "design client / homeowner"
    elif domain == "marketing":
        audience = "executive audience"
    else:
        audience = st.selectbox(
            "Audience",
            options=["6-year-old", "10-year-old", "12-year-old", "adult"],
            index=0,
        )

    # ---------- Render toggle ----------
    st.divider()
    render_video = st.checkbox(
        "🎬 Render Video Clip Sequence",
        value=False if domain in ("interior_design", "branding", "marketing") else True,
        help=(
            "ON: full pipeline — TTS narration + MoviePy assembly + final mp4. "
            "OFF: assets-only mode — instant grid output, no TTS, no video. "
            "Use OFF for rapid iteration on visuals; flip ON for the final render."
        ),
    )
    if render_video:
        st.caption("Full video render: Director + visuals + voice + assembly.")
    else:
        st.caption("Asset-only mode: Director + visuals only. Fast + cheap.")

    # ---------- Playback mode (only meaningful when rendered) ----------
    if render_video:
        st.divider()
        playback_mode = st.radio(
            "Playback mode",
            options=["Linear", "Interactive"],
            index=1 if domain in (None, "teaching") else 0,
            help="Linear: full assembled .mp4. Interactive: per-scene + checkpoints (best for Teaching).",
        )
    else:
        playback_mode = "Linear"

    # ---------- Voice override ----------
    st.divider()
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

    # ---------- Image generator ----------
    st.divider()
    image_generator_label = st.selectbox(
        "🖼️ Image generator",
        options=[
            "Imagen 4 (Google · fast · reliable)",
            "Nano Banana 2 (Gemini · multimodal · character consistency)",
            "FLUX.1 schnell (open-source · via MCP/HF · may queue)",
        ],
        index=0,
        help="Every choice falls back to Imagen 4 per-scene on failure.",
    )
    if image_generator_label.startswith("Imagen"):
        image_generator = "imagen"
    elif image_generator_label.startswith("Nano Banana"):
        image_generator = "nano_banana"
    else:
        image_generator = "flux"

    # ---------- Veo (only when rendering) ----------
    st.divider()
    if render_video:
        allow_veo = st.checkbox(
            "🎥 Allow Veo (expensive)",
            value=False,
            help="Off: Imagen for every scene, ~$0.20/video. On: Veo for narrative scenes (~$5–15).",
        )
        if allow_veo:
            st.warning("Veo enabled — story-type videos may cost $5–15.")
    else:
        allow_veo = False

    st.divider()
    if campaign_mode:
        btn_label = "Run Campaign Pass"
    elif render_video:
        btn_label = "Generate video"
    else:
        btn_label = "Generate"
    generate_btn = st.button(
        btn_label,
        type="primary",
        use_container_width=True,
    )


# ====================================================================
# Generation
# ====================================================================

if generate_btn:
    progress_box = st.empty()
    log_expander = st.expander("Generation log", expanded=True)

    def _progress(msg: str) -> None:
        progress_box.markdown(f"**{msg}**")
        with log_expander:
            st.write(msg)

    # -----------------------------------------------------------------
    # Campaign-mode dispatch: skip the manual topic/audience flow entirely.
    # The job already came from tasks.md; we just need to execute it.
    # -----------------------------------------------------------------
    if campaign_mode and active_profile_path is not None:
        try:
            with st.spinner("Dispatching campaign tasks..."):
                job = dispatcher_mod.dispatch_campaign(active_profile_path)

            if not job.has_work:
                st.warning(
                    f"Profile '{job.campaign}' has no tasks assigned to "
                    "mcp_visual_factory. Nothing to render."
                )
            else:
                # One pass per visual task (usually one, occasionally a few).
                results: list[DispatchedAssetResult] = []
                for task in job.visual_tasks:
                    with st.spinner(f"Running Task {task.index} — {task.title}"):
                        results.append(generate_dispatched_asset(
                            job=job, task=task,
                            image_generator=image_generator,
                            progress=_progress,
                        ))
                # Store the last result (most campaigns dispatch a single hero task).
                st.session_state.campaign_result = results[-1]
                st.session_state.result = None  # park the manual-mode result
                progress_box.markdown(
                    f"**✅ Campaign pass complete · "
                    f"{len(results)} asset(s) + manifest(s) written.**"
                )
        except Exception as e:
            st.error(f"Campaign dispatch failed: {type(e).__name__}: {e}")
            st.exception(e)
    else:
        # -----------------------------------------------------------------
        # Manual mode: existing multi-domain video / asset generator path.
        # -----------------------------------------------------------------
        voice_name: str | None
        if voice_override.startswith("auto"):
            voice_name = None
        elif voice_override.startswith("en-US-Neural2-C"):
            voice_name = "en-US-Neural2-C"
        else:
            voice_name = voice_override

        composed_topic = topic
        if secondary_topic:
            composed_topic = f"{topic} — {secondary_topic}"

        try:
            with st.spinner("Generating..."):
                result = generate_video(
                    topic=composed_topic,
                    audience=audience,
                    content_type=None,
                    voice_name=voice_name,
                    allow_veo=allow_veo,
                    image_generator=image_generator,
                    domain=domain,
                    render_video=render_video,
                    progress=_progress,
                )
            st.session_state.result = result
            st.session_state.campaign_result = None
            _reset_playback_state()
            progress_box.markdown(
                f"**✅ Done in {result.elapsed_seconds:.1f}s · run `{result.run_id}`**"
            )
        except Exception as e:
            st.error(f"Generation failed: {type(e).__name__}: {e}")
            st.exception(e)


# ====================================================================
# Main panel — domain-aware workspace routing
# ====================================================================

campaign_result: DispatchedAssetResult | None = st.session_state.campaign_result
result: RunResult | None = st.session_state.result
preview_job = st.session_state.dispatcher_preview

# ----- Campaign mode: dispatcher audit (always) + dual-tab result (when ready)
if campaign_mode:
    if preview_job is not None:
        _render_dispatcher_audit(preview_job)
        st.divider()

    if campaign_result is None:
        st.info(
            "Review the dispatcher audit above. When ready, click "
            "**Run Campaign Pass** in the sidebar — the Manifest Director "
            "will render the caught task(s) into `outputs/run_<id>/`."
        )
    else:
        _render_campaign_result(campaign_result)
        st.divider()
        _render_campaign_telemetry(campaign_result)

# ----- Manual mode: existing multi-domain workspace
elif result is None:
    st.info("Pick a domain and a topic in the sidebar, then click **Generate**.")
    st.markdown(
        """
        | Domain | Workspace | Example topic |
        |---|---|---|
        | **Interior Design** | 3-up layout variants, material board, layer-modification stub | `Mid-century Scandinavian living room` |
        | **Logo / Branding** | Brand mark + palette + typography + archetype | `Nexora — confident heritage tech brand` |
        | **Marketing / Presentation** | Sequential slide cards + narrative thesis | `Launching Code Mode for AI workflows` |
        | **Teaching** | Lesson + checkpoints + end quiz | `Fractions for 6-year-olds` |
        | **General** | Original video pipeline | anything |
        | **🏭 Campaign Profile** | Multi-agent dispatch → hero asset + manifest.md | pick from `config/marketing/` |
        """
    )
else:
    if domain == "interior_design":
        _render_workspace_interior(result)
    elif domain == "branding":
        _render_workspace_branding(result)
    elif domain == "marketing":
        _render_workspace_marketing(result)
    elif domain == "teaching":
        _render_workspace_teaching(result, playback_mode)
    else:
        _render_workspace_general(result, playback_mode)

    st.divider()
    _render_telemetry(result, playback_mode, domain)
