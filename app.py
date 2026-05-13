"""Streamlit UI for the AI Educational Video Generator."""
from __future__ import annotations

import logging

import streamlit as st

from src.pipeline import generate_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(page_title="AI Video Generator", page_icon="🎬", layout="wide")

st.title("🎬 AI Educational Video Generator")
st.caption("Google AI stack · Gemini · Imagen · Veo · Cloud TTS")

# ---------- Sidebar ----------
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
    voice_name = st.selectbox(
        "Voice",
        options=[
            "en-US-Neural2-C",
            "en-US-Neural2-A",
            "en-US-Neural2-F",
            "en-IN-Neural2-A",
            "en-IN-Neural2-B",
            "en-GB-Neural2-A",
        ],
        index=0,
        help="Cloud TTS Neural2 voice",
    )

    st.divider()
    allow_veo = st.checkbox(
        "🎥 Allow Veo (expensive)",
        value=False,
        help=(
            "Veo generates real video clips for narrative scenes (~$3 per scene). "
            "When OFF (default), every scene uses Imagen + motion (~$0.04 per scene)."
        ),
    )
    if allow_veo:
        st.warning("Veo is enabled — a story-type video may cost $5–15.")
    else:
        st.caption("All scenes will use Imagen + motion. Predictable ~$0.20 per video.")

    generate_btn = st.button("Generate video", type="primary", use_container_width=True)

# ---------- Main ----------
if generate_btn:
    progress_box = st.empty()
    log_expander = st.expander("Generation log", expanded=True)

    def progress(msg: str) -> None:
        progress_box.markdown(f"**{msg}**")
        with log_expander:
            st.write(msg)

    try:
        with st.spinner("Working..."):
            result = generate_video(
                topic=topic,
                audience=audience,
                content_type=content_type if content_type != "auto" else None,
                voice_name=voice_name,
                allow_veo=allow_veo,
                progress=progress,
            )

        progress_box.markdown(
            f"**✅ Done in {result.elapsed_seconds:.1f}s · run `{result.run_id}`**"
        )

        col_video, col_meta = st.columns([2, 1])

        with col_video:
            st.subheader(result.plan.title)
            st.video(str(result.final_video))
            with open(result.final_video, "rb") as f:
                st.download_button(
                    "⬇️ Download MP4",
                    data=f.read(),
                    file_name=f"{result.run_id}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

        with col_meta:
            st.subheader("Run details")
            st.metric("Content type", result.plan.content_type)
            st.metric("Scenes", len(result.plan.scenes))
            st.metric("Estimated cost", f"${result.cost.total_usd:.4f}")

            if result.veo_attempted and result.veo_fallback_used:
                st.info(
                    "Veo succeeded for some scenes and failed for others — "
                    "failed scenes fell back to Imagen + motion. Visuals may look "
                    "inconsistent across scenes."
                )
            elif not result.veo_attempted:
                st.caption(
                    "Veo was disabled — all scenes used Imagen + motion. "
                    "Predictable cost, consistent look."
                )

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

    except Exception as e:
        st.error(f"Generation failed: {type(e).__name__}: {e}")
        st.exception(e)

else:
    st.info("Enter a topic in the sidebar and click **Generate video** to start.")
    st.markdown(
        """
        **Example topics that exercise each content type:**

        | Topic | Audience | Content type |
        |---|---|---|
        | `Fractions` | `6-year-old` | math |
        | `Hampi` | `adult` | place |
        | `The Lion and the Mouse` | `10-year-old` | story |

        The Director (Gemini) auto-detects content type if you leave it on `auto`.
        """
    )
