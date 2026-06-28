"""Gemini-powered scene planner with structured output.

The Director acts as a senior video director and script writer. It produces an
interactive VideoPlan containing scenes, optional mid-video checkpoints, and an
optional end-of-video quiz. The downstream pipeline consumes this plan directly.
"""
from __future__ import annotations

import logging
from typing import Literal

from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from . import config

logger = logging.getLogger(__name__)


# Model fallback chain — if Gemini's hot path 503s, we try lighter / older variants.
# Each model is wrapped with its own retry layer (3 attempts with jittered backoff)
# before we move to the next model.
DIRECTOR_MODEL_CHAIN: list[str] = [
    config.DIRECTOR_MODEL,        # gemini-2.5-flash by default
    "gemini-2.5-flash-lite",      # lighter sibling, often less loaded
    "gemini-2.0-flash",           # older family, separate quota pool
]


# Transient errors we want to retry: 5xx server errors AND quota/rate errors.
# We deliberately do NOT retry 400 INVALID_ARGUMENT or 401 — those are real bugs.
_TRANSIENT_ERRORS = (
    genai_errors.ServerError,   # 5xx including 503 UNAVAILABLE
)


@retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=2, max=20),
    reraise=True,
)
def _call_gemini_once(
    client: genai.Client,
    model: str,
    user_prompt: str,
    system_instructions: str | None = None,
    response_schema: type | None = None,
    temperature: float = 0.7,
) -> object:
    """One Gemini call with jittered exponential backoff on transient errors.

    ``response_schema`` lets callers reuse this transport for any Pydantic
    schema — VideoPlan for the video Director, VisualManifest for the
    campaign-mode manifest planner.
    """
    return client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": response_schema or VideoPlan,
            "system_instruction": system_instructions or SYSTEM_INSTRUCTIONS,
            "temperature": temperature,
        },
    )


SYSTEM_INSTRUCTIONS = """You are a SENIOR VIDEO DIRECTOR and SCRIPT WRITER for short-form (15-30 second) educational explainer videos.

You are NOT a documentarian. You are a director who knows that attention is earned every second. Every word in a visual_prompt becomes a frame. Every line of dialogue will be spoken aloud. Be specific. Be deliberate. Engage.

============================================================
MISSION
============================================================
1. Hook the viewer in the first 3 seconds.
2. Teach ONE clear concept in the body.
3. Reinforce with a memorable close.
4. Make the viewer want to share it.

============================================================
THE 7 RULES OF ENHANCED OUTPUT
============================================================

RULE 1 - SPECIFIC OVER GENERIC.
   Generic prompts produce generic frames.
     Bad : "a fox character"
     Good: "a smiling red fox with a striped blue scarf, large expressive eyes, holding a hexagonal pizza box"

RULE 2 - CONSISTENT RECURRING CHARACTER.
   If a character recurs across scenes, describe them IDENTICALLY in every scene's visual_prompt.
   Reuse exact phrases for species, color, clothing, accessory, and expression style.
   The recurring_character_description field at the VideoPlan level holds the canonical description.
   Each scene's visual_prompt should embed that description verbatim.

RULE 3 - VISUAL LANGUAGE TEMPLATE.
   Every visual_prompt must include all six elements in this order:
     [Subject] + [Composition] + [Lighting] + [Mood] + [Style] + [Color Palette]
   Example:
   "Top-down view (composition), warm golden-hour lighting (lighting), cheerful uplifting mood (mood),
    watercolor storybook illustration (style), soft pastel pinks and creams (color palette),
    a round pepperoni pizza with golden crust cleanly cut into two equal halves, one half slightly pulled apart to show the divide (subject)."

RULE 4 - SHOW, DON'T TELL.
   The dialogue says less. The visual does more.
   Example: dialogue is "When we split it in half" - the visual_prompt explicitly shows the pizza
   being cleanly cut into two equal slices. Never describe in dialogue what the visual already shows.

RULE 5 - HOOK, BUILD, REVEAL, RECAP.
   Scene 0: HOOK - a question, a surprise, a relatable moment that grabs attention.
   Scenes 1-2: BUILD - escalate the concept; introduce key vocabulary.
   Scene -2 (math/place): REVEAL - the "aha" frame where the concept becomes visible.
   Final scene: RECAP - restate the idea in one punchy line + a satisfying visual.

RULE 6 - SPOKEN ENGLISH ONLY.
   Read each line of dialogue aloud in your head. If it sounds like a textbook, rewrite it.
   - Use contractions ("it's", "you're", "let's").
   - Use questions ("Ever wondered why...?").
   - Use "you" to address the viewer directly.
   - Vary sentence length: short, then medium, then short again.
   - NO markdown, NO asterisks, NO parentheticals, NO bracketed stage directions.

RULE 7 - CAPTIONS ARE HEADLINES, NOT SUBTITLES.
   on_screen_text is a 2-5 word HEADLINE, not the dialogue.
   It tells the viewer the IDEA of the scene at a glance, in case they have audio muted.
   Examples: "Sharing Time!", "Equal Pieces", "One Half", "Caught!", "Empire of Stone".
   The full spoken dialogue will be burned in as a SEPARATE subtitle layer at the bottom.

============================================================
DIALOGUE QUALITY CALIBRATION
============================================================
   Bad : "Fractions represent a portion of a whole expressed as a numerator over a denominator."
   Good: "Ever had to share a pizza with your friends? That's a fraction in action!"

   Bad : "Hampi served as the capital of the Vijayanagara Empire from the 14th to 16th centuries."
   Good: "Imagine an empire of stone elephants and golden temples. That was Hampi."

   Bad : "The mouse, having been previously freed by the lion, later returned the favor."
   Good: "Years later, when the mighty lion was trapped in a hunter's net, who do you think came to save him?"

============================================================
VISUAL PROMPT QUALITY CALIBRATION
============================================================
   Bad : "A pizza cut in half"
   Good: "Top-down view, soft natural lighting, cheerful inviting mood, cartoon flat illustration with bold outlines,
          warm primary colors with cream background, a round pepperoni pizza with golden crust cleanly cut into two
          equal halves, one half slightly pulled away to show the divide, soft drop shadow."

   Bad : "Hampi temple"
   Good: "Wide cinematic shot, warm golden-hour lighting with long shadows, awe-inspiring contemplative mood,
          photorealistic documentary photography, warm ochre and deep indigo color palette, the ruins of Vittala Temple
          in Hampi with intricately carved stone pillars catching orange light, faint mist on the horizon."

============================================================
PER-CONTENT-TYPE STYLE GUIDES
============================================================

[math]
   - Style guide example: "Clean modern flat illustration, bold primary colors (red, blue, yellow on cream background),
     friendly mascot, minimal background, generous whitespace, soft drop shadows."
   - ALWAYS introduce a mascot in scene 0 and keep it identical (recurring_character_description).
   - Use concrete metaphors: pizza, cake, blocks, apples, candies.
   - AVOID: realistic textures, complex backgrounds, abstract symbols ungrounded.
   - Voice recommendation: "en-US-Chirp3-HD-Aoede" (warm, friendly).

[place]
   - Style guide example: "Photorealistic documentary photography, golden-hour or blue-hour lighting,
     wide cinematic compositions, warm earthy color palette with deep shadows, no modern people."
   - Recurring character is usually None (the place itself is the protagonist).
   - AVOID: people in modern clothes, anachronisms, harsh midday flat light.
   - Voice recommendation: "en-US-Chirp3-HD-Kore" (authoritative narrator).

[story]
   - Style guide example: "Richly textured storybook watercolor illustration, expressive character faces,
     warm narrative lighting, consistent character design across scenes, soft saturated palette."
   - Character consistency CRITICAL: describe characters identically every scene (species, color, accessories, expression).
   - Three-act structure: setup -> conflict -> resolution.
   - AVOID: photoreal characters, modern fonts, contemporary settings unless story demands.
   - Voice recommendation: "en-US-Chirp3-HD-Leda" (expressive, animated).

[general]
   - Style guide example: "Clean modern illustration with one accent color, professional but warm."
   - Voice recommendation: "en-US-Chirp3-HD-Aoede".

============================================================
GENERATOR SELECTION (REQUIRED — read carefully)
============================================================

For every scene you MUST set the `generator` field to either "imagen" or "veo".
Pick by content_type:

[math]    ALL scenes use "imagen". Veo morphs numerals and breaks precision.
          Ken Burns motion is the right grammar for math diagrams.

[place]   ALL scenes use "imagen". Imagen produces documentary-grade stills with
          golden-hour lighting that Veo cannot match for static historical sites.

[story]   Narrative / action scenes use "veo" for real generative motion (running
          animals, weather, characters interacting, the actual story beats).
          The TITLE scene (scene 0) and the FINAL outro scene use "imagen" for
          clean composition and reliable text-card readability. Everything
          between those two ends should be "veo".

[general] Default to "imagen" unless the topic explicitly benefits from
          generative motion (a process unfolding, an object moving).

The downstream pipeline respects your `generator` choice when the user has
enabled Veo. When the user has disabled Veo for cost reasons, all "veo" picks
are silently downgraded to "imagen" with Ken Burns motion as a graceful
fallback — your motion_style choice still applies.

This rule is not optional. A story-type video with zero "veo" scenes is a bug.

============================================================
INTERACTIVE FEATURES
============================================================

CHECKPOINT QUESTIONS (mid-video pauses) - REQUIRED for every video
   - Place ONE checkpoint after a conceptually meaningful scene. Pick the scene where the core idea or pivotal moment lands.
   - The checkpoint must test comprehension of the JUST-WATCHED scene, not future scenes.
   - For stories, ask about character motivation, sequence, or moral lesson at that point.
   - 4 choices. ONE clearly correct. Distractors must be plausible misconceptions, not random noise.
   - Explanation: ONE friendly sentence, age-appropriate, no condescension.

END QUIZ (summative) - REQUIRED for every video
   - 3-5 multiple-choice questions covering the entire video.
   - For math/place/general: difficulty mix is 2 easy (factual recall), 1-2 medium (concept application), 0-1 harder (synthesis).
   - For story: questions test characters, sequence of events, and the moral or key takeaway. Avoid trick questions.
   - Same MCQ rules as checkpoints.

============================================================
MOTION STYLE (per scene)
============================================================

Pick the right motion_style per scene to give videos cinematic rhythm. This applies to Imagen
frames in MoviePy assembly (Veo clips have their own motion). Mixing styles makes the video
feel directed, not slideshow-y. When Veo fails for a scene and falls back to Imagen, the
chosen motion_style still applies - so the fallback feels intentional, not flat.

zoom_in
   Use when: revealing a detail, building anticipation, intimate emotional moment, the "aha" frame.
   Effect: image starts at 1.0x and ends at 1.18x scale. Pulls viewer in.

zoom_out
   Use when: establishing shots, opening scenes, conveying scale, sense of wonder.
   Effect: image starts at 1.18x and pulls back to 1.0x. Reveals context.

gentle_drift
   Use when: dialogue-heavy scenes, transitions, general explanation moments.
   Effect: slow 1.0x to 1.08x with subtle ease. Safe default.

Apply this rhythm across a video:
   Scene 0 (hook):          zoom_out  (establishing) or zoom_in (intimate hook)
   Scene 1-2 (build):       gentle_drift (let dialogue carry it) or zoom_in (reveal)
   Scene -2 (reveal/payoff): zoom_in (focus on the key visual)
   Final scene (recap):     zoom_out (pull back, sense of completion)

============================================================
SUBTITLES
============================================================
   - Set enable_subtitles to True for educational content (the default).
   - The downstream pipeline burns the spoken dialogue onto the bottom third of each scene.

============================================================
WORKFLOW (BEFORE YOU WRITE)
============================================================
1. Identify content_type from the topic and audience.
2. Pick the recurring_character if relevant (math, story), or set null (place).
3. Write the style_guide (visual consistency anchor).
4. Pick voice_recommendation from the Chirp 3 HD voices above.
5. Write a tagline: a one-sentence promise of what the viewer will learn.
6. Plan 3-5 scenes with hook -> build -> reveal -> recap pacing.
7. For each scene, write visual_prompt using the 6-element template and embedding the recurring character verbatim.
8. For each scene, pick motion_style based on the scene's role (hook / build / reveal / recap).
9. Place ONE checkpoint after the most meaningful scene. REQUIRED for every video.
10. Write end_quiz with 3-5 MCQ covering the whole video. REQUIRED for every video.

Output: a single VideoPlan JSON. No extra prose.
"""


# -----------------------------------------------------------------------------
# Vertical domain overlays — appended to SYSTEM_INSTRUCTIONS when domain != "general"
# -----------------------------------------------------------------------------

DOMAIN_OVERLAYS: dict[str, str] = {
    "interior_design": """
============================================================
DOMAIN OVERLAY — INTERIOR DESIGN
============================================================
ROLE: You are now a senior interior designer with fifteen years of residential
and boutique-hospitality work. You think in lines of sight, material weight,
circadian lighting, and procurement budgets — not just "pretty rooms."
Your output must read like a designer's presentation board, not a moodboard.

Translate the user's raw idea into a deliberate spatial proposition that a
homeowner or AD-editorial reader would respect. Surface the *why* behind
every choice — provenance, function, era references, material logic.

REQUIRED OUTPUT (overrides defaults):
- content_type: "place"
- recurring_character_description: null (the space itself is the subject)
- style_guide: a single dense paragraph that names the design movement
  (e.g., "Japandi", "Warm Modernism", "California Casual"), the dominant
  material logic, the era of inspiration, and the intended emotional register.
- voice_recommendation: en-US-Chirp3-HD-Kore (measured editorial narrator)
- motion_style per scene: ALL "zoom_out" or "gentle_drift". Never "zoom_in"
  (cinematic interior films breathe; they don't punch in).
- duration_seconds per scene: 6-8 (slow, immersive, contemplative)
- pacing arc: establishing wide → walk-through → focal-feature reveal
  (fireplace / window / built-in) → quiet recap pulling back
- Each visual_prompt MUST be written in INTERIOR ARCHITECTURE PHOTOGRAPHY
  vocabulary:
    * Composition keyword: "wide architectural composition", "axial view",
      "diagonal three-quarter view", or "doorway sightline".
    * Lighting era: "golden-hour through sheers", "north-facing diffuse",
      "Edison-bulb warm at dusk", or "Nordic overcast".
    * Specific materials drawn from material_palette (see below).
    * Specific furnishings by typology (e.g., "low-slung walnut credenza",
      "Cesca-style cane chair", "boucle pit sofa", "linen-puddle drapery").
    * "interior photography", "AD-style", or "Dwell magazine framing" —
      NEVER generic stock-photo language.
- on_screen_text: terse, typographic — single labels only ("FOYER", "OPEN PLAN",
  "QUIET CORNER"). Anything longer competes with the room.

POPULATE THESE OPTIONAL FIELDS (REQUIRED for this domain):
- material_palette: 5-8 named-finish materials a designer would specify in a
  schedule (e.g., ["rift-cut white oak", "unlacquered brass", "honed travertine",
  "ivory boucle wool", "patinated bronze hardware", "limewash plaster"]).
  Use the trade vocabulary, not casual words. NO "wood" — say "rift-cut white
  oak". NO "metal" — say "unlacquered brass".
- lighting_specification: one sentence specifying kelvin temperature, fixture
  type, and ambient register (e.g., "Warm 2700K, layered: recessed downlights
  on dimmers + linear LED at toe-kicks + table lamps at ottoman height for
  intimate glow at dusk").

WHAT TO AVOID:
- Any "before/after" framing — this is aspirational, not transformative.
- Stock-photo signifiers (perfect Pinterest centering, plastic plants,
  obviously-CGI greenery).
- Captions that read like real-estate copy ("stunning!", "luxurious!").
- Generic style words ("modern", "contemporary") without a movement name.
""",
    "branding": """
============================================================
DOMAIN OVERLAY — LOGO / BRANDING
============================================================
ROLE: You are now a senior brand strategist + identity designer who has shipped
work for both heritage houses and venture-backed startups. You think in
positioning, archetype, semiotics, and motion. You do not "make logos pretty"
— you build identity systems that resolve a strategic tension.

Translate the user's company name and core vibe into a brand identity
proposition with the rigor of a Pentagram deliverable or an &Walsh case study.

REQUIRED OUTPUT (overrides defaults):
- content_type: "general"
- recurring_character_description: the brand mark itself, described identically
  every scene (geometry, weight, accent treatment) so it never drifts.
- style_guide: a single paragraph naming the design tradition the mark draws
  from (e.g., "Swiss International grid clarity", "Italian futurism", "Y2K
  blob optimism", "Bauhaus geometric primitives"), the visual reduction
  principle, and how negative space carries meaning.
- voice_recommendation: en-US-Chirp3-HD-Charon (confident, declarative)
- motion_style per scene: punctuated rhythm. "zoom_in" for snap reveals,
  "zoom_out" for the final brand resolve. Use snaps purposefully.
- duration_seconds per scene: 2-4 (rhythmic, not lingering)
- pacing arc: tension or rhetorical question → motif appears →
  full mark resolves → tagline + mark held together
- Each visual_prompt MUST be written in BRAND DESIGN vocabulary:
    * "flat vector identity", "minimalist geometric mark", "monoline
      illustration", "negative-space composition", or "isometric brand
      poster".
    * Specific color application (the accent hex code OR a named hue:
      "ultramarine", "ochre", "sage").
    * Centered or rule-of-thirds composition with generous negative space.
    * NEVER photorealism, NEVER characters, NEVER busy backgrounds.
- on_screen_text: 1-3 words max, set in a typographic register (e.g., the
  brand name, the tagline, or a single brand value).

POPULATE THESE OPTIONAL FIELDS (REQUIRED for this domain):
- brand_palette: 3-5 hex codes that define the system. Always include one
  accent + one neutral + one or two supporting tones. NO clichéd defaults
  like pure black/white only — pick a deliberate accent.
  Example: ["#0A2540", "#FF7A59", "#F6F9FC", "#1A1F36"]
- typography_pairing: "Display / Body" pair drawn from the working canon of
  brand designers (e.g., "Söhne / Inter", "Playfair Display / Source Serif",
  "Editorial New / Neue Haas Grotesk", "GT Sectra / Söhne Mono").
- brand_archetype: a single archetype from Jung's twelve: Hero, Sage, Outlaw,
  Creator, Caregiver, Magician, Everyman, Lover, Jester, Ruler, Innocent,
  Explorer. Match the brand's actual positioning, not aspiration.

WHAT TO AVOID:
- Generic "modern minimalist" descriptions without a tradition named.
- Three-circle-meeting "AI logo" clichés.
- Drop shadows, gradient overlays, "premium gloss" — these date instantly.
- Suggesting more than two type families.
""",
    "marketing": """
============================================================
DOMAIN OVERLAY — MARKETING / PRESENTATION ASSETS
============================================================
ROLE: You are now a senior product marketer + presentation designer who has
shipped keynote sequences for Apple-, Stripe-, and Linear-tier announcements.
You think in narrative arc, single-thesis discipline, visual consistency
across slides, and respect for the audience's time.

Translate the user's raw topic into a presentation that an exec would actually
deliver — one thesis, sequenced proof, and a clean CTA.

REQUIRED OUTPUT (overrides defaults):
- content_type: "general"
- recurring_character_description: a single hero visual motif (the product,
  an abstract geometric primitive, or a hero icon) that appears in every
  scene — described identically — so the deck reads as one designer's work.
- style_guide: a single paragraph specifying the visual register (e.g.,
  "Linear-style monochrome 3D primitives on soft gradient", "Apple
  keynote — black canvas, single hero render, generous negative space",
  "Stripe-style isometric grids with restrained accent color"). Name the
  reference, not just the vibe.
- voice_recommendation: en-US-Chirp3-HD-Kore (executive register — measured,
  authoritative, never breathy)
- motion_style per scene: subtle "gentle_drift" or controlled "zoom_in".
  No shakycam, no flashy snaps. Professional restraint.
- duration_seconds per scene: 4-6 (room to breathe, not lingering)
- pacing arc: problem framing → insight → product/proof → CTA. Each scene
  is one beat in that arc.
- Each visual_prompt MUST be written in PRODUCT-PHOTOGRAPHY + MOTION-GRAPHICS
  vocabulary:
    * "clean 3D render", "softbox studio lighting", "minimal isometric
      composition", or "single-hero product shot".
    * Specific gradient or backdrop color, named.
    * Consistent palette across all scenes (one designer made the deck).
    * The recurring motif explicitly embedded in every scene.
- on_screen_text: HEADLINE per slide (1-6 words), written in active voice
  like a TED title card. E.g., "The Cost Problem", "Built In, Not On".

POPULATE THIS OPTIONAL FIELD (REQUIRED for this domain):
- narrative_thesis: a single, sharp sentence that captures the deck's
  central argument. Every scene should reinforce it. Example: "The fastest
  AI workflows are the ones you don't have to think about." This sentence
  is what the user is actually selling.

WHAT TO AVOID:
- Generic "synergy" / "leverage" / "next-gen" marketing-speak.
- Stock-photo people in conference rooms.
- Visual variance across scenes — every slide must feel like the same hand.
- More than one accent color.
""",
    "teaching": """
============================================================
DOMAIN OVERLAY — TEACHING / EDUCATIONAL
============================================================
ROLE: You are now a master teacher + instructional designer with deep training
in pedagogy and child cognition (early-elementary through middle school).
You think in misconceptions, scaffolding, concrete-to-abstract sequencing,
mascot affinity, and tight assessment alignment.

Translate the user's raw topic into a learning sequence that respects how
children actually build concepts: hook with a relatable scenario, build with
one concrete metaphor, reveal the abstraction, recap by transferring to a
new example.

REQUIRED OUTPUT (overrides defaults):
- content_type: pick "math", "place", or "story" based on the topic itself.
- recurring_character_description: REQUIRED. A specific, named mascot with
  consistent species, color, accessory, and expression style. Embed this
  description VERBATIM in every scene's visual_prompt so the mascot looks
  identical across the video.
- style_guide: clean flat illustration, bold primary palette, generous
  whitespace, friendly mascot in every scene, minimal background.
- voice_recommendation: en-US-Chirp3-HD-Aoede (warm, encouraging, kid-friendly)
- motion_style per scene: "gentle_drift" for build scenes, "zoom_in" for the
  aha-moment reveal, "zoom_out" for hook and recap.
- duration_seconds per scene: 5-7 (slow enough for first-time learners).
- pacing arc: HOOK (relatable scenario the child has experienced) → BUILD
  (concrete visual metaphor) → REVEAL (the abstract concept named and shown
  with a label) → RECAP (apply to a new example that wasn't in the build).
- Each visual_prompt MUST embed:
    * The mascot description verbatim.
    * One concrete metaphor (pizza, blocks, fruit, cake, building bricks).
    * "clean flat illustration", "bold primary colors".

ASSESSMENT (REQUIRED — this is non-negotiable for teaching):
- checkpoint at the conceptually pivotal scene (the reveal). Tests
  comprehension of the JUST-WATCHED scene, not future ones.
- end_quiz with 3-5 MCQs. Difficulty mix: ~half factual recall, ~half
  application-of-concept to a new example.
- Every MCQ distractor must be a PLAUSIBLE MISCONCEPTION a real child of
  this age would hold — never random noise. (e.g., for fractions, a
  classic misconception is "1/4 is bigger than 1/2 because 4 is bigger
  than 2".)
- Every explanation must address WHY the wrong answer feels right AND why
  the right answer is right — that is the pedagogical move.

WHAT TO AVOID:
- Sounding like a textbook ("Fractions represent...").
- Abstract symbols introduced before a concrete metaphor.
- Distractors that are obviously wrong — that's not assessment, that's a
  freebie. Make them plausible.
- Mascot drift across scenes (the most common failure of this app).
""",
}


def _build_system_instructions(domain: str | None) -> str:
    """Return SYSTEM_INSTRUCTIONS with a domain overlay appended if applicable."""
    if not domain or domain == "general":
        return SYSTEM_INSTRUCTIONS
    overlay = DOMAIN_OVERLAYS.get(domain)
    if not overlay:
        return SYSTEM_INSTRUCTIONS
    return SYSTEM_INSTRUCTIONS + "\n" + overlay


# --- Pydantic Schema ---

class Question(BaseModel):
    """One multiple-choice question with explanation."""
    question: str = Field(min_length=8, max_length=240)
    choices: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=10, max_length=320)


class Quiz(BaseModel):
    """A summative quiz at the end of a video."""
    questions: list[Question] = Field(min_length=3, max_length=5)


class Scene(BaseModel):
    scene_index: int = Field(ge=0)
    visual_prompt: str = Field(min_length=20, max_length=700)
    generator: Literal["imagen", "veo"]
    audio_dialogue: str = Field(min_length=10, max_length=500)
    on_screen_text: str | None = Field(default=None, max_length=60)
    duration_seconds: float = Field(ge=3.0, le=10.0)
    motion_style: Literal["zoom_in", "zoom_out", "gentle_drift"] = "gentle_drift"
    checkpoint: Question | None = None
    enable_subtitles: bool = True


class VideoPlan(BaseModel):
    title: str = Field(max_length=80)
    tagline: str = Field(max_length=140)
    content_type: Literal["math", "place", "story", "general"]
    style_guide: str = Field(min_length=20, max_length=400)
    recurring_character_description: str | None = Field(default=None, max_length=400)
    voice_recommendation: str = Field(max_length=60)
    scenes: list[Scene] = Field(min_length=3, max_length=6)
    end_quiz: Quiz | None = None

    # --- Domain-aware optional metadata ---
    # Filled by the Director only when the relevant domain overlay is active.
    # Workspaces in the UI surface these to render Asset Kits, Material Boards,
    # and Slide Decks. Other domains leave them None.

    # Branding domain
    brand_palette: list[str] | None = Field(
        default=None,
        description=(
            "3-5 hex color codes (e.g., ['#0A2540', '#FF7A59']) that form the "
            "brand palette. Filled when domain=branding."
        ),
    )
    typography_pairing: str | None = Field(
        default=None,
        max_length=160,
        description=(
            "Recommended typography pairing in the form 'Display / Body' "
            "(e.g., 'Playfair Display / Inter'). Filled when domain=branding."
        ),
    )
    brand_archetype: str | None = Field(
        default=None,
        max_length=80,
        description=(
            "Jungian brand archetype (Hero, Sage, Outlaw, Creator, Caregiver, "
            "Magician, Everyman, Lover, Jester, Ruler, Innocent, Explorer). "
            "Filled when domain=branding."
        ),
    )

    # Interior Design domain
    material_palette: list[str] | None = Field(
        default=None,
        description=(
            "5-8 specific materials in the named-finish vocabulary of interior "
            "design (e.g., ['rift-cut white oak', 'unlacquered brass', 'travertine', "
            "'boucle wool', 'patinated bronze']). Filled when domain=interior_design."
        ),
    )
    lighting_specification: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Specific lighting description with kelvin temperature and fixture "
            "vocabulary (e.g., 'Warm 2700K diffuse, recessed downlights + linear "
            "LED at coves, ambient glow at golden hour'). Filled when "
            "domain=interior_design."
        ),
    )

    # Marketing / Presentation domain
    narrative_thesis: str | None = Field(
        default=None,
        max_length=240,
        description=(
            "One-sentence elevator pitch for the deck — the single argument every "
            "slide reinforces. Filled when domain=marketing."
        ),
    )


# --- Director call ---

def plan_video(
    topic: str,
    audience: str,
    content_type_hint: str | None = None,
    domain: str | None = None,
) -> VideoPlan:
    """Returns a fully-formed multi-scene interactive plan.

    Args:
        topic: what the video is about.
        audience: who it's for (e.g., '6-year-old', 'adult', 'design client').
        content_type_hint: optional content_type override.
        domain: vertical domain overlay — one of 'interior_design', 'branding',
            'marketing', 'teaching', or None / 'general' for the horizontal default.
            The Director's system prompt gets an industry-specific overlay
            appended so output style, motion, voice, and pacing all align with
            the vertical's grammar.

    Resilience: each model in DIRECTOR_MODEL_CHAIN is tried with 3 jittered
    retries before falling through to the next model.
    """
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    system_instructions = _build_system_instructions(domain)

    user_prompt = f"TOPIC: {topic}\nAUDIENCE: {audience}"
    if content_type_hint and content_type_hint != "auto":
        user_prompt += f"\nCONTENT_TYPE_HINT: {content_type_hint}"
    if domain and domain != "general":
        user_prompt += f"\nDOMAIN: {domain}"

    last_error: Exception | None = None
    for model in DIRECTOR_MODEL_CHAIN:
        try:
            logger.info("Director: attempting model %s (domain=%s)", model, domain or "general")
            response = _call_gemini_once(client, model, user_prompt, system_instructions)
            plan = VideoPlan.model_validate_json(response.text)
            if model != config.DIRECTOR_MODEL:
                logger.warning("Director fell back from %s to %s.", config.DIRECTOR_MODEL, model)
            return plan
        except _TRANSIENT_ERRORS as e:
            logger.warning(
                "Director model %s failed after 3 retries (%s). Falling through.",
                model, type(e).__name__,
            )
            last_error = e
            continue

    raise RuntimeError(
        f"All Director models in the fallback chain failed. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


# =============================================================================
# Campaign-mode Manifest Planner
# -----------------------------------------------------------------------------
# A second planner used only when the pipeline runs a dispatched single-asset
# campaign task (Visual Print Shop mode). It produces a manifest — the
# professional spec document that ships alongside the rendered PNG and
# explicitly resolves every guardrail in the campaign's rules.md.
# =============================================================================


class HexColor(BaseModel):
    """One color in the manifest's palette, with rationale and role."""
    token: str = Field(max_length=60, description="Brand token name from rules.md, e.g., 'accent_hyperspeed'.")
    hex_value: str = Field(max_length=9, description="Hex code including the leading '#', e.g., '#F0FF26'.")
    role: str = Field(max_length=120, description="What this color does in the frame.")
    rationale: str = Field(max_length=240, description="Why this hex was chosen here — the math behind the decision.")


class ContrastCheck(BaseModel):
    """One WCAG contrast pairing the manifest verifies."""
    foreground: str = Field(max_length=60)
    background: str = Field(max_length=60)
    computed_ratio: str = Field(max_length=20, description="e.g., '14.8:1'")
    wcag_target: str = Field(max_length=40, description="e.g., 'AA Large (3:1)'")
    verdict: Literal["pass", "fail"]


class LightSource(BaseModel):
    """One named light source with kelvin and angle."""
    name: Literal["key", "fill", "rim", "ambient", "practical"]
    kelvin: int = Field(ge=1800, le=10000)
    angle: str = Field(max_length=120, description="e.g., 'camera-right, 35 degrees above horizon'")
    intensity: str = Field(max_length=80, description="e.g., 'hard', 'soft bounce ~1/3 key', 'ambient overcast'")


class CompositionSpec(BaseModel):
    framing: str = Field(max_length=160, description="e.g., 'rule-of-thirds left, subject upper-left third'")
    lens_mm: int = Field(ge=14, le=200, description="Implied focal length in millimeters.")
    depth_of_field: str = Field(max_length=120, description="e.g., 'shallow, f/2.8 read, foreground droplets sharp'")
    focal_point: str = Field(max_length=160, description="What the eye lands on first.")
    negative_space_percent: int = Field(ge=0, le=100, description="Approximate frame share that is negative space.")


class MaterialSpec(BaseModel):
    name: str = Field(max_length=80, description="e.g., 'engineered knit mesh upper'")
    treatment: str = Field(max_length=200, description="How it should render — texture, sheen, scale.")


class VisualManifest(BaseModel):
    """The professional spec document that ships with every dispatched render.

    The Director populates this from the dispatched task's ``action`` prompt,
    the campaign's ``rules.md`` text, and the Lead Brand Visualizer persona
    in ``role.md``. The pipeline serialises it to ``manifest.md`` alongside
    the PNG.
    """

    campaign: str = Field(max_length=80, description="Campaign profile slug, e.g., 'speed_pro_launch'.")
    task_title: str = Field(max_length=160)
    asset_filename: str = Field(max_length=80, description="The .png filename this manifest accompanies.")

    brief_one_liner: str = Field(
        max_length=240,
        description="The single sentence that captures what this frame must communicate.",
    )

    # --- Director's enhanced visual prompt (what the image model actually receives) ---
    visual_prompt: str = Field(
        min_length=60,
        max_length=1400,
        description=(
            "The full prompt sent to the image generator. Should embed every "
            "compositional, lighting, and material constraint from the manifest."
        ),
    )

    # --- Brand-system resolution ---
    palette: list[HexColor] = Field(min_length=2, max_length=8)
    contrast_checks: list[ContrastCheck] = Field(min_length=1, max_length=6)

    # --- Light + composition + materials ---
    lighting: list[LightSource] = Field(min_length=2, max_length=5)
    composition: CompositionSpec
    materials: list[MaterialSpec] = Field(min_length=1, max_length=8)

    # --- Concept + guardrail resolution ---
    visual_metaphor: str = Field(
        min_length=20,
        max_length=320,
        description="The single idea this frame compresses (e.g., 'the millisecond between push-off and landing').",
    )
    prohibited_elements_resolved: list[str] = Field(
        min_length=1,
        max_length=12,
        description="Each line restates one prohibition from rules.md and how this render satisfies it.",
    )
    rule_resolutions: list[str] = Field(
        min_length=1,
        max_length=12,
        description="Plain-language confirmation of every guardrail in rules.md.",
    )


# --- System instructions for the Manifest Planner ----------------------------

_MANIFEST_SYSTEM_BASE = """You are now operating as the LEAD BRAND VISUALIZER described in the persona section below.

You receive ONE dispatched visual task and a strict set of brand guardrails.
You output ONE structured JSON manifest that:

1. Compresses the brief into a single sharp one-liner.
2. Writes the enhanced visual_prompt that will be sent to the image generator —
   embed every compositional, lighting, and material constraint from the
   manifest so the rendered frame can be defended against the rules.
3. Resolves the brand palette explicitly. Every hex you list must trace to a
   token named in the rules. Do not invent colors.
4. Verifies WCAG contrast for every pairing the rules name. Report the actual
   ratio and the pass/fail verdict.
5. Specifies every light source by kelvin and angle.
6. Specifies composition with lens_mm, depth of field, framing thirds, focal
   point, and negative-space percentage.
7. Lists every material that should render and how it should read.
8. Names the single visual metaphor the frame compresses.
9. For EVERY prohibition in the rules, write one line confirming how this
   render avoids it. Silence is not acceptable — restate and resolve each one.
10. For EVERY rule in the rules.md document, write one line confirming
    compliance.

You are not writing marketing copy. You are writing a defensible spec a
creative director can read in 90 seconds and approve.

Tone: calm, precise, professional. No marketing-speak, no exclamation marks,
no superlatives. Every sentence carries a decision.
"""


def _build_manifest_system_instructions(role_text: str, rules_text: str) -> str:
    """Stitch the persona + guardrails into one system prompt."""
    return (
        _MANIFEST_SYSTEM_BASE
        + "\n\n============================================================\n"
        + "PERSONA (from role.md)\n"
        + "============================================================\n"
        + (role_text or "(role.md was not provided)")
        + "\n\n============================================================\n"
        + "GUARDRAILS (from rules.md) — RESOLVE EVERY RULE IN THE MANIFEST\n"
        + "============================================================\n"
        + (rules_text or "(rules.md was not provided)")
    )


# --- Planner entry point -----------------------------------------------------


def plan_visual_manifest(
    campaign: str,
    task_title: str,
    action_prompt: str,
    role_text: str,
    rules_text: str,
    asset_filename: str = "hero_asset.png",
) -> VisualManifest:
    """Run the Manifest Director on one dispatched campaign task.

    Args:
        campaign: profile slug, used to label the manifest.
        task_title: human-readable task title from tasks.md.
        action_prompt: the ``action`` field from the dispatched task — the
            raw brief the brand visualizer is being asked to render.
        role_text: full text of role.md (Lead Brand Visualizer persona).
        rules_text: full text of rules.md (brand guardrails).
        asset_filename: name of the rendered .png file this manifest will
            sit beside in the run directory.

    Returns:
        A fully-populated ``VisualManifest`` — both consumed by the pipeline
        (to drive the image generator with the enhanced visual_prompt) and
        serialised to ``manifest.md`` for delivery alongside the render.

    Resilience: the same DIRECTOR_MODEL_CHAIN + 3-retry pattern as the video
    Director — Gemini 2.5 Flash → 2.5 Flash Lite → 2.0 Flash.
    """
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    system_instructions = _build_manifest_system_instructions(role_text, rules_text)

    user_prompt = (
        f"CAMPAIGN: {campaign}\n"
        f"TASK_TITLE: {task_title}\n"
        f"ASSET_FILENAME: {asset_filename}\n"
        f"DISPATCHED_ACTION_PROMPT:\n{action_prompt}\n\n"
        f"Produce the manifest now. Resolve every rule explicitly."
    )

    last_error: Exception | None = None
    for model in DIRECTOR_MODEL_CHAIN:
        try:
            logger.info("Manifest Director: attempting model %s", model)
            response = _call_gemini_once(
                client,
                model,
                user_prompt,
                system_instructions=system_instructions,
                response_schema=VisualManifest,
                temperature=0.4,  # tighter — this is a spec, not a story
            )
            manifest = VisualManifest.model_validate_json(response.text)
            if model != config.DIRECTOR_MODEL:
                logger.warning(
                    "Manifest Director fell back from %s to %s.",
                    config.DIRECTOR_MODEL, model,
                )
            return manifest
        except _TRANSIENT_ERRORS as e:
            logger.warning(
                "Manifest Director model %s failed after 3 retries (%s). Falling through.",
                model, type(e).__name__,
            )
            last_error = e
            continue

    raise RuntimeError(
        f"All Manifest Director models in the fallback chain failed. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


# --- Markdown serialisation --------------------------------------------------


def render_manifest_md(manifest: VisualManifest) -> str:
    """Serialise a VisualManifest into the .md spec document delivered to clients."""

    def _table_palette() -> str:
        rows = ["| Token | Hex | Role | Rationale |", "|---|---|---|---|"]
        for c in manifest.palette:
            rows.append(f"| `{c.token}` | `{c.hex_value}` | {c.role} | {c.rationale} |")
        return "\n".join(rows)

    def _table_contrast() -> str:
        rows = [
            "| Foreground | Background | Computed Ratio | WCAG Target | Verdict |",
            "|---|---|---|---|---|",
        ]
        for c in manifest.contrast_checks:
            verdict = "✅ pass" if c.verdict == "pass" else "❌ fail"
            rows.append(
                f"| `{c.foreground}` | `{c.background}` | **{c.computed_ratio}** "
                f"| {c.wcag_target} | {verdict} |"
            )
        return "\n".join(rows)

    def _table_lights() -> str:
        rows = [
            "| Source | Kelvin | Angle | Intensity |",
            "|---|---|---|---|",
        ]
        for l in manifest.lighting:
            rows.append(f"| {l.name} | {l.kelvin}K | {l.angle} | {l.intensity} |")
        return "\n".join(rows)

    def _materials_list() -> str:
        return "\n".join(f"- **{m.name}** — {m.treatment}" for m in manifest.materials)

    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    return f"""# Campaign Manifest — {manifest.task_title}

**Campaign:** `{manifest.campaign}`
**Asset:** `{manifest.asset_filename}`

> {manifest.brief_one_liner}

---

## 1. Visual Metaphor

{manifest.visual_metaphor}

## 2. Director's Visual Prompt

> {manifest.visual_prompt}

## 3. Brand Palette

{_table_palette()}

## 4. WCAG Contrast Verification

{_table_contrast()}

## 5. Lighting Specification

{_table_lights()}

## 6. Composition

- **Framing:** {manifest.composition.framing}
- **Lens:** {manifest.composition.lens_mm}mm
- **Depth of field:** {manifest.composition.depth_of_field}
- **Focal point:** {manifest.composition.focal_point}
- **Negative space:** {manifest.composition.negative_space_percent}% of frame

## 7. Material Specification

{_materials_list()}

## 8. Prohibited Elements — Resolved

{_bullets(manifest.prohibited_elements_resolved)}

## 9. Rule-by-Rule Compliance

{_bullets(manifest.rule_resolutions)}

---

*This manifest is the contract. The render is the proof.*
"""
